"""Local voice sidecar for the Gemma playground.

Endpoints:
  POST /stt    multipart audio upload -> {"text": "..."}   (NVIDIA Parakeet via MLX)
  POST /tts    {"text": "..."}        -> WAV audio bytes   (Resemble Chatterbox)
  GET  /health server + model-load status

Models are loaded lazily on first use and cached. Heavy imports live inside
the engine classes so the module stays importable without torch/mlx (tests
mock the engines via FastAPI dependency overrides).

Voice cloning: drop a short reference clip at voices/default.wav to give
Chatterbox a custom voice, or pass {"voice": "<name>.wav"} per request to
use voices/<name>.wav.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

VOICES_DIR = Path(__file__).resolve().parent / "voices"
STT_MODEL_ID = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "")
HOST = os.environ.get("VOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("VOICE_PORT", "8090"))


class SpeechToText:
    """Parakeet TDT 0.6B running on Apple Silicon via MLX.

    MLX binds its GPU stream to the thread it runs on, and FastAPI serves
    sync endpoints from a threadpool — so all model work (loading and
    inference) is pinned to one dedicated thread. The single worker also
    serializes concurrent requests.
    """

    def __init__(self, model_id: str = STT_MODEL_ID):
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = self._executor.submit(self._load, model_id).result()

    @staticmethod
    def _load(model_id: str):
        from parakeet_mlx import from_pretrained

        return from_pretrained(model_id)

    def transcribe(self, audio_path: str) -> str:
        result = self._executor.submit(self._model.transcribe, audio_path).result()
        return result.text.strip()


class TextToSpeech:
    """Chatterbox TTS on MPS (Apple Silicon) or CPU."""

    def __init__(self, device: str = TTS_DEVICE):
        import torch

        if not device:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Chatterbox checkpoints were saved on CUDA; remap tensors while loading.
        original_load = torch.load

        def load_remapped(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device(device))
            return original_load(*args, **kwargs)

        torch.load = load_remapped
        try:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=device)
        finally:
            torch.load = original_load
        self._lock = threading.Lock()

    def synthesize(
        self,
        text: str,
        voice_path: Optional[Path] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> bytes:
        import torchaudio

        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
        if voice_path is not None:
            kwargs["audio_prompt_path"] = str(voice_path)
        with self._lock:
            wav = self._model.generate(text, **kwargs)
        buf = io.BytesIO()
        torchaudio.save(buf, wav.cpu(), self._model.sr, format="wav")
        return buf.getvalue()


_stt: Optional[SpeechToText] = None
_tts: Optional[TextToSpeech] = None
_stt_lock = threading.Lock()
_tts_lock = threading.Lock()


def get_stt() -> SpeechToText:
    global _stt
    with _stt_lock:
        if _stt is None:
            _stt = SpeechToText()
    return _stt


def get_tts() -> TextToSpeech:
    global _tts
    with _tts_lock:
        if _tts is None:
            _tts = TextToSpeech()
    return _tts


def clean_text_for_speech(text: str) -> str:
    """Strip markdown so the TTS engine reads prose, not formatting."""
    text = re.sub(r"```[\s\S]*?```", " ", text)              # code blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)                 # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)        # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)     # links -> label
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)       # headers
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)          # bold
    text = re.sub(r"(?<!\w)(\*|_)(?!\s)(.*?)(?<!\s)\1(?!\w)", r"\2", text)  # italics
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)     # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)     # numbered lists
    return re.sub(r"\s+", " ", text).strip()


def resolve_voice(voice: Optional[str]) -> Optional[Path]:
    """Map a requested voice name to a file in voices/, defaulting to default.wav."""
    if voice:
        candidate = VOICES_DIR / Path(voice).name  # basename only: no traversal
        if not candidate.is_file():
            raise HTTPException(404, f"Voice not found: {Path(voice).name}")
        return candidate
    default = VOICES_DIR / "default.wav"
    return default if default.is_file() else None


def slugify_voice_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]


def convert_to_wav(src_path: str, dest: Path) -> None:
    """Re-encode any browser-recorded clip (webm/m4a/wav) to mono WAV for Chatterbox."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "24000", "-t", "30", str(dest)],
        capture_output=True,
    )
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Could not decode the recording as audio")


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: Optional[str] = None
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)


app = FastAPI(title="Gemma Playground Voice Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "stt_loaded": _stt is not None, "tts_loaded": _tts is not None}


@app.post("/stt")
def stt(audio: UploadFile = File(...), engine: SpeechToText = Depends(get_stt)):
    data = audio.file.read()
    if not data:
        raise HTTPException(400, "Empty audio upload")
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        text = engine.transcribe(tmp.name)
    finally:
        os.unlink(tmp.name)
    return {"text": text}


@app.post("/tts")
def tts(req: TTSRequest, engine: TextToSpeech = Depends(get_tts)):
    text = clean_text_for_speech(req.text)
    if not text:
        raise HTTPException(400, "No speakable text after stripping markdown")
    voice_path = resolve_voice(req.voice)
    wav = engine.synthesize(
        text,
        voice_path=voice_path,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
    )
    return Response(content=wav, media_type="audio/wav")


@app.get("/voices")
def list_voices():
    VOICES_DIR.mkdir(exist_ok=True)
    return {"voices": sorted(p.name for p in VOICES_DIR.glob("*.wav"))}


@app.post("/voices")
def create_voice(name: str = Form(...), audio: UploadFile = File(...)):
    slug = slugify_voice_name(name)
    if not slug:
        raise HTTPException(400, "Voice name needs at least one letter or number")
    data = audio.file.read()
    if not data:
        raise HTTPException(400, "Empty audio upload")
    VOICES_DIR.mkdir(exist_ok=True)
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        dest = VOICES_DIR / f"{slug}.wav"
        convert_to_wav(tmp.name, dest)
    finally:
        os.unlink(tmp.name)
    return {"voice": dest.name}


@app.delete("/voices/{name}")
def delete_voice(name: str):
    target = VOICES_DIR / Path(name).name
    if target.suffix != ".wav" or not target.is_file():
        raise HTTPException(404, "Voice not found")
    target.unlink()
    return {"deleted": target.name}


if __name__ == "__main__":
    import uvicorn

    if os.environ.get("VOICE_PRELOAD", "1") != "0":
        print("Preloading Parakeet (STT)...")
        get_stt()
        print("Preloading Chatterbox (TTS)...")
        get_tts()
        print("Models ready.")
    uvicorn.run(app, host=HOST, port=PORT)
