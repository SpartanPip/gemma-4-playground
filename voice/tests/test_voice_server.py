"""Tests for the voice sidecar (Parakeet STT + Chatterbox TTS).

Unit tests swap the real engines for fakes via FastAPI dependency overrides,
so they run in milliseconds with no model weights. The integration tests at
the bottom (pytest -m integration) load the real models and round-trip
audio: Chatterbox speaks a phrase, Parakeet transcribes it back.
"""
import io
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import voice_server
from voice_server import (
    app,
    clean_text_for_speech,
    get_stt,
    get_tts,
    resolve_voice,
    slugify_voice_name,
)


def make_wav_bytes(seconds=0.1, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{int(rate * seconds)}h", *([0] * int(rate * seconds))))
    return buf.getvalue()


class FakeSTT:
    def __init__(self):
        self.paths = []

    def transcribe(self, audio_path):
        self.paths.append(audio_path)
        return "hello world"


class FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice_path=None, exaggeration=0.5, cfg_weight=0.5):
        self.calls.append(
            {"text": text, "voice_path": voice_path,
             "exaggeration": exaggeration, "cfg_weight": cfg_weight}
        )
        return make_wav_bytes()


@pytest.fixture
def fake_stt():
    return FakeSTT()


@pytest.fixture
def fake_tts():
    return FakeTTS()


@pytest.fixture
def client(fake_stt, fake_tts):
    app.dependency_overrides[get_stt] = lambda: fake_stt
    app.dependency_overrides[get_tts] = lambda: fake_tts
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---- /health ----

def test_health_reports_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "stt_loaded", "tts_loaded"}


# ---- /stt ----

def test_stt_returns_transcript(client, fake_stt):
    res = client.post("/stt", files={"audio": ("clip.webm", b"fake-audio-bytes", "audio/webm")})
    assert res.status_code == 200
    assert res.json() == {"text": "hello world"}
    assert len(fake_stt.paths) == 1


def test_stt_temp_file_uses_upload_extension_and_is_cleaned_up(client, fake_stt):
    client.post("/stt", files={"audio": ("clip.m4a", b"fake", "audio/mp4")})
    path = Path(fake_stt.paths[0])
    assert path.suffix == ".m4a"
    assert not path.exists()


def test_stt_rejects_empty_upload(client):
    res = client.post("/stt", files={"audio": ("clip.webm", b"", "audio/webm")})
    assert res.status_code == 400


def test_stt_requires_audio_field(client):
    res = client.post("/stt")
    assert res.status_code == 422


# ---- /tts ----

def test_tts_returns_wav_audio(client):
    res = client.post("/tts", json={"text": "Hello there."})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content[:4] == b"RIFF"


def test_tts_strips_markdown_before_synthesis(client, fake_tts):
    client.post("/tts", json={"text": "**Bold** and `code` and [a link](https://x.com)."})
    assert fake_tts.calls[0]["text"] == "Bold and code and a link."


def test_tts_passes_generation_parameters(client, fake_tts):
    client.post("/tts", json={"text": "Hi.", "exaggeration": 1.2, "cfg_weight": 0.3})
    call = fake_tts.calls[0]
    assert call["exaggeration"] == 1.2
    assert call["cfg_weight"] == 0.3


def test_tts_rejects_text_that_is_only_markdown(client, fake_tts):
    res = client.post("/tts", json={"text": "```\nprint('hi')\n```"})
    assert res.status_code == 400
    assert fake_tts.calls == []


def test_tts_rejects_missing_and_oversized_text(client):
    assert client.post("/tts", json={}).status_code == 422
    assert client.post("/tts", json={"text": ""}).status_code == 422
    assert client.post("/tts", json={"text": "x" * 2001}).status_code == 422


def test_tts_unknown_voice_returns_404(client):
    res = client.post("/tts", json={"text": "Hi.", "voice": "nope.wav"})
    assert res.status_code == 404


def test_tts_uses_named_voice_clip(client, fake_tts, tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "VOICES_DIR", tmp_path)
    (tmp_path / "me.wav").write_bytes(make_wav_bytes())
    res = client.post("/tts", json={"text": "Hi.", "voice": "me.wav"})
    assert res.status_code == 200
    assert fake_tts.calls[0]["voice_path"] == tmp_path / "me.wav"


# ---- voice resolution ----

def test_resolve_voice_blocks_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "VOICES_DIR", tmp_path)
    (tmp_path / "me.wav").write_bytes(b"x")
    assert resolve_voice("../../me.wav") == tmp_path / "me.wav"
    with pytest.raises(Exception) as exc:
        resolve_voice("../../../etc/passwd")
    assert exc.value.status_code == 404


def test_resolve_voice_defaults_to_default_wav_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "VOICES_DIR", tmp_path)
    assert resolve_voice(None) is None
    (tmp_path / "default.wav").write_bytes(b"x")
    assert resolve_voice(None) == tmp_path / "default.wav"


# ---- markdown cleaning ----

@pytest.mark.parametrize("raw,expected", [
    ("Plain sentence.", "Plain sentence."),
    ("**bold** _italic_ *starred*", "bold italic starred"),
    ("Use `pip install` here", "Use pip install here"),
    ("```python\ncode\n```\nAfter.", "After."),
    ("# Header\nBody text", "Header Body text"),
    ("See [the docs](https://example.com) now", "See the docs now"),
    ("![alt text](img.png) caption", "caption"),
    ("- item one\n- item two", "item one item two"),
    ("1. first\n2. second", "first second"),
    ("multi   space\n\nand newlines", "multi space and newlines"),
    ("2 * 3 = 6 and 4 * 5 = 20", "2 * 3 = 6 and 4 * 5 = 20"),
])
def test_clean_text_for_speech(raw, expected):
    assert clean_text_for_speech(raw) == expected


# ---- voice library (/voices) ----

@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "VOICES_DIR", tmp_path)
    return tmp_path


def test_list_voices_empty_and_sorted(client, voices_dir):
    assert client.get("/voices").json() == {"voices": []}
    (voices_dir / "zoe.wav").write_bytes(b"x")
    (voices_dir / "amy.wav").write_bytes(b"x")
    (voices_dir / "notes.txt").write_bytes(b"x")  # non-wav files are ignored
    assert client.get("/voices").json() == {"voices": ["amy.wav", "zoe.wav"]}


def test_create_voice_converts_upload_to_wav(client, voices_dir):
    res = client.post(
        "/voices",
        data={"name": "My Voice!"},
        files={"audio": ("clip.wav", make_wav_bytes(seconds=0.5), "audio/wav")},
    )
    assert res.status_code == 200
    assert res.json() == {"voice": "my-voice.wav"}
    saved = voices_dir / "my-voice.wav"
    assert saved.is_file()
    assert saved.read_bytes()[:4] == b"RIFF"
    assert client.get("/voices").json() == {"voices": ["my-voice.wav"]}


def test_create_voice_rejects_bad_input(client, voices_dir):
    wav = make_wav_bytes()
    assert client.post("/voices", data={"name": "!!!"},
                       files={"audio": ("c.wav", wav, "audio/wav")}).status_code == 400
    assert client.post("/voices", data={"name": "ok"},
                       files={"audio": ("c.wav", b"", "audio/wav")}).status_code == 400
    assert client.post("/voices", data={"name": "ok"},
                       files={"audio": ("c.wav", b"not audio at all", "audio/wav")}).status_code == 400
    assert client.post("/voices", files={"audio": ("c.wav", wav, "audio/wav")}).status_code == 422
    assert list(voices_dir.glob("*.wav")) == []


def test_delete_voice(client, voices_dir):
    (voices_dir / "me.wav").write_bytes(b"x")
    assert client.delete("/voices/me.wav").status_code == 200
    assert not (voices_dir / "me.wav").exists()
    assert client.delete("/voices/me.wav").status_code == 404


def test_delete_voice_only_touches_wav_files_in_voices_dir(client, voices_dir):
    (voices_dir / "keep.txt").write_bytes(b"x")
    assert client.delete("/voices/keep.txt").status_code == 404
    assert client.delete("/voices/..%2Fkeep.txt").status_code == 404
    assert (voices_dir / "keep.txt").exists()


@pytest.mark.parametrize("raw,expected", [
    ("Phillip", "phillip"),
    ("My Cool Voice!", "my-cool-voice"),
    ("  spaces  ", "spaces"),
    ("../../etc/passwd", "etc-passwd"),
    ("###", ""),
    ("x" * 100, "x" * 40),
])
def test_slugify_voice_name(raw, expected):
    assert slugify_voice_name(raw) == expected


# ---- integration: real models, real audio (pytest -m integration) ----

@pytest.mark.integration
def test_real_tts_stt_round_trip(tmp_path):
    tts = get_tts()
    wav = tts.synthesize("Testing one two three.")
    assert wav[:4] == b"RIFF"
    assert len(wav) > 10000  # more than a fraction of a second of audio

    clip = tmp_path / "round_trip.wav"
    clip.write_bytes(wav)
    stt = get_stt()
    text = stt.transcribe(str(clip)).lower()
    for word in ("testing", "one", "two", "three"):
        assert word in text, f"expected {word!r} in transcript: {text!r}"
