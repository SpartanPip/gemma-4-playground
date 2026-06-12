# Gemma 4 E4B Local Chat Playground

Chat with Google's Gemma 4 E4B model running 100% locally on your Mac. No cloud, no API keys, no internet required after setup.

## Prerequisites

- macOS with Apple Silicon (M1+)
- [Homebrew](https://brew.sh)

## Setup

### 1. Install llama.cpp

```bash
brew install llama.cpp
```

### 2. Download the model

```bash
pip3 install huggingface_hub

huggingface-cli download bartowski/google_gemma-4-E4B-it-GGUF \
  --include "google_gemma-4-E4B-it-Q4_K_M.gguf" \
  --local-dir ./models
```

This downloads the Q4_K_M quantized model (~5 GB).

## Usage

### Start the server

```bash
./start-server.sh
```

This launches llama-server on `http://127.0.0.1:8080` with Metal GPU acceleration.

### Open the chat UI

Open `index.html` in your browser, or serve it:

```bash
npx serve .
```

The chat UI connects to the local llama-server and streams responses in real time.

## Configuration

Click the gear icon in the chat UI to adjust:

- **System prompt** — set the assistant's behavior
- **Temperature** — control randomness (0 = deterministic, 2 = creative)
- **Max tokens** — limit response length
- **API endpoint** — change the server URL if needed
- **Voice server** — URL of the speech sidecar (default `http://127.0.0.1:8090`)

## Voice mode

Talk to Gemma and have it talk back — 100% local, like everything else:

- **Speech-to-text**: [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) running on the Apple Neural Engine/GPU via MLX
- **Text-to-speech**: [Resemble Chatterbox](https://github.com/resemble-ai/chatterbox) running on MPS

### Setup (one time)

```bash
brew install ffmpeg   # audio decoding for Parakeet
./voice/setup.sh      # creates voice/.venv and installs dependencies
```

Model weights (~4 GB) download from Hugging Face automatically on first launch.

### Usage

`./start-server.sh` now launches the voice server alongside llama-server.
In the chat UI:

- Click the **🎤 mic button** (or it appears next to send), speak, click again to stop — your words are transcribed and sent automatically
- Replies are **spoken aloud sentence-by-sentence** as they stream in
- Toggle the **🔊 speaker icon** in the header to mute/unmute voice replies

### Voice cloning

Chatterbox does zero-shot voice cloning from ~10 seconds of audio. The
**Voices sidebar** in the chat UI manages a voice library:

- Click **Record new voice**, name it, speak naturally for ~10 seconds, then
  click **Stop & save** — the clip is converted to WAV and stored in `voice/voices/`
- Click any saved voice to select it; all replies are then spoken in that voice
- **Built-in** switches back to Chatterbox's stock voice
- Hover a voice and click **×** to delete it

Only clone your own voice or one you have permission to use. You can also drop
WAV files into `voice/voices/` by hand — they appear in the sidebar.

### Voice API

The sidecar runs at `http://127.0.0.1:8090`:

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/stt` | POST | multipart `audio` file (webm/m4a/wav) | `{"text": "..."}` |
| `/tts` | POST | `{"text": "...", "voice"?, "exaggeration"?, "cfg_weight"?}` | WAV audio |
| `/voices` | GET | — | `{"voices": ["name.wav", ...]}` |
| `/voices` | POST | multipart `name` + `audio` clip | `{"voice": "name.wav"}` |
| `/voices/{name}` | DELETE | — | `{"deleted": "name.wav"}` |
| `/health` | GET | — | server + model-load status |

### Tests

```bash
cd voice
.venv/bin/python -m pytest                  # unit tests (fast, mocked engines)
.venv/bin/python -m pytest -m integration   # real-model round trip: TTS → STT
```
