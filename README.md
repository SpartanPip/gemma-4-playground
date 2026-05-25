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
