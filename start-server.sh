#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/models/google_gemma-4-E4B-it-Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at: $MODEL_PATH"
  echo "Download it with:"
  echo "  huggingface-cli download bartowski/google_gemma-4-E4B-it-GGUF \\"
  echo "    --include 'google_gemma-4-E4B-it-Q4_K_M.gguf' \\"
  echo "    --local-dir ./models"
  exit 1
fi

VOICE_PYTHON="$SCRIPT_DIR/voice/.venv/bin/python"
if [ -x "$VOICE_PYTHON" ]; then
  echo "Starting voice server (Parakeet STT + Chatterbox TTS) on 127.0.0.1:8090 ..."
  "$VOICE_PYTHON" "$SCRIPT_DIR/voice/voice_server.py" &
  VOICE_PID=$!
  trap 'kill "$VOICE_PID" 2>/dev/null || true' EXIT
else
  echo "Voice server not set up — run ./voice/setup.sh to enable speech. Continuing text-only."
fi

echo "Starting Gemma 4 E4B (Q4_K_M) on localhost:8080 ..."
echo "Chat UI: http://127.0.0.1:8080"
echo ""

llama-server \
  -m "$MODEL_PATH" \
  --chat-template gemma \
  --no-jinja \
  --host 127.0.0.1 \
  --port 8080 \
  --path "$SCRIPT_DIR" \
  -ngl 99 \
  -c 8192
