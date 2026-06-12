#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg is required (audio decoding for Parakeet): brew install ffmpeg"
  exit 1
fi

if command -v uv >/dev/null; then
  uv venv --python 3.11 "$SCRIPT_DIR/.venv"
  # pkuseg (a chatterbox-tts dependency) needs numpy at build time,
  # so install build prerequisites first and skip build isolation.
  # setuptools<81: perth (chatterbox watermarker) still imports pkg_resources.
  uv pip install --python "$SCRIPT_DIR/.venv/bin/python" numpy cython "setuptools<81" wheel
  uv pip install --python "$SCRIPT_DIR/.venv/bin/python" --no-build-isolation -r "$SCRIPT_DIR/requirements.txt"
else
  python3.11 -m venv "$SCRIPT_DIR/.venv"
  "$SCRIPT_DIR/.venv/bin/pip" install numpy cython "setuptools<81" wheel
  "$SCRIPT_DIR/.venv/bin/pip" install --no-build-isolation -r "$SCRIPT_DIR/requirements.txt"
fi

echo ""
echo "Voice setup complete. Models download from Hugging Face on first run."
echo "Start everything with ../start-server.sh"
