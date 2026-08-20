#!/bin/sh
# Seed the babeldoc asset cache (fonts + onnx models, baked at build time)
# into the /data volume, then start the API.
set -e
mkdir -p "${DATA_DIR:-/data}"
if [ ! -d "$HOME/.cache/babeldoc" ] && [ -d /opt/babeldoc-cache/.cache/babeldoc ]; then
    mkdir -p "$HOME/.cache"
    cp -r /opt/babeldoc-cache/.cache/babeldoc "$HOME/.cache/babeldoc"
fi
exec uvicorn server.app:app --host 0.0.0.0 --port 8000
