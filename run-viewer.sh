#!/usr/bin/env bash
# Start the Qwenputer viewer (calc demo at /, E1M1 at /game): http://<host>:7890
# Extra args go to app/server.py (--port N, --model DIR, --no-compile, --no-text-graph).
# Weights: $QWENPUTER_MODEL, default models/Qwen3.5-4B-Qwenputer (./setup.sh downloads them).
# Set QWENPUTER_GPU_WRAP to run under a GPU scheduler, e.g. QWENPUTER_GPU_WRAP="my_broker --gpus 1 --".
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/upstream/experimental/riscv"      # the firmware/source paths in server.py are relative to here
exec ${QWENPUTER_GPU_WRAP:-} uv run python "$HERE/app/server.py" --port 7890 "$@"
