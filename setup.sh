#!/usr/bin/env bash
# One-time setup:  ./setup.sh [path/to/doom1.wad]
#   1. upstream qwenputer at the pinned commit + our patch, its uv env (+ flash-linear-attention)
#   2. the RISC-V toolchain image (Docker)
#   3. the Qwenputer weights (18 GB) into $QWENPUTER_MODEL (default models/Qwen3.5-4B-Qwenputer), unless present
#   4. E1M1 -> build/map32.h -> build/e1m1.elf, from YOUR doom1.wad (not shipped; toolchain/get-shareware-wad.sh fetches one)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
UPSTREAM=https://github.com/divagr18/qwenputer
COMMIT=ca13310e2f9f788d1dd3feeea8213c1b0c23913c
PATCH=patches/upstream-chat-force.patch
MODEL=${QWENPUTER_MODEL:-$ROOT/models/Qwen3.5-4B-Qwenputer}
WAD=${1:-wad-src/doom1.wad}

echo "== upstream @ ${COMMIT:0:7}"
[ -d upstream/.git ] || git clone "$UPSTREAM" upstream
git -C upstream checkout -q "$COMMIT" 2>/dev/null || { git -C upstream fetch -q origin && git -C upstream checkout -q "$COMMIT"; }
if git -C upstream apply --reverse --check "$ROOT/$PATCH" 2>/dev/null; then echo "   patch already applied"
else git -C upstream apply "$ROOT/$PATCH"; echo "   patch applied"; fi
(cd upstream/experimental/riscv && uv sync && uv pip install flash-linear-attention==0.5.2)  # fla: ~25% faster chip decode

echo "== toolchain image"
docker build -q -t qwenputer-rv toolchain

echo "== weights -> $MODEL"
if [ -f "$MODEL/chipformer.json" ]; then echo "   present"
else uvx --from huggingface_hub hf download divagr1925/Qwen3.5-4B-Qwenputer --local-dir "$MODEL"; fi

echo "== E1M1 from $WAD"
if [ ! -f "$WAD" ]; then
  echo "   no WAD: run toolchain/get-shareware-wad.sh (or pass the path to your doom1.wad), then ./setup.sh again"; exit 0
fi
mkdir -p build
python3 -I tools/wad2grid.py "$WAD" E1M1 32 build/map32.h
toolchain/rvcc build/e1m1.elf firmware/raycast.c -DMAPH='"/w/build/map32.h"'
echo "done: ./run-viewer.sh, then open http://localhost:7890/game"
