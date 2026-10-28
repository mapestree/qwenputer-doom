# Performance work and benchmark (2026-10-28)

Hardware: RTX 5090 (one broker-leased card), AMD EPYC 7702P host. The repo's numbers are from an NVIDIA L4.
Reproduce: `app/bench.py MODE` on one GPU from `upstream/experimental/riscv`. Medians of 3 runs, lock check on.

## Diagnosis (inspect, don't reason)
- Chip-token decode is **GPU launch-bound, not Python-bound**: CUDA graph replay was 1.94 ms/token (~1,100 tiny kernels,
  ~1.7 us each: norms, copies, fills, adds); Python overhead ~5%.
- Qwen text step: ~6,200 kernel launches/token, 50 ms wall vs 26.5 ms of GPU time -> CPU/launch-bound.

## Changes (all in `app/fast_text.py` + one pip install)
1. `uv pip install flash-linear-attention==0.5.2` (the version the repo's GPU verification used): 485 -> 630 chip tok/s.
2. `compile_chip(chat)`: `torch.compile` the 11 chip layers used by the decode CUDA graph, `dynamic=False`, with
   `torch._dynamo.config.cache_size_limit=64` (at the default 8 only some layers compile: 918 tok/s instead of 1,220).
   Prefill keeps the eager layers (different sequence length per prompt would recompile). ~10 s warm-up at start.
3. `install(chat)`: Qwen text step as a CUDA graph. Static KV buffer (MAXLEN 4096) for the 8 full-attention layers written
   by `index_copy_` at a device-side position; mask built in-graph; linear layers keep their in-place states; each
   conversation's prefill cache is copied into the graph's persistent state. Beyond 4096 positions it falls back to eager.

## Results (end to end, our decode path)
| Config | chip tok/s | text ms/tok | calc | c-gmpmul |
|---|---|---|---|---|
| Original (measured before fla) | 485 | ~50 | 11.7 s | 51 s |
| + fla | 630 | 50 | 10.3 s | 39.7 s |
| + compiled chip layers | 1,220 | 50 | 8.5 s | 22.6 s |
| + graphed text step (current viewer) | 1,220 | 17.8 | 4.8 s | 20.6 s |
| Executor reference (fp32 circuit, not the weights) + graphed text | 3,500 | 17.5 | 2.9 s | 7.9 s |
Viewer `mpy-s1` (39,888 instr): 109 s -> ~44 s.

## Versus the repo's numbers (repo docs/rv32.md, L4)
| Metric | Repo | Ours |
|---|---|---|
| chip tok/s, HF/weights path | ~250 | 1,220 (485 before our changes) |
| chip tok/s, executor | 3,200 | 3,500 |
| Qwen text step, batch 1 | ~68 ms | 17.8 ms |
| executor batch 1 (compiled graph) | 2,706 tok/s, 0.37 ms | 2,505 tok/s, 0.40 ms |
| executor batch 8 | 13,367 tok/s, 0.60 ms | 9,471 tok/s, 0.85 ms (~30% behind; cause not investigated) |
| executor batch 32 | 37,743 tok/s, 0.85 ms | 35,548 tok/s, 0.90 ms |
| dense HF chip layers, eager (`timing`, synthetic hidden 3072) | 11.0 ms/token | 17.9 ms/token (host-CPU launch overhead) |
Full output of the repo's script on our box: `../logs/repo-timing-script-5090.txt`. Not run: the repo's L40S estimate.

## Exactness
Circuit side exact in every config (0 lock errors, 0 head mismatches, 0 tracer mismatches). The graphed text step is NOT
bit-identical to eager: 4/5 prompts identical (all 3 tool-call prompts + a long-number one); a free-form joke flipped one
word at token 14 (near tie under the masked-SDPA kernel). The repo documents the same effect for its static-cache batch path.

## Next levers (untested)
- `lm_head` reads ~478k rows per text step (Qwen vocab + 230k chip tokens); only the Qwen slice is needed.
- `causal_conv1d` is the torch fallback (needs nvcc, not installed); fla has Triton conv kernels that could be patched in.
- Investigate the executor batch-8 gap; persistent single-kernel chip step (the repo's own "next").
- Verify bit-exactness of the compiled chip layers with the repo's `gpucheck`, not just free-running lock checks.
