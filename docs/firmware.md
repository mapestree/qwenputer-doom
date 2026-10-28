# Firmware inventory (`upstream/experimental/riscv/tests/data/rv/`)

Instruction counts are from the repo's `docs/rv32.md` unless marked (observed). We ran only what is marked (observed).

| Image | What it is | Instructions | In viewer? |
|---|---|---|---|
| `calc-bridge.elf` | bignum calculator driven by Qwen's tool call (`env/calc/calc.c`). The only one with a text bridge | 1,570 (observed) | yes (`calc`) |
| `c-crc32p` / `c-crc32` | CRC32 of "The quick brown fox..." (printf / raw) | ~4-6k | `c-crc32p` |
| `c-gmpmul` | mini-gmp: multiply two 20-digit numbers, prints `mul=1219326311370217952249809480012498094790` | 14,767 (observed) | yes |
| `c-sieve` / `c-sieve4` | primes below 2,000 / 10,000 (1229) | 127k (10,000 version) | `c-sieve` only |
| `c-sort`, `c-strops` | small sort / string tests | small | yes |
| `c-fib` | recursive fib(25) = 75025 | 2.1M | no (too slow) |
| `c-qsort` | sort 1,000 ints | 257k | no |
| `c-sudoku` | easy sudoku | 8.4M (verified against reference, never free-run through the model) | no |
| `mpy-s1.elf` | MicroPython: `print(2**100)` -> 1267650600228229401496703205376 | 39,888 (observed) | yes |
| `mpy-s5.elf` | MicroPython: 18-line prime/word-count script | ~490k | no |
| `rv32ui-*`, `rv32um-*`, `rv32uzba-*` | official RISC-V conformance tests, one image per instruction | tiny | no |
| `*.qemu.elf` | the same code built for qemu (cross-check) | | no |

Status caveat from the repo docs: MicroPython "has run on the standalone core and executor, not yet inside the merged
model; it needs a bridge build". We ran `mpy-s1` through the real weights anyway (forced `</tool_call>` -> `GO`), exact.

## Starting a standalone program (our workaround, not the repo's demo)
Machines are NOT parked (`Machine.from_program`, entry point); prompt = chat template with user "run" and `tools=()`;
`generate(..., first=chat.T.TCC)` forces `</tool_call>`, the circuit emits `GO`, the ELF runs from `_start`.
Needs `patches/upstream-chat-force.patch` (`generate(first=...)`). Afterwards Qwen adds a stray "I'm ready to help!".

## Adding a program to the viewer
Add it to `FIRMWARES` in `app/server.py` (name -> elf, label, source file for the source panel, bridge?). Keep it under
~50k instructions: events are streamed as JSON and every event is kept client-side.
