# How Qwenputer works

Sources: the model card, `upstream/experimental/riscv/docs/rv32.md`, `chipformer/rv/{spec,bridge,chat,ref}.py`, and
what we observed in the viewer. Anything marked (observed) we saw ourselves.

## The idea
Qwen3.5-4B with **11 extra layers in front of Qwen's own 32**. The extra layers' weights are not trained: a compiler
computes them so they implement an RV32IM CPU. The CPU's registers, RAM and program counter live in the Gated-DeltaNet
recurrent state (delta rule = overwrite one row exactly, so a row is a memory cell). Qwen's own weights are unchanged
(KL vs base 6e-4 per the card). Model facts: hidden 3968 (Qwen 2560 + the computer's dims), vocab = Qwen's + 230,324
"chip" tokens, 18.8 GB bf16 mostly zero padding, ~1M nonzero weights in the computer.

## Who does what for a `calc` request (observed)
1. Qwen writes a normal tool call: `<tool_call><function=calc><parameter=expression>123456789 * 987654321 ...`.
   Tokens between `<tool_call>` and `</tool_call>` are captured into a text buffer in machine memory (0x8000, 16 B/token).
2. At `</tool_call>` the circuit predicts `GO` and the machine runs from where the firmware was parked.
3. The firmware is `calc.c` (~600 lines of C, a bignum calculator), **pre-written and pre-compiled** to RISC-V
   (`tests/data/rv/calc-bridge.elf`, ~3,300 instructions). No model wrote it. The **chat driver (`RvChat`) injects the
   firmware into the DeltaNet state after the prompt prefill**; the weights contain the CPU, not the program.
4. Each executed instruction is one *record* of chip tokens (observed: 1,570 instructions -> 2,430 chip tokens, ~1.55
   per instruction). Qwen's layers never see chip tokens (they are transparent to its cache and RoPE positions).
5. `putchar` outputs are echoed to Qwen as ordinary text tokens; `exit` parks the machine. Qwen reads the printed answer
   and writes the final sentence.
Qwen decides to call the tool and writes the expression; the arithmetic is exact because the firmware computes it.

## Chip-token records (`spec.py`)
One record per instruction; PC is implied (`PC = ROM_BASE + 4*(anchor + n)`).
- `V` 15-bit value chunk (not final); `F0/F1/F2` final value token (1, 2 or 3 chunks); `J` jump (carries the new anchor);
  `R` anchor refresh (every 63 instructions, nothing executes); `NT` no-write (not-taken branch, write to x0).
- `ST0/ST1` word store; `SS0/SS1` merged word after byte/half store; `BYTE` putchar; `HALT` exit; `GO` start of run.
- Bignum "digit file" (8 registers x 128 decimal digits, custom ops BMAC/BMSB/BLD/BGET/BSCAN/BPRD): `BD` = one digit
  (+carry) per row step; `SC` = one ASCII digit found. d7 holds the constant 1 in calc.
- Divide: `QS` + `QD`... quotient digits. Faults: `FLT` + 8 `HEX` + `HALT`.

## Memory map and capacity (the reason Doom doesn't fit; see doom.md)
RAM 32 KB at 0 (2 sublayers of 16 KB), DROM 128 KB at 0x20000 (read-only data, loads only, 1 sublayer), ROM 256 KB at
0x40000 (code, pre-decoded 64 KB/sublayer, 4 sublayers). Each memory sublayer is one layer, so capacity scales with depth.
Core depth is 11 layers (the dependency chain), 99 heads, ~42k MLP units.

## Exactness
Every circuit-owned token is checked against a reference machine (`bridge.Session`, "lock"). All our runs: 0 lock errors,
0 head mismatches, and our shadow tracer in `app/vm_trace.py` matched every streamed record (0 trace mismatches).
The repo claims bf16 bit-exactness on an L4 with fla kernels; we did not independently verify at bit level.

## Two ways to run the chip layers
- **Weights path (default, what the viewer uses):** the actual HF modules (via a compacted CUDA-graph view).
- **Executor (`RvChat(executor=True)`):** `tritonexec.Executor2` runs the circuit's fp32 arithmetic - same maths, NOT the
  weights. ~3x faster. Useful as a reference/upper bound; we kept the demo on the real weights.
