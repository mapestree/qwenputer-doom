# Doom E1M1 on a CPU compiled into an LLM's weights

```
@@@@@@@@@@@@
@@@@@@@@@@@@##
@@@@@@@@@@@@###
@@@@@@@@@@@@####
@@@@@@@@@@@@####%                     %%
@@@@@@@@@@@@####%%   ++++++********%%%%%
@@@@@@@@@@@@####%%,,,++++++********%%%%%
@@@@@@@@@@@@####%,,,,,,,,,,,,,,,,,,,,,%%
@@@@@@@@@@@@####,,,,,,,,,,,,,,,,,,,,,,,,
@@@@@@@@@@@@###.........................
@@@@@@@@@@@@##..........................
@@@@@@@@@@@@............................
```

That frame was computed by a 32-bit RISC-V CPU that exists only as **weights inside a language model**. The CPU executed
17,540 instructions to draw it, and each one came out of the model as tokens checked against a reference
machine. You play in a browser, one turn at a time: your keys go into the conversation as a tool call, the model's CPU
layers run a raycaster over the floor plan of Doom's first level (E1M1), and the next frame prints as text.

It is built on **[Qwenputer](https://github.com/divagr18/qwenputer)** by divagr18, which compiles a CPU into Qwen3.5-4B.
We didn't change the model. We wrote firmware for it, converted the level, and added a way for a person to play.

> This is E1M1's **floor plan** explored in first person, not the Doom engine. See
> [what it is and isn't](#what-it-is-and-isnt).

## How a CPU lives inside a transformer

Qwenputer is Qwen3.5-4B with **11 extra layers placed in front of Qwen's own 32**. Nobody trained those 11 layers:
a compiler calculated their weights so that they implement an RV32IM + Zba CPU. Three ideas make it work.
(Full details: [docs/how-it-works.md](docs/how-it-works.md) and upstream's
[rv32.md](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/docs/rv32.md).)

**Memory is recurrent state.** Qwen3.5 uses Gated-DeltaNet layers, whose recurrent state is a matrix. The delta rule can
overwrite one row of that matrix exactly, so each row works as a memory cell. Registers, the 32 KB of RAM and the ROM
holding the program are all rows of state in the extra layers. The weights contain only the CPU; the driver writes the
program into that state when a conversation starts.

**Execution is decoding.** The model's own next-token predictions *are* the CPU's execution trace. Each executed
instruction produces one *record* of "chip tokens" from a 230k-token extra vocabulary:

- the value written to the destination register, in 15-bit chunks (`V` … `F0`/`F1`/`F2`);
- `NT` for a branch not taken; `J` for a jump, which carries the new program counter;
- `ST` for a store; `BYTE` for `putchar`; `HALT` for `exit`.

Feeding each predicted token back in advances the machine by one step. The program counter is never written out: it is
implied by counting records since the last jump.

**Two kinds of token, two owners.** Qwen predicts ordinary text. From `</tool_call>` until `HALT`, the circuit predicts
instead. Qwen's layers treat chip tokens as transparent (they never enter its KV cache or RoPE positions), but each
`putchar` is followed by an **echo**: the printed byte as an ordinary text token. That's how a program's output
reaches the conversation, and why Qwen can read the answer.

**Lock check.** While the model decodes, the host steps upstream's reference RV32 machine in parallel and compares every
token. The first token that differs stops generation. We only advance the game after a turn with zero differences.

## One turn, token by token

Here is a real turn: the first one of a fresh game, with keys `wwd`. You type the keys; the page forces this tool call
into the conversation:

```
<tool_call> \n < function = move > \n < parameter = keys > \n ww d \n </ parameter > \n </ function > \n </tool_call>
```

Every token between `<tool_call>` and `</tool_call>` is copied into a text buffer at RAM address `0x8000`, 16 bytes per
row. At `</tool_call>` the circuit predicts `GO`, and the CPU resumes from where it was parked. Here are its first
instructions, with the chip tokens each one produced (disassembly from our shadow decoder):

```
  #  pc       instruction            chip tokens   what it is
  1  0x401f8  lw a5, 424(zero)       F0(0)         if (!started) ...  (loads 0)
  2  0x401fc  bne a5, zero, ...      NT            not taken: run init()
  3  0x40200  lui a5, 0x3a8          V(0) F1(117)  px = 58.5 cells in 16.16 (0x3a8000 = 117 * 32768)
  4  0x40204  sw a5, 436(zero)       ST0
  5  0x40208  lui a5, 0x328          V(0) F1(101)  py = 50.5 cells
  ...
  7  0x40210  li a5, 192             F0(192)       heading 192/256 = facing north
 10  0x4021c  lui a5, 0x8            V(0) F1(1)    a5 = 0x8000: the text buffer
 16  0x40234  lbu a4, -16(a1)        F0(10)        first captured byte: '\n'
 ...
987  0x402b4  ecall                  BYTE(10)      putchar('\n'), then its echo '\n' as a text token
 ...
25487  0x401f4  ecall                HALT          exit: park until the next </tool_call>
```

In between, the CPU casts 40 rays and prints about 400 bytes: the frame inside `<tool_response>`. After the `HALT`
echo, the turn ends.

## What we had to work with

| Resource | Size | What it meant for Doom |
|---|---|---|
| RAM | 32 KB | fine for a raycaster; real Doom needs MBs (even a 32 KB clamp-meter port streams its data from 4 MB of flash) |
| Code ROM | 256 KB | fine |
| Read-only data | 128 KB | the real WAD is 4 MB, so it can't come in whole |
| Input | the last tool call's text, up to 80 tokens | enough for a string of keys |
| Output | `putchar` only | no framebuffer, so frames are ASCII. Every echoed byte is a text token that Qwen's 32 layers must process: **one full Qwen forward pass (~18 ms) per byte printed** |
| Speed | ~1,220 chip tokens/s ≈ 770 instructions/s (RTX 5090, after our speedups) | ~15–30k instructions per frame is affordable; Doom's millions per frame are not |

So no Doom engine and no WAD in memory. What fits is a **Wolfenstein-style grid raycaster** with E1M1's floor plan as
its map, played one turn at a time.

## How we got Doom into the machine

### 1. A toolchain we could trust
Upstream ships prebuilt ELFs built in WSL. We put `gcc-riscv64-unknown-elf` and picolibc in a Debian container
([toolchain/Dockerfile](toolchain/Dockerfile)) with upstream's exact flags, runtime and linker script. Check: rebuilding
upstream's `c-sort` from source gives a chip-token trace **identical** to the shipped ELF's (8,132 instructions, same
hash).

### 2. Firmware that waits between turns ([firmware/raycast.c](firmware/raycast.c))
We copied the pattern from upstream's calculator firmware:
- **Park, don't exit.** `main` is a loop that starts with the `exit` ecall. The machine halts there, and the next `GO`
  resumes right after it with registers and RAM intact. So the player's position just lives in `static` variables.
- **Read the keys** from the text buffer: everything after the second `>` up to the next `<`. In the trace above that's
  the tokens `ww` and `d`.
- **Cast 40 rays** with a textbook DDA in 16.16 fixed point. Every instruction costs exactly one record, so even a
  hardware `DIV` per ray is cheap. The trap is a *64-bit* divide: libgcc's `__udivdi3` is a loop of hundreds of
  instructions. So the ray step `|1/ray|` is computed as `(1<<30) / (|r| >> 2)` instead.
- **Print a 40×12 frame** shaded by distance (`@#%*+=-:`), and trim trailing spaces, because every byte costs ~18 ms.
- It's small: 2.6 KB of code and 0.2 KB of RAM; the map takes 13.8 KB of the 128 KB read-only data.

### 3. E1M1 → grid ([tools/wad2grid.py](tools/wad2grid.py))
E1M1 is 475 walls (linedefs) and 85 sectors, at arbitrary angles and heights; a grid raycaster needs square cells. The
converter:
1. finds the sector under each cell's centre: cast a ray in +x to the nearest wall and check which side of it the point
   is on (the back of a one-sided wall is the void);
2. flood-fills from the player start using Doom's movement rules (step up ≤ 24 units, headroom ≥ 56, doors and lifts
   pass); everything unreachable becomes wall;
3. marks the walls touching the exit switch's floor (linedef special 11) as `X`; reaching that floor ends the level.

At 32 map units per cell, E1M1 becomes 146×91 cells. (At 64 units per cell the exit switch's alcove disappeared.) The
grid is compiled in as a C header; the WAD itself never enters the machine.

### 4. Letting a person drive ([app/game.py](app/game.py))
In upstream's demo, Qwen decides to call the tool. Here the player does, which needed two things:
- **Forced tokens.** You can't put the keys in the prompt. The driver loads the machine snapshot *after* the prompt is
  processed, and that load clears every circuit state slot the snapshot doesn't set, including the text buffer. So we
  added `force=[tokens]` to `RvChat.generate`: the tool call is fed in token by token as if Qwen had written it, and
  `stop_after_halt` ends generation after the frame. That's our only change to upstream
  ([patches/upstream-chat-force.patch](patches/upstream-chat-force.patch), 10 lines).
- **One turn = one short conversation.** `generate` always starts from a fresh prompt, so each turn loads the machine
  state left by the previous turn, just as the driver loads any program. That state comes from the reference machine,
  which the circuit's output has just matched token for token; a turn with any lock error doesn't count. Each
  conversation is short (~335 prompt tokens + ~430 per turn), so it stays inside our CUDA-graphed text step.

Qwen's own text is never sampled in this mode: every text token is either your forced call or the circuit's frame.

### 5. Making it fast enough to play
Chip decoding started at 485 tokens/s on our 5090. Profiling showed it was limited by kernel launches (~1,100 tiny
kernels per token), not by Python. Three fixes ([docs/performance.md](docs/performance.md)):
1. install `flash-linear-attention==0.5.2`: 630 tok/s;
2. `torch.compile` the 11 decode-time chip layers, with dynamo's cache limit raised from 8 to 64 (at 8, only some layers
   compile): 1,220 tok/s;
3. capture Qwen's text step as a CUDA graph (static KV buffer, mask built in-graph): 50 → 17.8 ms per text token.

## Numbers

Measured on one RTX 5090, real weights (not upstream's fp32 "executor"):

| Turn | Instructions | Chip tokens | Wall time (chip + text) | Lock |
|---|---|---|---|---|
| first frame | 26,631 | 33,159 | 33.6 s (27.3 + 6.0) | 0 errors, 0 head mismatches |
| `www` | 27,255 | 33,949 | 34.9 s (28.0 + 6.5) | 0 / 0 |
| `dddddddd` (turn 90°) | 17,570 | 22,160 | 25.7 s (18.3 + 7.2) | 0 / 0 |

70–80% of a turn is the CPU running (chip tokens); the rest is Qwen's layers processing the ~400 echoed bytes. Open
rooms cost more than corridors, because rays travel further before hitting a wall. Each frame was byte-identical to the
reference machine's.

A scripted shortest-path run on the reference machine reaches the exit in **28 turns** (121 cells; 489k instructions,
619k chip tokens). At the measured rates that's **~12–15 min of GPU time for the level**. That's an estimate: we haven't
timed a full playthrough on the GPU.

## What it is and isn't
- **Is:** E1M1's real floor plan from `doom1.wad`, explored in first person. Every frame is computed by the CPU in the
  model's weights and checked token for token.
- **Isn't:** the Doom engine. No heights, textures, lighting, monsters, items or damage. Doors are always open, and areas
  the flood fill can't reach (some secrets) are solid wall.
- **Host's part:** between turns, the host carries the machine state into the next conversation, the same way upstream's
  driver loads any program. Within a turn, all of the computation is the model's.
- **Qwen's part:** Qwen's 32 layers process every text token (your keys and every echoed byte), but in this mode their
  predictions are never used. In upstream's calculator demo, by contrast, Qwen decides to call the tool and reads the
  answer.

## Run it

You need an NVIDIA GPU with room for the 18 GB bf16 model plus caches (we used a 32 GB RTX 5090), Docker, `uv`, and about
25 GB of disk.

```bash
./toolchain/get-shareware-wad.sh   # or put your own doom1.wad in wad-src/
./setup.sh                         # upstream @ ca13310 + patch, uv env, toolchain image, weights, E1M1 -> build/e1m1.elf
./run-viewer.sh                    # ~40 s to load; then open http://localhost:7890/game
```

`get-shareware-wad.sh` fetches the shareware v1.9 WAD from Debian's non-free archive (`doom-wad-shareware`). This repo
contains **no Doom data**. `wad-src/`, the generated map header and the built ELF are git-ignored, because the shareware
licence only allows redistributing the WAD unmodified.

The weights go to `models/Qwen3.5-4B-Qwenputer` by default; set `QWENPUTER_MODEL` to use an existing copy. On a shared
box, set `QWENPUTER_GPU_WRAP` to run under your GPU scheduler. Upstream's calculator demo, with a live per-instruction
trace, is at `/`.

No GPU? The reference machine runs the same firmware in pure Python:

```bash
cd upstream/experimental/riscv
uv run python ../../../toolchain/play_ref.py ../../../build/e1m1.elf --map=../../../build/map32.h www dddddddd ww
uv run python ../../../toolchain/walkthrough.py ../../../build/e1m1.elf ../../../build/map32.h   # start -> exit
```

## Repo map

| Path | What |
|---|---|
| `firmware/raycast.c` | the game firmware (bridge protocol, raycaster, input parsing) |
| `tools/wad.py`, `tools/wad2grid.py` | WAD reader; Doom map → grid header |
| `toolchain/` | Docker RISC-V gcc (`rvcc`), WAD fetcher, reference-machine runners (`play_ref.py`, `walkthrough.py`, `refrun.py`) |
| `app/server.py`, `app/game.py`, `app/game.html` | SSE server (loads the model once), the game loop, the game page |
| `app/index.html`, `app/vm_trace.py` | upstream-demo viewer with a per-instruction trace decoded by a shadow machine |
| `app/fast_text.py`, `app/bench.py` | decode speedups and their benchmark |
| `patches/upstream-chat-force.patch` | the only change to upstream: forced tokens + stop after HALT |
| `docs/` | how Qwenputer works, firmware inventory, performance, the Doom feasibility notes as they happened |

## Credits and legal
- Qwenputer (the model, compiler, reference machine and driver) is by **divagr18**, MIT:
  [github](https://github.com/divagr18/qwenputer), [model](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer).
- Qwen3.5-4B is by the Qwen team.
- DOOM is a trademark of id Software / ZeniMax Media. This is an unaffiliated fan experiment; it ships no id Software
  data and needs your own copy of `doom1.wad`.
- Code in this repo: MIT ([LICENSE](LICENSE)).
