# Could this machine run Doom?

Short answer: not as built. Machine numbers are from the repo (`spec.py`, `rv32.md`); the Doom-size figures are my
estimates from memory unless a source is linked. Speed numbers are ours (see performance.md).

## Our machine's limits
- RAM 32 KB (2 sublayers x 16 KB; one sublayer = one layer). DROM 128 KB read-only, loads only. ROM 256 KB code.
- J/R tokens carry 16-bit word indices (caps code at 256 KB); value tokens 15 bits. No framebuffer: output is `putchar`.
- ~1.55 chip tokens/instruction; ~770 instr/s on the weights path (1,220 tok/s), ~2,300 instr/s on the executor.

## First answer (naive Doom, my estimate): no
RAM 2-4 MB -> ~128-256 extra layers; WAD ~4 MB -> ~32 data layers; 10-30M instructions/frame -> hours per frame.

## Revised after research: how small has Doom been squeezed?
- **Clamp meter, 32 KB RAM** ([CircuitDigest](https://circuitdigest.com/news/doom-on-a-30-fnirsi-clamp-meter-32-kb-of-ram-no-framebuffer-and-a-steady-29-fps),
  [DMC100-Doom](https://github.com/ArturBieniek4/DMC100-Doom), GPL-2.0, Artur Bieniek): Artery AT32F415, 32 KB RAM,
  99 KB (101,376 B) program flash, 4 MB SPI NOR for game data. Streams map data from NOR, loads code overlays into RAM,
  strip renderer (no framebuffer), low detail (160 columns drawn twice). E1M1-E1M9, title, menu, 4 save slots; drops automap
  and Nightmare. ~29 fps avg on shareware DEMO1 (24 fps Freedoom DEMO1). README gives no CPU clock.
- **Doom64KB / Sega Genesis** ([Generation Amiga](https://www.generationamiga.com/2026/06/28/someone-made-doom-run-on-sega-genesis-and-the-hardware-is-probably-crying/)):
  64 KB RAM, 38x28 grid of 8x8 solid tiles, only E1M1 and E1M8, no screen wipe / fuzz.
- RP2040: 256 KB RAM, compressed 4 MB WAD in 2 MB flash ([CNX](https://www.cnx-software.com/2022/03/15/doom-ported-to-raspberry-pi-rp2040/)).
- nRF5340: 512 KB RAM; packed structs, LUTs, 16-bit vars ([Nordic](https://devzone.nordicsemi.com/nordic/nordic-blog/b/blog/posts/doom-on-nrf5340)).
- Bare-metal RV32 ports exist: [doom-riscv](https://git.knazarov.com/knazarov/doom-riscv), [doom_riscv](https://github.com/smunaut/doom_riscv) (RAM not checked).

## Revised verdict
| Constraint | Status |
|---|---|
| RAM 32 KB | Not the blocker in principle (clamp-meter port has the same size, using overlays + streaming + strip renderer) |
| Code | ~99 KB fits in 256 KB ROM |
| WAD (~4 MB) | **Blocker**: our data ROM is 128 KB, no streaming. Needs ~32 data layers OR a streaming path through the token bridge (CPU asks for a chunk, host feeds it back as text tokens, like putchar echoes) |
| Speed | Several orders of magnitude short even for the stripped-down build; instructions/frame for that build unknown |

## Feasible stepping stone
A ~20 KB ASCII raycaster printing frames via `putchar`: fits all limits today; a frame at ~100k instructions (guess) would
take about a minute on the weights path. Unverified: whether a RISC-V gcc is installed here (the repo builds in WSL via
`env/build_*.sh`; the repo ships prebuilt ELFs only).

## Built 2026-10-28: toolchain + raycaster, measured on the reference machine (no GPU yet)
- Toolchain: `toolchain/Dockerfile` (Debian bookworm: `gcc-riscv64-unknown-elf`, `picolibc-riscv64-unknown-elf`,
  qemu-user) -> image `qwenputer-rv`; `toolchain/rvcc OUT.elf file.c` uses the same flags as upstream `build_tests.sh`.
  Check: `c-sort` rebuilt from source gives the identical chip-token trace to the shipped ELF (8,132 instr).
- Firmware: `firmware/raycast.c`, a bridge program like calc: each `</tool_call>` -> reads keys (w/s, a/d turn, q/e strafe)
  from the text buffer, moves, prints a 40x12 frame in `<tool_response>`, parks. Player state stays in RAM between turns.
  No `<think></think>` after the response (calc emits one; it would switch Qwen's thinking off).
  Size: 2.5 KB code, 0.8 KB rodata, 0.2 KB RAM. Map is a 24x16 placeholder, **not E1M1**.
- `toolchain/play_ref.py ELF keys...` plays turns through `bridge.Session` (the lock reference). Per turn: 12.5-15.7k instr,
  16-20k chip tokens (~1.27/instr), 300-350 echoed chars. Profile: ~60% raycast, ~35% draw/putchar.
- Estimated wall time per turn (from performance.md rates, not measured): weights path ~14 s chip + ~6 s echo = ~20 s;
  executor ~5 s + ~6 s = ~11 s.

## E1M1 from doom1.wad (2026-10-28, reference machine only)
- WAD: Debian non-free `doom-wad-shareware` 1.9.fixed-4 -> `wad-src/doom1.wad`, md5 f0cefca49926d00903cf57551d901abe
  (shareware v1.9). Shareware licence: redistribution only unmodified, so `wad-src/` and the generated `build/map*.h`
  (derived geometry) stay local; do not publish them.
- `tools/wad2grid.py WAD E1M1 CELL OUT.h`: cell centres -> sector (+x ray to the nearest linedef, side test), flood fill
  from the player start with Doom's limits (step up <= 24, headroom >= 56, doors/lifts pass), unreached = wall, exit
  switch (special 11) -> `X` with `E` floor in front. E1M1: 475 linedefs, 85 sectors. At 32 units/cell: 146x91 cells,
  13 KB of DROM (limit 128 KB); 4 door cells (always open). The 64-unit grid (74x47) was too coarse: the exit-switch alcove vanished.
  Kept: floor plan. Dropped: heights, textures, light, monsters/items, nukage damage, the secret areas the flood can't reach.
- Firmware: `toolchain/rvcc build/e1m1.elf firmware/raycast.c -DMAPH='"/w/build/map32.h"'`; one `w` = one cell (32 units)
  in two collision-checked half steps; a/d = 11.25 degrees; walls drawn 2 cells tall.
- `toolchain/walkthrough.py`: shortest path start -> exit (121 cells), one call per straight run: **exit reached in 28 turns**.
  Per turn avg 17.5k instr, 22k chip tokens, 409 echoed chars, 27 tool-call tokens; whole run 489k instr, 619k chip tokens.
- Estimated playthrough time (rates from performance.md, not measured): weights path ~620k/1,220 = ~8.5 min chip + ~12.3k text
  positions x 17.8 ms = ~3.7 min -> **~12 min, ~26 s per turn**; executor ~6.5 min.
- For step 3 (GPU): the whole run is ~12k text positions, past the 4096-position static KV buffer of `fast_text.install`
  (falls back to eager ~50 ms/token after ~8 turns). Raise MAXLEN or reset the conversation each turn (RAM persists in the machine).

## Playable in the viewer (2026-10-28, real weights)
- `/game` (`app/game.py`, `app/game.html`): a person queues keys, each turn = one short conversation: the game's machine
  state is injected (as RvChat does for any firmware), the keys are forced in as a `move` tool call
  (`generate(force=..., stop_after_halt=True)`), the circuit runs the firmware and prints the frame. Lock-checked; the game
  only advances on a clean run, carrying the reference machine's state (which the stream just matched) to the next turn.
  Why not one long conversation: `inject` zeroes the text buffer after prefill, and generate is one-shot from a prompt;
  per-turn conversations also stay well inside the 4096-position graphed text step (~335 prompt + ~430 per turn).
- The game prompt uses the chat template's default (thinking on), not `prompt_ids` (which passes enable_thinking=False);
  Qwen is never sampled in this mode.
- Measured on the 5090 (weights path, compiled chip layers + graphed text): first frame 33.6 s (26.6k instr, 33k chip tok),
  `www` 34.9 s, `dddddddd` 25.7 s; ~1,210 chip tok/s; 0 lock errors, 0 head mismatches; frames identical to the reference.
  A full playthrough (28 turns) should take ~12-15 min of GPU time (estimate from the walkthrough's token counts).
- Rebuild after changing the firmware or map: `python3 -I tools/wad2grid.py wad-src/doom1.wad E1M1 32 build/map32.h` then
  `toolchain/rvcc build/e1m1.elf firmware/raycast.c -DMAPH='"/w/build/map32.h"'`, restart the viewer.
