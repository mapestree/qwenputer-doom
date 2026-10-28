"""Play the raycaster on the reference machine through the bridge (no GPU): per turn, instructions,
chip tokens, echoed characters, and the frame.   cwd: upstream/experimental/riscv
   uv run python ../../../toolchain/play_ref.py ELF [keys keys ...]"""
import os, subprocess, sys
from pathlib import Path
from chipformer.rv import bridge as BR, elf, ref

def syms(path):  # global/static data symbols -> address (nm in the toolchain image)
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{path.rsplit('/', 1)[0]}:/e", "qwenputer-rv",
                          "riscv64-unknown-elf-nm", "/e/" + path.rsplit("/", 1)[1]], capture_output=True, text=True).stdout
    return {l.split()[2]: int(l.split()[0], 16) for l in out.splitlines() if len(l.split()) == 3}

def topdown(m, sym, mapfile, r=8):
    """Debug view (host side, reads RAM): the map around the player, '^>v<' = heading."""
    word = lambda a: (lambda v: v - (1 << 32) if v >> 31 else v)(m.ram[a >> 2])
    px, py, ang = word(sym["px"]) / 65536, word(sym["py"]) / 65536, word(sym["ang"]) % 256
    rows = [l.split('"')[1] for l in open(mapfile) if l.startswith('    "')]
    c0, r0 = int(px), int(py)
    head = ">v<^"[((ang + 32) % 256) // 64]
    out = []
    for y in range(r0 - r // 2, r0 + r // 2 + 1):
        out.append("".join(head if (x, y) == (c0, r0) else rows[y][x] if 0 <= y < len(rows) and 0 <= x < len(rows[0])
                           else " " for x in range(c0 - r, c0 + r + 1)))
    return f"pos ({px:.2f}, {py:.2f}) ang {ang}\n" + "\n".join(out)

T = BR.Text(Path(os.environ.get("QWENPUTER_MODEL", Path(__file__).resolve().parent.parent / "models/Qwen3.5-4B-Qwenputer")) / "tokenizer.json")
mapfile = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--map=")), None)
sys.argv = [a for a in sys.argv if not a.startswith("--map=")]
sym = syms(sys.argv[1]) if mapfile else None
m = ref.Machine.from_program(elf.program(sys.argv[1])); BR.park(m)
s = BR.Session(T, m)
for keys in [""] + sys.argv[2:]:
    call = f"<tool_call>\n<function=move>\n<parameter=keys>\n{keys or '.'}\n</parameter>\n</function>\n</tool_call>"
    steps0, tok0 = m.steps, len(s.toks)
    s.qwen(T.encode(call))
    echo = s.runs[-1]
    instr, chip = m.steps - steps0, sum(1 for t in s.toks[tok0:] if not (t & BR.S.TEXT_TOKEN))
    print(f"--- keys={keys!r}: {instr} instr, {chip} chip tokens, {len(echo)} echoed tokens")
    frame = T.decode_bytes(echo).decode("utf-8", "replace").split("\n")
    if mapfile:
        td = topdown(m, sym, mapfile).split("\n")
        frame = [f"{a:40s}   {b}" for a, b in zip(frame + [""] * len(td), td + [""] * len(frame))]
    print("\n".join(frame))
