"""Play E1M1 start -> exit on the reference machine: shortest grid path, one call per straight run
(turn keys + w's). Reports turns, totals, and whether the firmware printed the exit message.
   cwd: upstream/experimental/riscv;  uv run python ../../../toolchain/walkthrough.py ELF MAP.h [max_run]"""
import os, sys
from pathlib import Path
from collections import deque
from chipformer.rv import bridge as BR, elf, ref

elf_path, mapfile = sys.argv[1], sys.argv[2]
max_run = int(sys.argv[3]) if len(sys.argv) > 3 else 10
rows = [l.split('"')[1] for l in open(mapfile) if l.startswith('    "')]
find = lambda ch: next((c, r) for r, row in enumerate(rows) for c, x in enumerate(row) if x == ch)
start, ang = find("P"), int(next(l.split()[2] for l in open(mapfile) if "START_ANG" in l))
prev, q = {start: None}, deque([start])
while q:
    c, r = q.popleft()
    if rows[r][c] == "E":
        goal = (c, r); break
    for dc, dr in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        n = (c + dc, r + dr)
        if n not in prev and rows[n[1]][n[0]] in ".DEP":
            prev[n] = (c, r); q.append(n)
path = [goal]
while prev[path[-1]]: path.append(prev[path[-1]])
path.reverse()
DIR = {(1, 0): 0, (0, 1): 64, (-1, 0): 128, (0, -1): 192}
calls, i = [], 0
while i < len(path) - 1:
    d = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]); n = 0
    while i < len(path) - 1 and (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) == d and n < max_run:
        i += 1; n += 1
    turn = (DIR[d] - ang) % 256; ang = DIR[d]
    calls.append(("d" * (turn // 8) if turn <= 128 else "a" * ((256 - turn) // 8)) + "w" * n)
print(f"path {len(path) - 1} cells, {len(calls)} calls")

T = BR.Text(Path(os.environ.get("QWENPUTER_MODEL", Path(__file__).resolve().parent.parent / "models/Qwen3.5-4B-Qwenputer")) / "tokenizer.json")
m = ref.Machine.from_program(elf.program(elf_path)); BR.park(m)
s = BR.Session(T, m)
tot = dict(instr=0, chip=0, echo=0, text=0)
for k, keys in enumerate([""] + calls):
    call = T.encode(f"<tool_call>\n<function=move>\n<parameter=keys>\n{keys or '.'}\n</parameter>\n</function>\n</tool_call>")
    st, tk = m.steps, len(s.toks)
    s.qwen(call)
    tot["instr"] += m.steps - st; tot["text"] += len(call); tot["echo"] += len(s.runs[-1])
    tot["chip"] += sum(1 for t in s.toks[tk:] if not t & BR.S.TEXT_TOKEN)
    out = T.decode_bytes(s.runs[-1]).decode("utf-8", "replace")
    if "EXIT REACHED" in out:
        print(f"exit reached after call {k}: {keys!r}"); break
else:
    print("exit NOT reached; last frame:\n" + out)
n = k + 1
print(f"{n} turns; per turn avg: {tot['instr'] // n} instr, {tot['chip'] // n} chip tok, {tot['echo'] // n} echo, "
      f"{tot['text'] // n} tool-call tokens; totals {tot}")
print("max tool-call tokens in one call:", max(len(T.encode(c)) for c in calls), "(capture limit 80 incl. the call syntax)")
