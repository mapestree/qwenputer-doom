"""Qwenputer decode benchmark: python bench.py MODE   (eager | chipc | chipc+tg | exec | exec+tg)"""
import sys, json, time, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from chipformer.rv import bridge as BR, elf, ref
from chipformer.rv.chat import RvChat
import fast_text, paths
mode = sys.argv[1]
chat = RvChat(paths.MODEL, executor=mode.startswith("exec"))
if "chipc" in mode: fast_text.compile_chip(chat)
if mode.endswith("+tg"): fast_text.install(chat)
D = "tests/data/rv/"
def calc(max_new=600):
    m = ref.Machine.from_program(elf.program(D + "calc-bridge.elf")); BR.park(m); m2 = ref.Machine.from_program(elf.program(D + "calc-bridge.elf")); BR.park(m2)
    ids = chat.prompt_ids([{"role": "user", "content": "Use calc: what is 123456789 * 987654321?"}])
    lock = BR.Session(chat.T, m2); lock.history(ids)
    t = time.time(); r = chat.generate(ids, m, max_new, check_head=True, lock=lock); r["wall"] = time.time() - t; return r
def raw(name, max_chip):
    m = ref.Machine.from_program(elf.program(D + name)); m2 = ref.Machine.from_program(elf.program(D + name))
    ids = chat.prompt_ids([{"role": "user", "content": "run"}], tools=())
    lock = BR.Session(chat.T, m2); lock.history(ids)
    t = time.time(); r = chat.generate(ids, m, 300, max_chip=max_chip, check_head=True, lock=lock, first=chat.T.TCC); r["wall"] = time.time() - t; return r
calc(); raw("c-crc32p.elf", 50000)   # warm-up: compile / graph capture
rows = {}
for name, fn in [("calc", lambda: calc()), ("c-gmpmul", lambda: raw("c-gmpmul.elf", 400000))]:
    rs = [fn() for _ in range(3)]
    r = sorted(rs, key=lambda r: r["wall"])[1]   # median of 3 by wall time
    rows[name] = {"chip_tok": r["chip_tokens"], "chip_tok_s": round(r["chip_tokens"] / max(r["chip_seconds"], 1e-9)),
                  "text_tok": r["text_tokens"], "text_ms_tok": round(1e3 * r["text_seconds"] / max(r["text_tokens"], 1), 1),
                  "wall_s": round(r["wall"], 2), "lock_error": r["lock_error"], "head_mismatch": r["head_mismatch"]}
print("BENCH", mode, json.dumps(rows))
