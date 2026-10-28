"""Headless E1M1 turns through app/game.py on the real weights: python game_test.py [keys ...]
   (cwd: upstream/experimental/riscv; default turns: "", "www", "dddddddd")."""
import queue, sys, threading, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from chipformer.rv.chat import RvChat
import fast_text, game, paths
chat = RvChat(paths.MODEL)
fast_text.compile_chip(chat); fast_text.install(chat)
g = game.Game(chat, threading.Lock())
for keys in sys.argv[1:] or ["", "www", "dddddddd"]:
    q = queue.Queue(); g.play(keys, q)
    while (ev := q.get()) is not None:
        if ev[0] in ("done", "error"):
            d = ev[1]
            if ev[0] == "error": print("ERROR", d); sys.exit(1)
            print(f"--- {keys!r}: clean={d['clean']} instr={d['instructions']} chip={d['chip_tokens']} text={d['text_tokens']} "
                  f"{d['seconds']:.1f}s (chip {d['chip_seconds']} text {d['text_seconds']}) head_mismatch={d['head_mismatch']} "
                  f"lock={d['lock_error']}", flush=True)
            print(d["frame"], flush=True)
print("status", {k: v for k, v in g.status().items() if k != "frame"})
