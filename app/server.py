"""Qwenputer live viewer: loads the model once, streams every generated token (text vs CPU "chip" token) over SSE.

    cd upstream/experimental/riscv && uv run python ../../../app/server.py --port 7890   (or ./run-viewer.sh)
"""
import argparse, json, queue, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chipformer.rv import bridge as BR, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.chat import RvChat
import paths, vm_trace

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(paths.MODEL))
ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
ap.add_argument("--port", type=int, default=7890)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--no-text-graph", action="store_true", help="eager Qwen text step")
ap.add_argument("--no-compile", action="store_true", help="skip torch.compile of the decode-time chip layers")
args = ap.parse_args()

chat = RvChat(args.model)
import fast_text
if not args.no_compile:
    fast_text.compile_chip(chat)       # chip layers: 485 -> 1218 tok/s (c-gmpmul), exact
if not args.no_text_graph:
    fast_text.install(chat)            # Qwen text step as a CUDA graph: ~50 -> ~19 ms/token
ENV = Path("chipformer/rv/env")
D = Path("tests/data/rv")
# name -> (elf, label, source-for-viewer, bridge?)  bridge = calc style (Qwen writes a tool call); else a standalone program
FIRMWARES = {
    "calc": (D / "calc-bridge.elf", "calc: Qwen's tool call -> bignum calculator (~1.6k instr)", ENV / "calc/calc.c", True),
    "c-crc32p": (D / "c-crc32p.elf", "crc32 of a sentence via printf (~5k instr)", ENV / "progs/crc32p.c", False),
    "c-gmpmul": (D / "c-gmpmul.elf", "mini-gmp: multiply two 20-digit numbers (~16k instr)", ENV / "progs/gmpmul.c", False),
    "c-sieve": (D / "c-sieve.elf", "sieve of Eratosthenes below 2000", ENV / "progs/sieve.c", False),
    "c-sort": (D / "c-sort.elf", "sort 64 pseudo-random numbers", ENV / "progs/sort.c", False),
    "c-strops": (D / "c-strops.elf", "string operations", ENV / "progs/strops.c", False),
    "mpy-s1": (D / "mpy-s1.elf", "MicroPython: print(2**100) (~40k instr, ~2 min)", ENV / "mpy/scripts/s1.py", False),
}
_programs = {}
def program_json(name):
    if name not in _programs:
        elf_path, _, src_path, bridge = FIRMWARES[name]
        p = vm_trace.load_program(elf_path)
        p["src"] = vm_trace.c_sources(src_path) if bridge else {"*": src_path.read_text()}
        p["bridge"] = bridge
        _programs[name] = json.dumps(p)
    return _programs[name]
gpu_lock = threading.Lock()  # one generation at a time (CUDA graphs are single-conversation)
import game
GAME = game.Game(chat, gpu_lock) if game.ELF.exists() else None  # E1M1 (build it with toolchain/rvcc; see docs/doom.md)


def run(prompt, max_new, q, fw="calc"):
    elf_path, _, _, bridge = FIRMWARES[fw]
    def tok(i):
        if i >= chat.v0:
            return ["c", S.name(i - chat.v0)]
        return ["t", chat.tok.decode([i], skip_special_tokens=False)]
    buf, evs, last, tracer = [], [], time.time(), None
    def flush(force=False):
        nonlocal buf, evs, last
        if buf and (force or time.time() - last > 0.05):
            q.put(("tokens", {"tok": buf, "ev": evs})); buf, evs, last = [], [], time.time()
    try:
        with gpu_lock:
            m = ref.Machine.from_program(elf.program(elf_path)); m2 = ref.Machine.from_program(elf.program(elf_path))
            if bridge:
                BR.park(m); BR.park(m2)
                ids = chat.prompt_ids([{"role": "user", "content": prompt}])
            else:  # standalone program: unparked machines, a forced </tool_call> makes the circuit emit GO
                ids = chat.prompt_ids([{"role": "user", "content": "run"}], tools=())
            lock = BR.Session(chat.T, m2); lock.history(ids)
            q.put(("start", {"prompt_tokens": len(ids)}))
            tracer = vm_trace.Tracer(chat.T, elf_path, ids, chat.v0, park=bridge)
            def on_token(i):
                buf.append(tok(i))
                r = tracer.feed(i)
                evs.extend(r if isinstance(r, list) else [r])
                flush()
            r = (chat.generate(ids, m, max_new, check_head=True, lock=lock, on_token=on_token) if bridge else
                 chat.generate(ids, m, 300, max_chip=500000, check_head=True, lock=lock, on_token=on_token, first=chat.T.TCC))
            flush(True)
            q.put(("done", {**{k: r[k] for k in ("text_tokens", "chip_tokens", "seconds", "head_mismatch", "lock_error")},
                           "trace_mismatch": tracer.bad, "instructions": tracer.step}))
    except Exception as e:  # surface to the page
        q.put(("error", {"message": repr(e)}))
    q.put(None)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send_json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def stream(self, q):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while (ev := q.get()) is not None:
                self.wfile.write(f"event: {ev[0]}\ndata: {json.dumps(ev[1])}\n\n".encode()); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path in ("/", "/index.html", "/game"):
            b = (HERE / ("game.html" if self.path == "/game" else "index.html")).read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path == "/game/status":
            self.send_json(GAME.status() if GAME else {"error": "build/e1m1.elf not built"})
        elif self.path == "/firmwares":
            b = json.dumps([[k, v[1], v[3]] for k, v in FIRMWARES.items()]).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path.startswith("/program"):
            name = self.path.split("fw=")[-1] if "fw=" in self.path else "calc"
            if name not in FIRMWARES: return self.send_error(404)
            b = program_json(name).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        else:
            self.send_error(404)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        q = queue.Queue()
        if self.path == "/run":
            threading.Thread(target=run, args=(body["prompt"], int(body.get("max_new", 600)), q, body.get("fw", "calc")), daemon=True).start()
        elif self.path == "/game/turn" and GAME:
            threading.Thread(target=GAME.play, args=(str(body.get("keys", "")), q), daemon=True).start()
        elif self.path == "/game/new" and GAME:
            GAME.reset(); return self.send_json(GAME.status())
        else:
            return self.send_error(404)
        self.stream(q)

if True:
    t0 = time.time()
    _m = ref.Machine.from_program(elf.program(FIRMWARES["calc"][0])); BR.park(_m)
    chat.generate(chat.prompt_ids([{"role": "user", "content": "Use calc: 2+2"}]), _m, 40)
    print(f"compile warm-up {time.time() - t0:.0f}s", flush=True)
print(f"ready on http://{args.host}:{args.port}", flush=True)
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
