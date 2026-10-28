"""E1M1 on the Qwenputer: a person plays the raycaster firmware (build/e1m1.elf) turn by turn.

Each turn is one short conversation: the machine state after the previous turn is injected (as RvChat does for every
firmware), the player's keys are forced in as a `move` tool call, and the circuit (the real weights) runs the firmware
to its park, printing the frame. The run is lock-checked against the reference machine token for token; only a clean
run advances the game (the next turn starts from the reference machine's state, which the stream just matched).
Qwen's own text is never sampled here: every text token is either forced (the call) or the circuit's (the frame).
"""
import copy, re, threading, time
from pathlib import Path

from chipformer.rv import bridge as BR, elf, ref

ELF = Path(__file__).resolve().parent.parent / "build/e1m1.elf"
KEYS = re.compile(r"[wasdqe]{0,40}")
MOVE_TOOL = {"type": "function", "function": {
    "name": "move",
    "description": "Move the player in Doom E1M1 and return the new view as ASCII. Keys: w/s forward/back one cell, "
                   "a/d turn 11.25 degrees left/right, q/e strafe left/right; several per call, e.g. wwwdd.",
    "parameters": {"type": "object", "properties": {"keys": {"type": "string", "description": "e.g. wwwdd"}},
                   "required": ["keys"]}}}


class Game:
    def __init__(self, chat, gpu_lock):
        self.chat, self.gpu_lock = chat, gpu_lock
        self.prog = elf.program(ELF)
        # the template's default generation prompt (thinking on); Qwen is not sampled in this mode anyway
        out = chat.tok.apply_chat_template([{"role": "user", "content": "Let's play Doom E1M1."}], tools=[MOVE_TOOL],
                                           add_generation_prompt=True, tokenize=True)
        self.ids = [int(x) for x in (out if isinstance(out, list) else out["input_ids"])]
        self.state_lock = threading.Lock()
        self.reset()

    def reset(self):
        m = ref.Machine.from_program(self.prog)
        BR.park(m)
        with self.state_lock:
            self.m, self.turn, self.won, self.frame = m, 0, False, ""
            self.totals = {"instr": 0, "chip": 0, "seconds": 0.0}

    def status(self):
        return {"turn": self.turn, "won": self.won, "frame": self.frame, **self.totals}

    def play(self, keys, q):
        """One turn; events to q: start, progress, done (or error), then None."""
        chat, T = self.chat, self.chat.T
        try:
            if not KEYS.fullmatch(keys):
                raise ValueError("keys: up to 40 of w a s d q e")
            call = f"<tool_call>\n<function=move>\n<parameter=keys>\n{keys or '.'}\n</parameter>\n</function>\n</tool_call>"
            force = T.encode(call)
            with self.gpu_lock:
                m, m2 = copy.deepcopy(self.m), copy.deepcopy(self.m)
                lock = BR.Session(T, m2)
                lock.history(self.ids)
                steps0 = m2.steps
                q.put(("start", {"turn": self.turn + 1, "keys": keys, "prompt_tokens": len(self.ids)}))
                st = {"chip": 0, "text": [], "after": False, "last": time.time(), "t0": time.time()}

                def on_token(i):
                    if i >= chat.v0:
                        st["chip"] += 1
                    elif st["after"]:
                        st["text"].append(i)
                    elif i == T.TCC:
                        st["after"] = True
                    if time.time() - st["last"] > 0.25:
                        st["last"] = time.time()
                        q.put(("progress", {"chip": st["chip"], "seconds": round(time.time() - st["t0"], 1),
                                            "text": chat.tok.decode(st["text"], skip_special_tokens=False)}))
                r = chat.generate(self.ids, m, max_new=len(force) + 2000, max_chip=300000, check_head=True,
                                  lock=lock, on_token=on_token, force=force, stop_after_halt=True)
                clean = r["lock_error"] is None and len(lock.runs) == 1 and m2.fault is None and not m2.halted
                frame = chat.tok.decode(st["text"], skip_special_tokens=False)
                if clean:
                    with self.state_lock:
                        self.m = m2
                        self.turn += 1
                        self.frame = frame
                        self.won = self.won or "EXIT REACHED" in frame
                        self.totals["instr"] += m2.steps - steps0
                        self.totals["chip"] += r["chip_tokens"]
                        self.totals["seconds"] = round(self.totals["seconds"] + r["seconds"], 1)
            q.put(("done", {"frame": frame, "clean": clean, "instructions": m2.steps - steps0,
                            **{k: r[k] for k in ("text_tokens", "chip_tokens", "seconds", "chip_seconds", "text_seconds",
                                                 "head_mismatch", "lock_error")}, "game": self.status()}))
        except Exception as e:  # surface to the page
            q.put(("error", {"message": repr(e)}))
        q.put(None)
