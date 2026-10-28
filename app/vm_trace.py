"""Decode Qwenputer's chip-token stream into per-instruction events for the viewer.

A shadow ref.Machine is stepped one record at a time as chip tokens arrive; each record's tokens are
checked against what the shadow predicts (they should be identical: that is the repo's own lock check)
and turned into: PC, disassembly, register writes, memory accesses, bignum digit-file changes, putchar.
"""
import re, struct
from pathlib import Path

from chipformer.rv import bridge as BR, elf, ref
from chipformer.rv import spec as S

ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
       "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
R = lambda i: ABI[i]
M32 = 0xFFFFFFFF
RTYPE = {"ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND", "MUL", "MULH", "MULHSU", "MULHU",
         "DIV", "DIVU", "REM", "REMU", "SH1ADD", "SH2ADD", "SH3ADD"}
ITYPE = {"ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI", "SRAI"}
LOADS = {"LB", "LH", "LW", "LBU", "LHU"}
STORES = {"SB", "SH", "SW"}
BRANCHES = {"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"}
CUSTOM = {"BMAC", "BMSB", "BLD", "BGET", "BSCAN", "BPRD"}


def asm(word: int, pc: int) -> str:
    i = ref.decode(word); op = i.op; m = op.lower()
    if op in RTYPE: return f"{m} {R(i.rd)}, {R(i.rs1)}, {R(i.rs2)}"
    if op == "ADDI" and i.rs1 == 0: return f"li {R(i.rd)}, {i.imm}"
    if op == "ADDI" and i.imm == 0: return f"mv {R(i.rd)}, {R(i.rs1)}"
    if op in ITYPE: return f"{m} {R(i.rd)}, {R(i.rs1)}, {i.imm}"
    if op in LOADS: return f"{m} {R(i.rd)}, {i.imm}({R(i.rs1)})"
    if op in STORES: return f"{m} {R(i.rs2)}, {i.imm}({R(i.rs1)})"
    if op in BRANCHES: return f"{m} {R(i.rs1)}, {R(i.rs2)}, {(pc + i.imm) & M32:#x}"
    if op == "JAL": return f"j {(pc + i.imm) & M32:#x}" if i.rd == 0 else f"jal {R(i.rd)}, {(pc + i.imm) & M32:#x}"
    if op == "JALR":
        if i.rd == 0 and i.rs1 == 1 and i.imm == 0: return "ret"
        return f"jalr {R(i.rd)}, {i.imm}({R(i.rs1)})"
    if op in ("LUI", "AUIPC"): return f"{m} {R(i.rd)}, {(i.imm & M32) >> 12:#x}"
    if op in CUSTOM:
        if op == "BPRD": return f"bprd d{i.dd}, digit {i.imm}"
        return f"{m} {R(i.rd)}, {R(i.rs1)}, {R(i.rs2)}"
    return m


def load_program(path):
    """ELF -> disassembly rows, function symbols, and the C source of each function."""
    prog = elf.program(path)
    words = list(prog.text)
    while words and words[-1] == 0: words.pop()
    listing = [asm(w, S.ROM_BASE + 4 * i) for i, w in enumerate(words)]
    b = Path(path).read_bytes()
    shoff, = struct.unpack_from("<I", b, 0x20); shentsize, shnum = struct.unpack_from("<HH", b, 0x2E)
    secs = [struct.unpack_from("<10I", b, shoff + i * shentsize) for i in range(shnum)]
    syms = []
    for s in (s for s in secs if s[1] == 2):
        st = secs[s[6]]
        for k in range(s[5] // 16):
            nm, val, sz, info, _, _ = struct.unpack_from("<IIIBBH", b, s[4] + k * 16)
            if info & 15 == 2 and sz:
                syms.append([b[st[4] + nm:b.index(b"\0", st[4] + nm)].decode().split(".")[0], val, sz])
    syms.sort(key=lambda s: s[1])
    return {"base": S.ROM_BASE, "asm": listing, "syms": syms}


def c_sources(calc_c: Path) -> dict:
    """name -> source text of each top-level function in calc.c (main = calc_eval + gather + main)."""
    lines = calc_c.read_text().splitlines()
    out, i = {}, 0
    start = re.compile(r"^(?:__attribute__\(\(\w+\)\)\s+)?(?:static\s+)?[A-Za-z_][\w \*]*?\b(\w+)\([^;]*\)\s*(?:__attribute__\(\(.*\)\))?\s*\{")
    while i < len(lines):
        m = start.match(lines[i])
        if m and not lines[i].startswith(("#", "//")):
            j = i
            if not lines[i].rstrip().endswith("}") or lines[i].count("{") != lines[i].count("}"):
                depth = 0
                for j in range(i, len(lines)):
                    depth += lines[j].count("{") - lines[j].count("}")
                    if depth <= 0 and j >= i: break
            out[m.group(1)] = "\n".join(lines[i:j + 1]); i = j + 1
        else:
            i += 1
    out["main"] = "\n\n".join(out.get(k, "") for k in ("calc_eval", "gather", "main")).strip()
    return out


def bn_str(digits) -> str:
    s = "".join(map(str, reversed(digits))).lstrip("0")
    return s or "0"


class Tracer:
    def __init__(self, T, firmware, prompt_ids, v0, park=True):
        self.v0, self.T = v0, T
        self.m = ref.Machine.from_program(elf.program(firmware))
        if park: BR.park(self.m)
        self.ses = BR.Session(T, self.m)
        self.ses.history(prompt_ids)
        self.rec, self.step, self.bad = [], 0, 0

    def feed(self, mid: int) -> list:
        """One model id from the generation; returns the viewer events it completes."""
        if mid < self.v0:
            self.ses.qwen([mid], run=False)
            return []
        t = mid - self.v0
        fam, p = S.family(t)
        if fam == "GO":
            self.rec = []
            buf = b"".join(r.rstrip(b"\0") for r in self.m.text[1:min(self.ses.cap, S.CAP_MAX) + 1])
            return [{"k": "go", "pc": self.m.pc, "x": [[i, v] for i, v in enumerate(self.m.x) if v],
                     "call": buf.decode(errors="replace")}]
        self.rec.append(t)
        if fam not in S.FINAL:
            return []
        rec, self.rec = self.rec, []
        return [self._commit(rec)]

    def _commit(self, rec):
        m = self.m
        toks = [list(S.family(t)) for t in rec]
        if m.n == S.N_MAX:
            want = m.record(); ev = {"k": "R", "pc": m.pc, "toks": toks}
        else:
            pc = m.pc
            word = m.rom[(pc - S.ROM_BASE) >> 2]; ins = ref.decode(word)
            x0 = list(m.x)
            a, b = x0[ins.rs1], x0[ins.rs2]
            bn0 = [list(r) for r in m.bn] if ins.op in CUSTOM else None
            want = m.record()
            ev = {"k": "i", "pc": pc, "asm": asm(word, pc), "op": ins.op, "toks": toks}
            regs = [[r, m.x[r]] for r in range(1, 32) if m.x[r] != x0[r]]
            if regs: ev["x"] = regs
            if ins.op in LOADS | STORES and not m.halted:
                ea = (a + ins.imm) & M32
                if ins.op in STORES:
                    idx = m.ram_index(ea)
                    ev["st"] = [ea, m.ram[idx] if idx is not None else 0, {"SB": 1, "SH": 2, "SW": 4}[ins.op],
                                b & M32]
                else:
                    ev["ld"] = [ea, m.x[ins.rd]]
            if ins.op in ("BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"):
                ev["taken"] = bool(toks[-1][0] == "J")
            if bn0 is not None:
                if ins.op in ("BMAC", "BMSB", "BLD", "BGET"):
                    ev["bnargs"] = {"dd": b & 7, "ds": b >> 3 & 7, "ovw": b >> 6 & 1, "n": b >> 7 & 63,
                                    "off": b >> 13 & 127, "m": min(b >> 20 & 15, 9), "a": a}
                if ins.op == "BSCAN": ev["bnargs"] = {"a": a}
                ch = {r: bn_str(m.bn[r]) for r in range(8) if m.bn[r] != bn0[r]}
                if ch: ev["bn"] = ch
                ev["bnop"] = ins.op
            if want and S.family(want[-1])[0] == "BYTE":
                ev["out"] = S.family(want[-1])[1]
        ev["i"] = self.step; self.step += 1
        if want != rec:
            ev["bad"] = True; self.bad += 1
        if m.halted and m.fault is None:
            m.halted = False
            ev["halt"] = True
        return ev
