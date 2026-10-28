"""Run ELFs on the reference machine (no GPU): instructions, chip tokens, output, and whether two ELFs match.
   uv run python ../../../toolchain/refrun.py ELF [ELF ...]   (cwd: upstream/experimental/riscv)"""
import hashlib, sys
from chipformer.rv import elf, ref

for path in sys.argv[1:]:
    m = ref.Machine.from_program(elf.program(path))
    toks = m.run(10**8)
    h = hashlib.sha1(repr(toks).encode()).hexdigest()[:10]
    print(f"{path}: halted={m.halted} fault={m.fault} instr={m.steps} chip_tokens={len(toks)} "
          f"trace={h} out={bytes(m.out)[:100]!r}")
