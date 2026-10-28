# RISC-V toolchain for the chipformer RV32 machine (what env/build_*.sh expect): gcc + picolibc + qemu-user.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc-riscv64-unknown-elf picolibc-riscv64-unknown-elf binutils-riscv64-unknown-elf qemu-user python3 make \
    && rm -rf /var/lib/apt/lists/*
