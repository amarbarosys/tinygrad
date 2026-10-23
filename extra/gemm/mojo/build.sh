#!/usr/bin/env bash
# rebuild the prebuilt code object: bash extra/gemm/mojo/build.sh [N]   (needs Mojo 1.1 with MAX, e.g. MOJO=path/to/mojo)
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-4096}; MOJO=${MOJO:-mojo}; tmp=$(mktemp -d .build.XXXX); trap 'rm -rf "$tmp"' EXIT
"$MOJO" build --target-accelerator gfx1100 --emit asm -D WMMA_C_F16=0 -D TG_N="$N" -I . -o "$tmp/k.s" tg_wmma_gemm.mojo
PYTHONPATH=../../.. python3 mojo_kernel.py "$tmp/k.s" "wmma_pipe_gfx1100_n$N.co"
