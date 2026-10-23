# run Mojo-compiled gfx1100 kernels in tinygrad: the helpers live in tinygrad/runtime/support/mojo_gemm.py, this is the CLI
import sys, pathlib
from tinygrad.runtime.support.mojo_gemm import extract_code_object, kernel_name, program  # noqa: F401  (re-exported for extra/)

if __name__ == "__main__":  # usage: python3 mojo_kernel.py KERNEL.s OUT.co
  co = extract_code_object(pathlib.Path(sys.argv[1]).read_text())
  pathlib.Path(sys.argv[2]).write_bytes(co)
  print(f"{sys.argv[2]}: {kernel_name(co)} ({len(co)} bytes)")
