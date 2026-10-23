# run Mojo-compiled gfx1100 kernels in tinygrad: `mojo build --emit asm` embeds each GPU kernel as a finished AMDGPU code
# object; pull it out and hand it to tinygrad as a custom kernel binary
from __future__ import annotations
import re, codecs, sys, pathlib

def extract_code_object(asm:str) -> bytes:
  return codecs.escape_decode(re.search(r'\.asciz\s+"(\\177ELF.*?)"\n', asm, re.S).group(1).encode())[0]

def kernel_name(co:bytes) -> str: return re.search(rb'([A-Za-z_][A-Za-z0-9_]*)\.kd\x00', co).group(1).decode()

def program(co:bytes, bufs:list, local:int, grid:tuple[int, ...], ops:int=0, mem:int=0) -> UOp:
  """A PROGRAM UOp running code object `co` over `bufs` (kernel args in order) with `local` threads and `grid` workgroups."""
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.renderer import Estimates
  specials = [UOp.special(local, "lidx0")] + [UOp.special(g, f"gidx{i}") for i, g in enumerate(grid)]
  sink = UOp.sink(*[b.base for b in bufs], *specials, arg=KernelInfo(kernel_name(co), estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg="mojo"), UOp(Ops.BINARY, arg=co)))

if __name__ == "__main__":  # usage: python3 mojo_kernel.py KERNEL.s OUT.co
  co = extract_code_object(pathlib.Path(sys.argv[1]).read_text())
  pathlib.Path(sys.argv[2]).write_bytes(co)
  print(f"{sys.argv[2]}: {kernel_name(co)} ({len(co)} bytes)")
