# fp16 GEMM through a Mojo WMMA kernel (extra/gemm/mojo, mojo-baro matmul_wmma_pipe) on gfx1100: Tensor.dot routes eligible
# matmuls here when MOJO_GEMM=1 (opt-in: eager it costs more host graph_rewrite than it saves; it pays under TinyJit). One code
# object per (M, N, K, B layout, kernel source hash), built with `mojo` on first use and cached.
from __future__ import annotations
import codecs, functools, hashlib, math, os, pathlib, re, shutil, subprocess, tempfile
from typing import TYPE_CHECKING, cast
from tinygrad.helpers import MOJO_GEMM, cache_dir, prod
from tinygrad.dtype import dtypes
if TYPE_CHECKING: from tinygrad.tensor import Tensor

SRC = pathlib.Path(__file__).parents[3] / "extra" / "gemm" / "mojo"

def extract_code_object(asm:str) -> bytes:
  """`mojo build --emit asm` embeds each GPU kernel as a finished AMDGPU code object in an .asciz string."""
  if (m:=re.search(r'\.asciz\s+"(\\177ELF.*?)"\n', asm, re.S)) is None: raise RuntimeError("no AMDGPU code object in the Mojo asm")
  return codecs.escape_decode(m.group(1).encode())[0]

def kernel_name(co:bytes) -> str:
  if (m:=re.search(rb'([A-Za-z_][A-Za-z0-9_]*)\.kd\x00', co)) is None: raise RuntimeError("no kernel descriptor in the code object")
  return m.group(1).decode()

def program(co:bytes, bufs:list, local:int, grid:tuple[int, ...], ops:int=0, mem:int=0):
  """A PROGRAM UOp running code object `co` over `bufs` (kernel args in order) with `local` threads and `grid` workgroups."""
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.renderer import Estimates
  specials = [UOp.special(local, "lidx0")] + [UOp.special(g, f"gidx{i}") for i, g in enumerate(grid)]
  sink = UOp.sink(*[b.base for b in bufs], *specials, arg=KernelInfo(kernel_name(co), estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg="mojo"), UOp(Ops.BINARY, arg=co)))

def tiles(M:int, N:int) -> tuple[int, int, int]:
  """(block M, block N, threads): the size rule of tg_wmma_gemm.mojo, which must stay identical."""
  b128 = math.ceil(N / 128) * math.ceil(M / 128)
  wm, wn, tm, tn = (4, 2, 2, 4) if b128 >= 96 else (2, 4 if b128 >= 64 else 2, 2, 2)
  return wm * tm * 16, wn * tn * 16, wm * wn * 32

@functools.cache
def code_object(M:int, N:int, K:int, tb:int) -> bytes|None:
  src = hashlib.sha256(b"".join(f.read_bytes() for f in sorted(SRC.glob("*.mojo")))).hexdigest()[:12]
  path = pathlib.Path(cache_dir) / "mojo_gemm" / f"wmma_f16_m{M}_n{N}_k{K}_tb{tb}_{src}.co"
  if path.exists(): return path.read_bytes()
  if (mojo:=os.environ.get("MOJO") or shutil.which("mojo")) is None or not (SRC / "tg_wmma_gemm.mojo").exists(): return None
  with tempfile.TemporaryDirectory() as tmp:
    cmd = [mojo, "build", "--target-accelerator", "gfx1100", "--emit", "asm", "-I", str(SRC), "-D", "WMMA_C_F16=1", "-D", f"WMMA_TB={tb}",
           "-D", f"TG_M={M}", "-D", f"TG_N={N}", "-D", f"TG_K={K}", "-o", f"{tmp}/k.s", str(SRC / "tg_wmma_gemm.mojo")]
    if subprocess.run(cmd, capture_output=True, timeout=900).returncode != 0: return None
    co = extract_code_object(pathlib.Path(f"{tmp}/k.s").read_text())
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
  tmp_path.write_bytes(co)
  tmp_path.replace(path)
  return co

@functools.cache
def _grad(tb:int):
  """One gradient function per B layout: it is part of the CALL, so a fresh closure per call would defeat the schedule cache."""
  def grad(g, kernel):  # C = A @ B', B' = b.T (tb) or b: dA = G @ B'.T, db = A.T @ G (transposed back for tb)
    from tinygrad.tensor import Tensor
    ta, tbuf, tg = Tensor(kernel.src[1]), Tensor(kernel.src[2]), Tensor(g)
    return ((tg @ (tbuf if tb else tbuf.T)).uop, ((tg.T @ ta) if tb else (ta.T @ tg)).uop, None)
  return grad

def dot(x:Tensor, w:Tensor) -> Tensor|None:
  """x @ w with the Mojo kernel when eligible (fp16, gfx1100, dims multiples of 128); None means use tinygrad's own."""
  from tinygrad.device import Device
  from tinygrad.uop.ops import Ops
  from tinygrad.tensor import Tensor
  if not MOJO_GEMM.value or not isinstance(x.device, str) or not x.device.startswith("AMD") or x.device != w.device: return None
  if x.dtype != dtypes.half or w.dtype != dtypes.half or w.ndim != 2 or x.ndim < 2: return None
  if not all(isinstance(s, int) for s in x.shape + w.shape) or getattr(dev:=Device[x.device], "arch", None) != "gfx1100": return None
  if type(getattr(dev, "iface", None)).__name__.startswith("MOCK"): return None  # the emulator cannot run Mojo code (no MUBUF)
  xs, ws = cast(tuple[int, ...], x.shape), cast(tuple[int, ...], w.shape)
  if xs[-1] != ws[0]: return None
  M, K, N = prod(xs[:-1]), xs[-1], ws[1]
  if M % 128 or N % 128 or K % 128: return None
  tb = int(w.uop.op is Ops.PERMUTE and w.uop.marg == (1, 0))  # x @ W.T (nn.Linear): read W as stored [N, K], no copy
  if (co:=code_object(M, N, K, tb)) is None: return None
  bm, bn, threads = tiles(M, N)
  a, b = x.reshape(M, K).contiguous(), (Tensor(w.uop.src[0]).contiguous() if tb else w.contiguous())
  c = Tensor.empty(M, N, dtype=dtypes.half, device=x.device)
  out = Tensor.custom_kernel(a, b, c, fxn=lambda a, b, c: program(co, [a, b, c], threads, (N // bn, M // bm), ops=2*M*N*K, mem=2*(M*K+K*N+M*N)),
                             grad_fxn=_grad(tb))[2]
  return out.reshape(*x.shape[:-1], N)
