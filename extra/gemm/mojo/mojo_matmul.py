# fp16 WMMA GEMM written in Mojo (mojo-baro matmul_wmma_pipe), run as a tinygrad custom kernel on gfx1100
# PYTHONPATH=. DEV=AMD CNT=300 python3 extra/gemm/mojo/mojo_matmul.py     (prebuilt for N=4096; other N: build.sh N)
import pathlib, statistics
from tinygrad import Tensor, dtypes, GlobalCounters, Context
from tinygrad.helpers import getenv
from tinygrad.engine.realize import run_linear
from extra.gemm.mojo.mojo_kernel import program

N, CNT = getenv("N", 4096), getenv("CNT", 5)
co = (pathlib.Path(__file__).parent / f"wmma_pipe_gfx1100_n{N}.co").read_bytes()

if __name__ == "__main__":
  a, b = Tensor.randn(N, N, dtype=dtypes.half).realize(), Tensor.randn(N, N, dtype=dtypes.half).realize()
  c = Tensor.empty(N, N, dtype=dtypes.float).realize()
  def fxn(a, b, c): return program(co, [a, b, c], 256, (N // 128, N // 128), ops=2 * N**3, mem=2 * N * N * 2 + N * N * 4)
  linear = Tensor.custom_kernel(a, b, c, fxn=fxn)[2].schedule_linear()
  ets = []
  for _ in range(CNT):
    GlobalCounters.reset()
    with Context(DEBUG=2): run_linear(linear)
    ets.append(GlobalCounters.time_sum_s)
  print(f"REAL TFLOPS {2 * N**3 / min(ets) * 1e-12:.2f} (median {2 * N**3 / statistics.median(ets) * 1e-12:.2f})")
  with Context(DEBUG=0): err = (c - (a.float() @ b.float())).square().mean().item()
  print(f"mean squared error {err}")
  if err > 1e-2: raise RuntimeError("matmul is wrong!")
