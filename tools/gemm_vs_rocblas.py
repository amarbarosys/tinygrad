# fp16 GEMM, Mojo WMMA (MOJO_GEMM path) vs rocblas-bench gemm_ex, same shapes and layouts, both timed as mean over CNT hot runs
# PYTHONPATH=. DEV=AMD MOJO_GEMM=1 CNT=300 python3 tools/gemm_vs_rocblas.py
# layout nn: C = A @ B (B stored [K, N]); nt: C = A @ W.T (W stored [N, K], nn.Linear). Row-major C[M,N] is column-major C^T,
# so rocBLAS runs m=N n=M with op(A) on B/W: nn -> N N, nt -> T N.
import statistics, subprocess, sys, traceback
from tinygrad import Tensor, dtypes, GlobalCounters, Context
from tinygrad.helpers import getenv
from tinygrad.engine.realize import run_linear
from tinygrad.runtime.support import mojo_gemm

CNT = getenv("CNT", 300)
SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
          (2048, 4096, 4096), (2048, 4096, 14336), (2048, 14336, 4096), (2048, 4096, 1024)]  # (M, K, N), llama 8B linears at 2048 tokens

def ours(M:int, K:int, N:int, layout:str) -> tuple[float, float]:
  a = Tensor.randn(M, K, dtype=dtypes.half).realize()
  w = Tensor.randn(N, K, dtype=dtypes.half).realize() if layout == "nt" else Tensor.randn(K, N, dtype=dtypes.half).realize()
  out = mojo_gemm.dot(a, w.T if layout == "nt" else w)
  if out is None: raise RuntimeError(f"mojo path not taken for {M}x{K}x{N} {layout}")
  linear = out.schedule_linear()
  ets = []
  for _ in range(CNT):
    GlobalCounters.reset()
    with Context(DEBUG=2): run_linear(linear)
    ets.append(GlobalCounters.time_sum_s)
  ref = a.float() @ (w.T if layout == "nt" else w).float()
  if (err:=(out.float() - ref).square().mean().item()) > 1e-2: raise RuntimeError(f"wrong result {M}x{K}x{N} {layout}: mse {err}")
  return statistics.mean(ets), statistics.median(ets)

def rocblas(M:int, K:int, N:int, layout:str) -> float:
  ta = "T" if layout == "nt" else "N"
  cmd = ["/opt/rocm/bin/rocblas-bench", "-f", "gemm_ex", "-m", str(N), "-n", str(M), "-k", str(K), "--transposeA", ta, "--transposeB", "N",
         *[x for t in ("a", "b", "c", "d") for x in (f"--{t}_type", "f16_r")], "--compute_type", "f32_r", "-i", str(CNT), "-j", "50"]
  res = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
  return float(res.split(",")[-1]) * 1e-6

if __name__ == "__main__":
  print("layout,M,K,N,mojo_mean_tflops,mojo_median_tflops,rocblas_mean_tflops,ratio_mean")
  only = getenv("ONLY", "")
  for layout in ("nn", "nt"):
    for M, K, N in SHAPES:
      if only and f"{layout},{M},{K},{N}" not in only.split(";"): continue
      fl = 2 * M * K * N * 1e-12
      try: mm, md = ours(M, K, N, layout)
      except Exception:
        print(f"FAIL mojo {M}x{K}x{N} {layout}:\n{traceback.format_exc()}", file=sys.stderr)
        mm = md = float("nan")
      rb = rocblas(M, K, N, layout)
      print(f"{layout},{M},{K},{N},{fl/mm:.1f},{fl/md:.1f},{fl/rb:.1f},{rb/mm:.2f}", flush=True)
