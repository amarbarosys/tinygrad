# fp16 GEMM, Mojo WMMA (MOJO_GEMM path) vs rocblas-bench gemm_ex, same shapes and layouts, both timed as mean over CNT hot runs
# PYTHONPATH=. DEV=AMD MOJO_GEMM=1 CNT=300 python3 tools/gemm_vs_rocblas.py
# layout nn: C = A @ B (B stored [K, N]); nt: C = A @ W.T (W stored [N, K], nn.Linear). Row-major C[M,N] is column-major C^T,
# so rocBLAS runs m=N n=M with op(A) on B/W: nn -> N N, nt -> T N.
import os, statistics, subprocess, sys, traceback
from tinygrad import Tensor, dtypes, GlobalCounters, Context
from tinygrad.helpers import getenv
from tinygrad.engine.realize import run_linear
from tinygrad.runtime.support import mojo_gemm

CNT = getenv("CNT", 300)
REPEATS = getenv("REPEATS", 3)
WARMUP_RUNS = getenv("WARMUP_RUNS", 20)
SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
          (2048, 4096, 4096), (2048, 4096, 14336), (2048, 14336, 4096), (2048, 4096, 1024)]  # (M, K, N), llama 8B linears at 2048 tokens

def ours_benchmark(M:int, K:int, N:int, layout:str) -> tuple[float, float]:
  a = Tensor.randn(M, K, dtype=dtypes.half).realize()
  w = Tensor.randn(N, K, dtype=dtypes.half).realize() if layout == "nt" else Tensor.randn(K, N, dtype=dtypes.half).realize()
  out = mojo_gemm.dot(a, w.T if layout == "nt" else w)
  if out is None: raise RuntimeError(f"mojo path not taken for {M}x{K}x{N} {layout}")
  linear = out.schedule_linear()

  # Warmup repeat to lock GPU clocks and instructions
  for _ in range(min(CNT, 100)):
    GlobalCounters.reset()
    with Context(DEBUG=2): run_linear(linear)

  # 3 repeats
  repeat_means = []
  for _ in range(REPEATS):
    ets = []
    for _ in range(CNT):
      GlobalCounters.reset()
      with Context(DEBUG=2): run_linear(linear)
      ets.append(GlobalCounters.time_sum_s)
    repeat_means.append(statistics.mean(ets))

  ref = (a.float() @ (w.T if layout == "nt" else w).float()).realize()
  err = (out.float() - ref).square().mean().item()
  if err > 1e-2: raise RuntimeError(f"wrong result {M}x{K}x{N} {layout}: mse {err}")

  mean_t = statistics.mean(repeat_means)
  spread_pct = ((max(repeat_means) - min(repeat_means)) / mean_t) * 100.0 if mean_t > 0 else 0.0
  return mean_t, spread_pct

def rocblas_one(M:int, K:int, N:int, layout:str, use_hipblaslt:bool=False) -> float:
  ta = "T" if layout == "nt" else "N"
  cmd = ["/opt/rocm/bin/rocblas-bench", "-f", "gemm_ex", "-m", str(N), "-n", str(M), "-k", str(K), "--transposeA", ta, "--transposeB", "N",
         *[x for t in ("a", "b", "c", "d") for x in (f"--{t}_type", "f16_r")], "--compute_type", "f32_r", "-i", str(CNT), "-j", "50"]
  env = dict(os.environ)
  if use_hipblaslt: env["ROCBLAS_USE_HIPBLASLT"] = "1"
  else: env.pop("ROCBLAS_USE_HIPBLASLT", None)
  res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env).stdout.strip().splitlines()[-1]
  return float(res.split(",")[-1]) * 1e-6

def rocblas_benchmark(M:int, K:int, N:int, layout:str) -> tuple[float, float]:
  # 3 repeats for default and hipBLASLt (-j 50 warms up clocks per invocation)
  repeats_default = [rocblas_one(M, K, N, layout, use_hipblaslt=False) for _ in range(REPEATS)]
  repeats_lt = [rocblas_one(M, K, N, layout, use_hipblaslt=True) for _ in range(REPEATS)]

  # Select best (lowest execution time)
  mean_default = statistics.mean(repeats_default)
  mean_lt = statistics.mean(repeats_lt)
  best_repeats = repeats_lt if mean_lt < mean_default else repeats_default

  mean_t = statistics.mean(best_repeats)
  spread_pct = ((max(best_repeats) - min(best_repeats)) / mean_t) * 100.0 if mean_t > 0 else 0.0
  return mean_t, spread_pct

if __name__ == "__main__":
  print("layout,M,K,N,mojo_mean_tflops,mojo_spread_pct,rocblas_mean_tflops,rocblas_spread_pct,ratio_mean,status")
  only = getenv("ONLY", "")
  for layout in ("nn", "nt"):
    for M, K, N in SHAPES:
      if only and f"{layout},{M},{K},{N}" not in only.split(";"): continue
      fl = 2 * M * K * N * 1e-12
      try:
        mm, m_spread = ours_benchmark(M, K, N, layout)
        mojo_tflops = fl / mm
      except Exception:
        print(f"FAIL mojo {M}x{K}x{N} {layout}:\n{traceback.format_exc()}", file=sys.stderr)
        mm, m_spread, mojo_tflops = float("nan"), float("nan"), float("nan")

      rb, r_spread = rocblas_benchmark(M, K, N, layout)
      rocblas_tflops = fl / rb
      ratio = rocblas_tflops / mojo_tflops if mojo_tflops > 0 else float("nan") # Wait, in original ratio was rb/mm which is mojo/rocblas
      ratio_mojo_vs_rocblas = mojo_tflops / rocblas_tflops if rocblas_tflops > 0 else float("nan")

      status = "OK"
      if m_spread > 1.0 or r_spread > 1.0:
        status = "VOID_HIGH_SPREAD"

      out_line = f"{layout},{M},{K},{N},{mojo_tflops:.1f},{m_spread:.2f}%,{rocblas_tflops:.1f},{r_spread:.2f}%,{ratio_mojo_vs_rocblas:.2f},{status}"
      print(out_line, flush=True)
