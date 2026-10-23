#!/usr/bin/env python3
# Where does a cold llama 1B run spend host time? Token-id driver (no tokenizer), cProfile buckets by tinygrad layer.
# usage: CACHELEVEL=0 python tools/prof_llama.py MODEL.gguf [steps]
import cProfile, pstats, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from llama3 import build_transformer
from tinygrad import Tensor

BUCKETS = [("graph_rewrite", ("tinygrad/uop/",)), ("schedule", ("tinygrad/engine/schedule", "tinygrad/schedule/")),
           ("codegen", ("tinygrad/codegen/", "tinygrad/renderer/")), ("compile", ("compiler_", "tinygrad/runtime/support/compiler")),
           ("runtime", ("tinygrad/runtime/", "tinygrad/device")), ("jit", ("tinygrad/engine/jit",)),
           ("tensor", ("tinygrad/tensor", "tinygrad/mixin", "tinygrad/gradient")), ("model", ("extra/models", "examples/"))]

def bucket(fn:str) -> str:
  for name, keys in BUCKETS:
    if any(k in fn for k in keys): return name
  return "other"

def main():
  model_path, steps = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30
  t0 = time.perf_counter()
  model = build_transformer(model_path, model_size="1B")
  print(f"load s {time.perf_counter()-t0:.2f}")
  pr, times, tok = cProfile.Profile(), [], 128000
  pr.enable()
  for pos in range(steps):
    t = time.perf_counter()
    tok = model(Tensor([[tok]]), pos, 0.0, 0, 0.8, 0.0, 0.0).item()
    times.append(time.perf_counter() - t)
  pr.disable()
  print("step s", " ".join(f"{x:.3f}" for x in times))
  print(f"total {sum(times):.2f}s  cold(first 3) {sum(times[:3]):.2f}s  warm median {sorted(times[3:])[len(times[3:])//2]*1e3:.2f}ms")
  pr.dump_stats(".work/prof_llama.pstats")
  # builtins ("~" file) bill their time to the callers' buckets, split by per-caller time
  agg: dict[str, float] = {}
  for (fn, _, _), (_, _, tt, _, callers) in pstats.Stats(pr).stats.items():
    if fn != "~" or not callers:
      agg[bucket(fn)] = agg.get(bucket(fn), 0.0) + tt
      continue
    ctot = sum(c[2] for c in callers.values()) or 1.0
    for (cfn, _, _), c in callers.items(): agg[bucket(cfn)] = agg.get(bucket(cfn), 0.0) + tt * c[2] / ctot
  tot = sum(agg.values())
  for k, v in sorted(agg.items(), key=lambda kv: -kv[1]): print(f"{k:14s} {v:7.2f}s {100*v/tot:5.1f}%")
  pstats.Stats(pr).sort_stats("tottime").print_stats(15)

if __name__ == "__main__": main()
