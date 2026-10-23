#!/usr/bin/env python3
# Why does a warm, identical llama prefill still rewrite and compile? Counts schedule-cache hits/misses and names every
# new entry in the compile cache, per call. The same prompt at start_pos 0 every call, so after the first call every lookup should hit.
# usage: PYTHONPATH=. python tools/cache_probe.py MODEL.gguf [T] [calls]
import contextlib, cProfile, io, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from llama3 import build_transformer
from tinygrad import Context, Tensor
import tinygrad.schedule as sched
import tinygrad.engine.realize as realize

class CountingDict(dict):
  hits = misses = 0
  def get(self, k, default=None):
    if k in self: CountingDict.hits += 1
    else: CountingDict.misses += 1
    return super().get(k, default)

def main():
  model_path = Path(sys.argv[1])
  T, calls = int(sys.argv[2]) if len(sys.argv) > 2 else 512, int(sys.argv[3]) if len(sys.argv) > 3 else 4
  sched.schedule_cache = CountingDict(sched.schedule_cache)
  model = build_transformer(model_path, model_size="1B", quantize="float16")
  toks = [128000] + [(i * 7919) % 120000 + 1000 for i in range(T - 1)]
  for c in range(calls):
    CountingDict.hits = CountingDict.misses = 0
    before = set(realize.to_program_cache)
    t = time.perf_counter()
    # calls 0-2 under DEBUG=2 (as .work/mojo-tg/prefill.py warms up), later calls without it; the last one profiled
    dbg = 2 if c < 3 else 0
    if c == calls - 1:
      pr = cProfile.Profile()
      pr.enable()
    with Context(DEBUG=dbg), contextlib.redirect_stdout(io.StringIO()):
      nxt = model(Tensor([toks]), 0, 0.0, 0, 0.8, 0.0, 0.0).item()
    if c == calls - 1:
      pr.disable()
      pr.dump_stats(".work/cache_probe.pstats")
    new = [k for k in realize.to_program_cache if k not in before]
    names = [str(getattr(getattr(realize.to_program_cache[k], "arg", None), "name", "?")) for k in new]
    tag = f"call {c} DEBUG={dbg}{' prof' if c == calls - 1 else ''}"
    print(f"{tag}: {time.perf_counter()-t:.3f} s  next {nxt}  schedule_cache hit {CountingDict.hits} miss {CountingDict.misses}"
          f"  compiled {len(new)} {names[:6]}")

if __name__ == "__main__": main()
