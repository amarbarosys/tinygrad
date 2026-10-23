#!/usr/bin/env python3
# Falsifier for "rewrite engine in Mojo, graph stays in Python": time UOp.toposort (Python) against mojo/ext/uop_topo.mojo
# on every root a cold llama run asks to sort. Exits non-zero if any order differs.
# usage: CACHELEVEL=0 python tools/topo_falsifier.py MODEL.gguf [steps]   (build the .so first, see mojo/ext/uop_topo.mojo)
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / ".work" / "mojo-ext")]
import uop_topo
from llama3 import build_transformer
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops

def main():
  model_path, steps = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3
  orig, roots, gated = UOp.toposort, [], 0
  def rec(self, gate=None, enter_calls=False):
    nonlocal gated
    if gate is None: roots.append((self, enter_calls))
    else: gated += 1
    return orig(self, gate, enter_calls)
  UOp.toposort = rec
  model, tok = build_transformer(model_path, model_size="1B"), 128000
  for pos in range(steps): tok = model(Tensor([[tok]]), pos, 0.0, 0, 0.8, 0.0, 0.0).item()
  UOp.toposort = orig
  nodes = sum(len(orig(r, None, e)) for r, e in roots)
  print(f"roots {len(roots)} (gated calls not covered: {gated}), nodes visited {nodes}")
  s_src, s_op, call = sys.intern("src"), sys.intern("op"), Ops.CALL
  def capi(r, e):
    d: dict = {}
    uop_topo.toposort_capi(id(r), id(call), e, id(d), id(s_src), id(s_op))
    return d
  arms = (("python", lambda r, e: orig(r, None, e)), ("mojo", lambda r, e: uop_topo.toposort(r, call, e)), ("capi", capi))
  for name, fn in arms[1:]:
    for (r, e) in roots:
      if list(orig(r, None, e)) != list(fn(r, e)): sys.exit(f"FAIL order ({name}): root {r.op} enter_calls={e}")
  print("order: identical on all roots, both mojo arms")
  for name, fn in arms * 3:
    t = time.perf_counter()
    for r, e in roots: fn(r, e)
    dt = time.perf_counter() - t
    print(f"{name:6s} {dt*1e3:8.1f} ms  {dt/nodes*1e9:6.1f} ns/node")

if __name__ == "__main__": main()
