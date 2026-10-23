#!/usr/bin/env python3
# Can a warm llama prefill skip graph_rewrite altogether? Same fixed-shape prompt at start_pos 0, eager vs TinyJit around
# Transformer.forward (llama only jits 1-token decode). Exits non-zero if the two arms disagree on the next token.
# usage: PYTHONPATH=. python tools/prefill_jit_probe.py MODEL.gguf [T] [reps]
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from llama3 import build_transformer
from tinygrad import Tensor, TinyJit

def main():
  model_path = Path(sys.argv[1])
  T, reps = int(sys.argv[2]) if len(sys.argv) > 2 else 512, int(sys.argv[3]) if len(sys.argv) > 3 else 6
  model = build_transformer(model_path, model_size="1B", quantize="float16")
  toks = [128000] + [(i * 7919) % 120000 + 1000 for i in range(T - 1)]
  jit = TinyJit(model.forward)
  arms = {"eager": lambda: model.forward(Tensor([toks]), 0, 0.0, 0, 0.8, 0.0, 0.0).item(),
          "jit": lambda: jit(Tensor([toks]), 0, 0.0, 0, 0.8, 0.0, 0.0).item()}
  out = {}
  for name, fn in arms.items():
    walls, nxt = [], set()
    for _ in range(reps):
      t = time.perf_counter()
      nxt.add(fn())
      walls.append(time.perf_counter() - t)
    out[name] = nxt
    warm = sorted(walls[2:])
    print(f"{name:5s} T={T} next {sorted(nxt)} calls(s) {' '.join(f'{w:.3f}' for w in walls)}  warm median {warm[len(warm)//2]*1e3:.1f} ms")
  if out["eager"] != out["jit"] or len(out["eager"]) != 1: sys.exit(f"FAIL next token: eager {out['eager']} jit {out['jit']}")
  # a stale replay would also pass the check above: new prompts of the same shape must move the jit's answer with eager's
  for seed in (3, 11, 17):
    p = [128000] + [(i * 104729 * seed) % 120000 + 1000 for i in range(T - 1)]
    e, j = model.forward(Tensor([p]), 0, 0.0, 0, 0.8, 0.0, 0.0).item(), jit(Tensor([p]), 0, 0.0, 0, 0.8, 0.0, 0.0).item()
    print(f"prompt seed {seed}: eager {e} jit {j}")
    if e != j: sys.exit(f"FAIL prompt seed {seed}: eager {e} jit {j}")

if __name__ == "__main__": main()
