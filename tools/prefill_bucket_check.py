#!/usr/bin/env python3
# Gate for the padded prefill jit (extra/models/llama.py PREFILL_JIT): drive examples/llama3.py's prefill() and 12 greedy decode
# steps three ways, A token-by-token (the old prefill), B whole prompt eager (PREFILL_JIT=0), C whole prompt through the bucket jit.
# C must equal B token for token; A vs C is reported. Lengths cover jit reuse inside a bucket and the bucket edges.
# Then times a warm prefill (eager vs jit) at T=500 and 512, paced 100 ms between calls (the card is undervolted).
# usage: PYTHONPATH=. python tools/prefill_bucket_check.py MODEL.gguf [reps] [lengths]   (MOJO_GEMM=0|1 picks the GEMM; reps 0 skips timing;
#        lengths e.g. 37,100)
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import llama3
from tinygrad import Context, Device, Tensor
from tinygrad.runtime.support.mojo_gemm import MOJO_GEMM

def generate(model, prompt:list[int], whole:bool) -> list[int]:
  llama3.last_seen_toks = []
  if whole: start_pos = llama3.prefill(model, prompt[:-1])
  else:
    start_pos = 0
    for tok in prompt[:-1]:
      model(Tensor([[tok]], device=llama3.device), start_pos, 0.0).realize()
      start_pos += 1
  out, tok = [], prompt[-1]
  for _ in range(12):
    tok = model(Tensor([[tok]], device=llama3.device), start_pos, 0.0).item()
    out.append(tok)
    start_pos += 1
  return out

def prompt_of(T:int) -> list[int]: return [128000] + [(i * 7919 + T * 104729) % 120000 + 1000 for i in range(T - 1)]

def main():
  llama3.device, llama3.TEMPERATURE = Device.DEFAULT, 0.0
  reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
  lengths = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [37, 100, 128, 129, 300, 511]
  model = llama3.build_transformer(Path(sys.argv[1]), model_size="1B", quantize="float16")
  bad = 0
  for T in lengths:
    prompt = prompt_of(T)
    print(f"T={T} start", flush=True)
    with Context(PREFILL_JIT=0): b = generate(model, prompt, whole=True)
    c, a = generate(model, prompt, whole=True), generate(model, prompt, whole=False)
    print(f"T={T:4d} bucket jits {sorted(model.prefill_jits)}  C==B {c == b}  A==C {a == c}  C {c}")
    if c != b:
      print(f"  FAIL B {b}")
      bad += 1
  if bad: sys.exit(f"FAIL {bad} prompt(s): bucket jit differs from eager whole-prompt prefill")
  print("OK bucket jit == eager whole-prompt prefill on all prompts")
  for T in ((500, 512) if reps else ()):
    toks = prompt_of(T)
    for name, jit in (("eager", 0), ("jit", 1)):
      walls = []
      with Context(PREFILL_JIT=jit):
        for _ in range(reps + 2):
          time.sleep(0.1)
          t = time.perf_counter()
          model(Tensor([toks]), 0, 0.0).item()  # item() waits for the GPU; realize() returns once a jit replay is queued
          walls.append(time.perf_counter() - t)
      warm = sorted(walls[2:])
      print(f"MOJO_GEMM={MOJO_GEMM.value} T={T} {name:5s} warm median {warm[len(warm)//2]*1e3:7.1f} ms  ({' '.join(f'{w*1e3:.0f}' for w in walls)})")

if __name__ == "__main__": main()
