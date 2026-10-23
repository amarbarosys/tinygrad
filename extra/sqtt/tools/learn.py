# learning loop over rule strengths: nudge one strength by +-1/+-2, keep the change only if it gains exact SRAM deltas on
# both halves of the captures (alternating order of all_captures); repeat until a full round keeps nothing.
# usage (worktree root): PYTHONPATH=. python3 extra/sqtt/tools/learn.py [NAME=VAL ...]   -> log on stdout, final strengths last
import sys, json, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import sweep
from extra.sqtt.heldout.score import all_captures

MODULE = ["COLD_READ", "MEM_AFTER_EXEC", "VMEM_AFTER_SALU", "LDS_TO_VALU_LOAD", "LDS_TO_VALU_STORE", "VALUB_OCC", "SALU_AFTER_VALUB",
          "STORE_LOAD_DELAY", "STORE_LOAD_WINDOW", "BARRIER_WINDOW", "BARRIER_BUSY", "BARRIER_RELEASE", "BARRIER_SERIAL", "ICACHE_WAVE_GAP", "SCC_HOLD"]
INNER = {"VALU_DEP_LAT": 5, "IMM_AFTER_ALU": 3, "IMM_AFTER_MEM": 3, "LDS_ACCEPT": 3, "VALU_AFTER_SALU": 5, "SALU_EXEC_LAT": 4,
         "SALU_DEP_LAT": 2, "VALU_TO_SCALAR": 9, "LDS_RET": 28, "SMEM_AFTER_SALU": 9, "WAITCNT_FREE_PENALTY": 4, "END_LAT": 16}

def evaluate(params:dict) -> dict[str, int]:
  sweep.OV = [f"@{k}={v}" if k in INNER else f"{k}={v}" for k, v in params.items()]
  with mp.Pool(15, initializer=sweep.init) as pool: res = pool.map(sweep.one, CAPS)
  return {stem: h for stem, h, _, _ in res}

def halves(hits:dict) -> tuple[int, int]:
  return sum(hits[p.stem] for p in CAPS[0::2]), sum(hits[p.stem] for p in CAPS[1::2])

if __name__ == "__main__":
  import test.mockgpu.amd.sqtt_enc as enc
  CAPS = all_captures()
  params = {k: getattr(enc, k) for k in MODULE} | dict(INNER)
  for a in sys.argv[1:]: k, v = a.split("="); params[k] = int(v)
  best = evaluate(params); bh = halves(best)
  print(f"start {sum(best.values())} halves {bh}", flush=True)
  changed = True
  while changed:
    changed = False
    for k in list(params):
      for step in (1, -1, 2, -2):
        trial = dict(params); trial[k] = params[k] + step
        if trial[k] < (0 if k.startswith("K_") is False else -3): continue
        hits = evaluate(trial); th = halves(hits)
        gain = [th[0] - bh[0], th[1] - bh[1]]
        moved = sum(1 for s in hits if hits[s] != best[s])
        if gain[0] > 0 and gain[1] > 0:
          params, best, bh, changed = trial, hits, th, True
          print(f"KEEP {k}={trial[k]} ({step:+d}): total {sum(hits.values())} halves {th} gain {gain}, {moved} captures moved", flush=True)
          break
  print("FINAL", json.dumps({k: v for k, v in params.items()}), sum(best.values()), flush=True)
