# per barrier, real vs emulator: release gap (first post-barrier token of the traced waves - last traced barrier token)
# and the untraced SIMD's offset (p-wave n-th LDS - w-wave n-th LDS, last LDS before the barrier); absolute cycles
import sys, io, contextlib, collections, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "x"] + sys.argv[1:]
from extra.sqtt.heldout.misses import insts
from extra.sqtt.heldout.score import all_captures
def tok(rows, who): return [t for t, k in rows if k.startswith(who + " ") and "WAVESTART" not in k and "WAVEEND" not in k]
def lds(rows, who): return [t for t, k in rows if k.startswith(who + " ") and "LDS" in k]
def one(p):
  import execdiff as ed
  code = [i for pc, i in insts(p) if not (hasattr(i, "op") and i.op.name in ("S_DELAY_ALU", "S_ENDPGM"))]
  with contextlib.redirect_stdout(io.StringIO()): real, emu, _ = ed.emu_run(str(p))
  out = []
  for k, ins in enumerate(code):
    if repr(ins).split("(")[0] != "s_barrier": continue
    r = []
    for rows in (real, emu):
      w = [tok(rows, f"w{s}") for s in (0, 1)]
      if any(len(x) <= k + 1 for x in w): r = None; break
      bar, nxt = max(x[k] for x in w), min(x[k + 1] for x in w)
      nl = sum(1 for j in range(k) if "ds_" in repr(code[j]))  # LDS ops before the barrier
      offs = []
      for s in (0, 1):
        wl, pl = lds(rows, f"w{s}"), lds(rows, f"p{s}")
        offs.append(pl[nl - 1] - wl[nl - 1] if nl and len(pl) >= nl and len(wl) >= nl else None)
      r.append((nxt - bar, offs, w[1][k] - w[0][k]))
    if r: out.append((p.stem, k, *r))
  return out
if __name__ == "__main__":
  with mp.Pool(14) as pool: res = [x for o in pool.map(one, all_captures()) for x in o]
  c = collections.Counter(); off = collections.Counter()
  for stem, k, (rg, ro, rw), (eg, eo, ew) in res:
    c[(rg - eg)] += 1
    if rg != eg: off[(rg - eg, tuple(None if a is None or b is None else a - b for a, b in zip(ro, eo)), rw - ew)] += 1
  print("barriers", len(res), "gap real-emu:", dict(sorted(c.items())))
  print("(gap diff, p-offset diff real-emu per slot, w1-w0 barrier diff real-emu): count")
  for k, v in off.most_common(25): print(k, v)
  if "-v" in sys.argv:
    for x in res: print(*x)
