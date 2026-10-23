# per capture: k-th LDS op of traced wave w (slot s) vs k-th LDS op of paired wave p (same slot), real and emulator
import sys, io, contextlib, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "x"] + sys.argv[1:]
import execdiff as ed
from extra.sqtt.heldout.score import all_captures
def lds(rows, who):
  return [t for t, k in rows if k.startswith(who + " ") and ("LDS" in k)]
tot = collections.Counter(); per = {}
for p in [Path(a) for a in sys.argv[2:] if a.endswith(".pkl")] or all_captures():
  with contextlib.redirect_stdout(io.StringIO()): real, emu, chosen = ed.emu_run(str(p))
  out = []
  for slot in (0, 1):
    rw, rp, ew, ep = lds(real, f"w{slot}"), lds(real, f"p{slot}"), lds(emu, f"w{slot}"), lds(emu, f"p{slot}")
    n = min(len(rw), len(rp), len(ew), len(ep))
    for k in range(n): out.append((slot, k, rp[k] - rw[k], ep[k] - ew[k]))
  per[p.stem] = out
  for s_, k, r, e in out: tot[(r, e)] += 1
print("(real p-w, emu p-w): count")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:30]: print(k, v)
first = collections.Counter((out[0][2], out[0][3], out[0][0]) for out in per.values() if out)
print("first LDS (real p-w, emu p-w, slot):", dict(first.most_common()))
first1 = collections.Counter((o[2], o[3]) for out in per.values() for o in out if o[1] == 0)
print("k=0 both slots:", dict(first1.most_common()))
if "-v" in sys.argv:
  for st, out in per.items(): print(st, [(s_, k, r, e) for s_, k, r, e in out if r != e][:12])
