# after an instruction-fetch stall (delta >= 200) on both traced waves at the same instruction: resume of w1 - w0, real
# and emulator, and what the next instruction after the resumed one is
import sys, io, contextlib, collections, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "x"] + sys.argv[1:]
from extra.sqtt.heldout.score import all_captures
from extra.sqtt.heldout.misses import insts
def tok(rows, who): return [t for t, k in rows if k.startswith(who + " ") and "WAVESTART" not in k and "WAVEEND" not in k]
def one(p):
  import execdiff as ed
  code = [i for pc, i in insts(p) if not (hasattr(i, "op") and i.op.name in ("S_DELAY_ALU", "S_ENDPGM"))]
  with contextlib.redirect_stdout(io.StringIO()): real, emu, _ = ed.emu_run(str(p))
  rw, ew = [tok(real, f"w{s}") for s in (0, 1)], [tok(emu, f"w{s}") for s in (0, 1)]
  out = []
  for k in range(1, min(map(len, rw + ew))):
    if all(w[k] - w[k - 1] >= 200 for w in rw):
      out.append((p.stem, k, rw[1][k] - rw[0][k], ew[1][k] - ew[0][k], rw[1][k - 1] - rw[0][k - 1], repr(code[k]).split("(")[0] if k < len(code) else "-"))
  return out
if __name__ == "__main__":
  with mp.Pool(14) as pool: res = [x for o in pool.map(one, all_captures()) for x in o]
  print("stalls", len(res), "(real w1-w0 resume, emu): ", dict(collections.Counter((r[2], r[3]) for r in res).most_common()))
  print("(real w1-w0 before stall -> resume)", dict(collections.Counter((r[4], r[2]) for r in res).most_common(20)))
  late = [r for r in res if r[1] > 1]
  print("k>1:", len(late), dict(collections.Counter((r[2], r[3]) for r in late).most_common()))
  for r in late: print(*r)
