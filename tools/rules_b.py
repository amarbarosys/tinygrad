#!/usr/bin/env python3
# Runtime rule census: run prof_llama with match stats on, then dump per-rule tried/matched/time to TSV.
# Exclusive time = time inside a rule's match fn minus time of pattern attempts it triggered (nested graph_rewrite/realize/hcq).
# usage: python tools/rules_b.py MODEL.gguf [steps]   (wrap with gpu-wait; writes .work/team-rules/b-fired.tsv)
import os, sys, time, importlib
from pathlib import Path
os.environ.setdefault("CACHELEVEL", "0")
os.environ.setdefault("TRACK_MATCH_STATS", "2")
os.environ.setdefault("PRINT_MATCH_STATS", "0")  # we dump the table ourselves
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".work/team-rules/b-fired.tsv"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from tinygrad import Tensor  # noqa: E402  (after env is set)
from tinygrad.device import Compiled  # noqa: E402
from tinygrad.uop.ops import match_stats, TRACK_MATCH_STATS, TrackedPatternMatcher, active_rewrites, UOp  # noqa: E402

# CPU stall guard: default 30 s signal wait tripped once in a run (device.py:374). Raised so a stall is recorded, not fatal.
Compiled.wait_timeout_ms = 300000.0

STEPS: list[int] = []
_orig_item = Tensor.item
def _item(self):
  ret = _orig_item(self)
  STEPS.append(1)
  return ret
Tensor.item = _item

# nested-time accounting: _acc = [inclusive seconds of direct child attempts, count of all descendant attempts]
_acc = [0.0, 0]
EXCL: dict = {}  # UPat -> [nested_s, nested_attempts]

def _rewrite(self, uop, ctx=None):
  if len(pats:=self.pdict.get(uop.op, [])):
    ler = {u.op for u in uop.src}
    for p,match,early_reject in pats:
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      outer_t, outer_n = _acc
      _acc[0], _acc[1] = 0.0, 0
      t0 = time.perf_counter()
      try: ret = match(uop, ctx)
      except BaseException:
        _acc[0], _acc[1] = outer_t, outer_n
        raise
      inc = time.perf_counter()-t0
      inner_t, inner_n = _acc
      _acc[0], _acc[1] = outer_t+inc, outer_n+1+inner_n
      rec = EXCL.setdefault(p, [0.0, 0])
      rec[0] += inner_t
      rec[1] += inner_n
      if ret is not None and ret is not uop:
        match_stats[p][0] += 1
        match_stats[p][3] += inc
        if TRACK_MATCH_STATS >= 2 and isinstance(ret, UOp) and active_rewrites:
          active_rewrites[-1].matches.append((uop.trace_num, ret.trace_num, p.location, inc))
        return ret
      match_stats[p][2] += inc
  return None
TrackedPatternMatcher.rewrite = _rewrite

def dump():
  if not match_stats: raise SystemExit("FAIL dump: match_stats empty, TRACK_MATCH_STATS=%s not active" % TRACK_MATCH_STATS.value)
  rows = []
  for p, v in match_stats.items():
    if v[1] == 0: continue  # never tried
    incl = v[2] + v[3]
    nested, nested_n = EXCL.get(p, [0.0, 0])
    rows.append((incl - nested, incl, nested, nested_n, v[1], v[0], p.location))
  rows.sort(key=lambda r: -r[0])
  total = sum(r[0] for r in rows) or 1.0
  OUT.parent.mkdir(parents=True, exist_ok=True)
  cum, cover90 = 0.0, None
  with open(OUT, "w") as f:
    f.write("rank\tlocation\ttried\tmatched\texcl_ms\tincl_ms\tnested_ms\tnested_attempts\trecurses\n")
    for i, (ex, incl, nested, nested_n, tried, matched, loc) in enumerate(rows, 1):
      cum += ex
      if cover90 is None and cum >= 0.9 * total: cover90 = i
      rel = os.path.relpath(loc[0], ROOT)
      f.write(f"{i}\t{rel}:{loc[1]}\t{tried}\t{matched}\t{ex*1e3:.3f}\t{incl*1e3:.3f}\t{nested*1e3:.3f}\t{nested_n}\t{int(nested_n > 0)}\n")
  fired = sum(r[5] > 0 for r in rows)
  print(f"rules_b: steps_done={len(STEPS)} tried={len(rows)} fired={fired} recurse={sum(r[3] > 0 for r in rows)} "
        f"rules_for_90pct_excl={cover90} excl_total_ms={total*1e3:.1f} wrote {OUT}")

if __name__ == "__main__":
  prof = importlib.import_module("prof_llama")
  try: prof.main()
  finally: dump()
