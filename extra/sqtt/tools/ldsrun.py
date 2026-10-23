# teacher-forced: busy server, both SIMD heads waiting: how often does the grant switch SIMD, by run length and by
# the cycles the current SIMD has held the server (sum of occupancies in the run)
import collections, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.argv = sys.argv[:1]
from ldswave import streams, ACC
from extra.sqtt.heldout.score import all_captures
by = collections.defaultdict(lambda: [0, 0]); byc = collections.defaultdict(lambda: [0, 0]); bya = collections.defaultdict(lambda: [0, 0])
for p in all_captures():
  req, svc = streams(p)
  if sum(map(len, req.values())) != len(svc): continue
  q = {sm: list(r) for sm, r in req.items()}; last = None; free = -10**9; run = 0; held = 0; runstart = 0
  for t, sm in svc:
    h = {s: q[s][0] for s in (0, 2) if q[s] and q[s][0][0] + ACC <= t}
    if len(h) == 2 and free >= t and last in h and h[0][0] != h[2][0]:
      other = 2 - last
      sw = sm != last
      by[run][0] += 1; by[run][1] += sw
      byc[t - runstart][0] += 1; byc[t - runstart][1] += sw
      wait = t - (h[other][0] + ACC)  # how long the other SIMD's head has waited
      bya[(min(wait, 8), h[other][0] > h[last][0])][0] += 1; bya[(min(wait, 8), h[other][0] > h[last][0])][1] += sw
    if sm not in h and q[sm]: pass
    r = q[sm].pop(0)
    if sm != last: runstart = t; run = 1
    else: run += 1
    last, free = sm, t + r[1]
print("run -> (n, switched)", dict(sorted(by.items())))
print("cycles held -> (n, switched)", dict(sorted(byc.items())))
print("(other head waited, other head newer) -> (n, switched)", dict(sorted(bya.items())))
