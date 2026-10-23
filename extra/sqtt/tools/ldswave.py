# teacher-forced LDS arbitration with wave identity: does a busy server stay with the same wave or the same SIMD?
import pickle, collections, sys
from extra.sqtt.heldout.score import all_captures
from tinygrad.renderer.amd.sqtt import decode, INST, VALUINST, VMEMEXEC
ACC = 3
def streams(p):
  for e in pickle.load(open(p, "rb")):
    if type(e).__name__ != "ProfileSQTTEvent": continue
    pk = list(decode(e.blob))
    if not any(isinstance(x, VALUINST) for x in pk): continue
    req = {0: [], 2: []}
    for x in pk:
      if isinstance(x, INST) and "LDS" in x.op.name:
        req[2 if x.op.name.startswith("OTHER_") else 0].append((x._time, 2 if x.op.name.endswith("_2") else 1, x.wave))
    svc = [(x._time, 0 if x.src.name == "LDS" else 2) for x in pk if isinstance(x, VMEMEXEC) and x.src.name in ("LDS", "LDS_ALT")]
    return req, svc
def rules(h, last, lastw, t, free, first, run):
  a0, a2 = h[0][0], h[2][0]
  out = {}
  base = (0 if h[0][1] == 2 else 2) if a0 == a2 and h[0][1] != h[2][1] else first if a0 == a2 else None
  busy = free >= t
  newest = max(h, key=lambda s: h[s][0]); oldest = min(h, key=lambda s: h[s][0])
  samew = next((s for s in h if (s, h[s][2]) == lastw), None)
  out["model"] = base if base is not None else last if busy and last in h else newest
  out["wave_newest"] = base if base is not None else samew if busy and samew is not None else newest
  out["wave_oldest"] = base if base is not None else samew if busy and samew is not None else oldest
  out["wave_any_newest"] = base if base is not None else samew if samew is not None else newest
  out["wave_any_oldest"] = base if base is not None else samew if samew is not None else oldest
  out["simd_oldest"] = base if base is not None else last if busy and last in h else oldest
  for k in (2, 3, 4):
    out[f"cap{k}"] = base if base is not None else (last if run < k else 2 - last) if busy and last in h else newest
  for a in (2, 3, 4, 5, 6):
    old = [sm for sm in h if sm != last and t - (h[sm][0] + ACC) >= a]
    out[f"starve{a}"] = base if base is not None else old[0] if busy and old else last if busy and last in h else newest
  return out
score = collections.Counter(); n = 0
for p in all_captures():
  req, svc = streams(p)
  if sum(map(len, req.values())) != len(svc): continue
  q = {sm: list(r) for sm, r in req.items()}; last = lastw = None; free = -10**9; first = None; run = 0
  for t, sm in svc:
    h = {s: q[s][0] for s in (0, 2) if q[s] and q[s][0][0] + ACC <= t}
    if sm not in h: r = q[sm].pop(0); run = run + 1 if sm == last else 1; last, lastw, free = sm, (sm, r[2]), t + 1; continue
    if len(h) == 2:
      if h[0][0] == h[2][0] and h[0][1] == h[2][1] and first is None: first = sm
      rr = rules(h, last, lastw, t, free, first if first is not None else sm, run)
      for k, v in rr.items(): score[k] += v == sm
      if rr["model"] != sm and "-v" in sys.argv:
        nx = {s_: [(a - t + ACC, o) for a, o, _ in q[s_][:3]] for s_ in (0, 2)}
        print(f"{p.stem:<7} t={t} real s{sm} last s{last}w{lastw[1] if lastw else '-'} run{run} free-t {free - t:+d} first s{first}",
              "| s0", nx[0], "| s2", nx[2])
      n += 1
    r = q[sm].pop(0); run = run + 1 if sm == last else 1; last, lastw, free = sm, (sm, r[2]), t + r[1]
print("contested", n, dict(score.most_common()))
