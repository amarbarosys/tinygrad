# teacher-forced LDS arbitration: at each hardware grant, rebuild the queue state from the real history and ask the model
# rule what it would pick; tabulate contested decisions (both SIMDs have an accepted head) by visible state
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
        req[2 if x.op.name.startswith("OTHER_") else 0].append((x._time, 2 if x.op.name.endswith("_2") else 1))
    svc = [(x._time, 0 if x.src.name == "LDS" else 2) for x in pk if isinstance(x, VMEMEXEC) and x.src.name in ("LDS", "LDS_ALT")]
    return req, svc
def model(h, last, t, free):
  a0, a2 = h[0][0], h[2][0]
  if a0 == a2 and h[0][1] != h[2][1]: return 0 if h[0][1] == 2 else 2
  if a0 == a2: return None  # same-kind tie: a fixed SIMD per dispatch (hidden), taken from the first tie
  if last in h and free >= t: return last
  return max(h, key=lambda s: h[s][0])
rows = []
if __name__ != "__main__": sys.argv = sys.argv[:1]
for p in [a for a in sys.argv[1:] if a.endswith(".pkl")] or all_captures():
  req, svc = streams(p)
  if sum(map(len, req.values())) != len(svc): rows.append((p.stem, "COUNT")); continue
  q = {sm: list(r) for sm, r in req.items()}; last = None; free = -10**9; lastsimd_tie = None; run = 0
  for k, (t, sm) in enumerate(svc):
    h = {s: q[s][0] for s in (0, 2) if q[s] and q[s][0][0] + ACC <= t}
    if sm not in h: rows.append((p.stem, "NOTREADY", k)); q[sm].pop(0); last, free = sm, t + 1; continue
    if len(h) == 2:
      pick = model(h, last, t, free)
      if pick is None: pick = lastsimd_tie if lastsimd_tie is not None else sm
      busy = "busy" if free >= t else f"idle{t - free}"
      d = h[2][0] - h[0][0]  # p head accept - w head accept
      kinds = ("st" if h[0][1] == 2 else "ld") + "/" + ("st" if h[2][1] == 2 else "ld")
      tie = h[0][0] == h[2][0]
      rows.append((p.stem, "DEC", k, pick == sm, busy, max(-6, min(6, d)), kinds, "last" + str(last), run, tie))
      if tie: lastsimd_tie = sm
    r = q[sm].pop(0)
    run = run + 1 if sm == last else 1
    last, free = sm, t + r[1]
dec = [r for r in rows if r[1] == "DEC"]
print("captures", len({r[0] for r in rows}), "skipped", collections.Counter(r[1] for r in rows if r[1] != "DEC"))
print(f"contested {len(dec)}, model right {sum(r[3] for r in dec)}")
by = collections.defaultdict(lambda: [0, 0])
for r in dec:
  key = (r[4] if r[4] == "busy" else "idle", r[5], r[6], r[7] if r[4] == "busy" else "")
  by[key][0] += 1; by[key][1] += r[3]
print(f"{'state':<10}{'p-w':>4} {'w/p':<6}{'last':<7}{'n':>5}{'right':>6}")
for key, (n, ok) in sorted(by.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
  if n - ok: print(f"{key[0]:<10}{key[1]:>4} {key[2]:<6}{key[3]:<7}{n:>5}{ok:>6}")
