# hardware only: global store issue - last SALU issue of the same wave, by the SALU run before it and what lies between
import collections, multiprocessing as mp
from extra.sqtt.heldout.score import all_captures, _replay
from extra.sqtt.heldout.misses import insts
def nm(i): return repr(i).split("(")[0]
def one(p):
  code = [i for pc, i in insts(p) if not (hasattr(i, "op") and i.op.name in ("S_DELAY_ALU", "S_ENDPGM"))]
  real, emu = _replay(p)
  out = []
  for (rw, rs), (ew, es) in zip(sorted(real.items()), sorted(emu.items())):
    if [k for k, _ in rs] != [k for k, _ in es]: continue
    rt, et = [], []; a = b = 0
    for (_, rd), (_, ed) in zip(rs, es): a += rd; b += ed; rt.append(a); et.append(b)
    for k, i in enumerate(code[:len(rt)]):
      if not nm(i).startswith("global_store"): continue
      j = max((j for j in range(k) if nm(code[j]).startswith("s_") and nm(code[j]) in ("s_mov_b32", "s_add_u32")), default=None)
      if j is None or k - j > 4: continue
      run = []
      x = j
      while x >= 0 and nm(code[x]) in ("s_mov_b32", "s_add_u32"): run.insert(0, nm(code[x])[2:5]); x -= 1
      between = "+".join(nm(code[m])[:9] for m in range(j + 1, k))
      prev = nm(code[x])[:9] if x >= 0 else "-"
      out.append(("/".join(run), prev, between, rt[k] - rt[j], et[k] - et[j]))
  return out
if __name__ == "__main__":
  with mp.Pool(14) as pool: res = [x for o in pool.map(one, all_captures()) for x in o]
  c = collections.defaultdict(collections.Counter)
  for run, prev, between, r, e in res: c[(run, prev, between)][(r, e)] += 1
  for k, v in sorted(c.items(), key=lambda kv: -sum(kv[1].values())):
    if k[2] == "s_waitcnt" and any(r < 12 for r, e in v): print(k, dict(v))
