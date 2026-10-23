# parallel SRAM scoring of every capture in extra/sqtt/heldout/captures (or given pkls), with sqtt_enc constant overrides
# usage: python3 extra/sqtt/tools/sweep.py [NAME=VAL ...] [--agg N] [--miss] [pkl ...]
import sys, collections, multiprocessing as mp
from pathlib import Path
OV = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
PK = [Path(a) for a in sys.argv[1:] if a.endswith(".pkl")]
def init():
  import re, types, test.mockgpu.amd.sqtt_enc as enc, test.mockgpu.amd.emu as emu
  inner = [a[1:] for a in OV if a.startswith("@")]
  if inner:  # constants local to make_encoder: patch the source tuple assignment and load it as the encoder
    src = open("test/mockgpu/amd/sqtt_enc.py").read()
    for a in inner:
      k, v = a.split("=", 1)
      m = re.search(r"^(\s+)([A-Z_0-9, ]+) = ([^#\n]+)", "\n".join(l for l in src.split("\n") if re.match(rf"\s+[A-Z_0-9, ]*\b{k}\b[A-Z_0-9, ]* = ", l)), re.M)
      names, vals = [x.strip() for x in m.group(2).split(",")], [x.strip() for x in m.group(3).split(",")]
      vals[names.index(k)] = v
      src = src.replace(m.group(0), f"{m.group(1)}{m.group(2)} = {', '.join(vals)}", 1)
    mod = types.ModuleType("enc_patched"); exec(compile(src, "enc_patched", "exec"), mod.__dict__)
    emu._make_sqtt_encoder = mod.make_encoder; enc = mod
  for a in OV:
    if not a.startswith("@"): k, v = a.split("=", 1); setattr(enc, k, eval(v))
def one(p):
  from extra.sqtt.heldout.score import _replay, dram
  from extra.sqtt.heldout.misses import insts
  real, emu = _replay(p)
  code = [(pc, i) for pc, i in insts(p) if not (hasattr(i, "op") and i.op.name in ("S_DELAY_ALU", "S_ENDPGM"))]
  rows, hit, n = [], 0, 0
  for rw, ew in zip(sorted(real), sorted(emu)):
    rs, es = real[rw], emu[ew]
    if [k for k, _ in rs] != [k for k, _ in es]: rows.append((p.stem, rw[0], -1, "SEQ", 0, 0, "", "")); continue
    for i, ((k, rd), (_, ed)) in enumerate(zip(rs, es)):
      if dram(rs, i): continue
      n += 1; hit += rd == ed
      if rd != ed:
        nm = lambda j: (repr(code[j][1]).split("(")[0] if 0 <= j < len(code) else "-")
        rows.append((p.stem, rw[0], i, k, rd, ed, nm(i), nm(i - 1)))
  return p.stem, hit, n, rows
if __name__ == "__main__":
  from extra.sqtt.heldout.score import all_captures
  pk = PK or all_captures()
  with mp.Pool(14, initializer=init) as pool: res = pool.map(one, pk)
  sets = collections.defaultdict(lambda: [0, 0])
  for stem, h, n, _ in res:
    g = "v2b" if stem.startswith("v2_") and int(stem[3:]) >= 120 else stem.rstrip("0123456789_")
    sets[g][0] += h; sets[g][1] += n
  th, tn = sum(v[0] for v in sets.values()), sum(v[1] for v in sets.values())
  print(" ".join(OV) or "base", "|", " | ".join(f"{g} {h}/{n}" for g, (h, n) in sets.items()), f"| ALL {th}/{tn} {100*th/tn:.2f}%")
  rows = [r for *_, rs in res for r in rs]
  if "--miss" in sys.argv:
    for r in rows: print(*r)
  if "--agg" in sys.argv:
    N = int(sys.argv[sys.argv.index("--agg") + 1])
    by = collections.defaultdict(list)
    for r in rows: by[(r[3], r[6], r[7])].append(r[5] - r[4])
    cls = collections.Counter(r[3] for r in rows); print("by kind:", dict(cls.most_common()))
    for key, e in sorted(by.items(), key=lambda kv: -len(kv[1]))[:N]:
      print(f"{len(e):>4} {key[0]:<11} {key[1]:<22} after {key[2]:<22} {dict(collections.Counter(e).most_common(6))}")
