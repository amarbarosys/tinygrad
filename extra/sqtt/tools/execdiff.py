# real vs emulator timeline of the traced SIMD: instruction tokens, VALU/SALU exec (ALUEXEC), LDS service (VMEMEXEC)
# usage: python3 extra/sqtt/tools/execdiff.py PKL [NAME=VAL ...]   absolute cycles, emu shifted to wave 0's first wait
import sys, os, re, ctypes, pickle, types
TAG = os.environ.get("TAG")
from pathlib import Path
from tinygrad.helpers import Context, getenv
from tinygrad.renderer.amd.sqtt import decode, INST, VALUINST, IMMEDIATE, IMMEDIATE_MASK, WAVESTART, WAVEEND, VMEMEXEC, ALUEXEC
import test.mockgpu.amd.emu as emu_mod
from test.mockgpu.amd.emu import run_asm, sqtt_traces
from tinygrad.runtime.support.elf import elf_loader
src = open("test/mockgpu/amd/sqtt_enc.py").read()
hook = next(l + '\n' for l in src.split('\n') if l.startswith('        valu_free[simd[w]], s["valu_exec"] = ex + '))
assert hook in src
src = src.replace('\n' + hook, '\n' + hook + '        LOG.append((ex, "EXEC_VALU", simd[w], rank[w], cycle, inst, base, sorted(reads), sorted(writes), dict(s["reg_ready"]), valub))\n', 1)
h2 = '          for r in writes: s["reg_ready"][r] = s["salu_exec"] + SALU_DEP_LAT\n'
assert h2 in src
src = src.replace(h2, h2 + '          LOG.append((s["salu_exec"], "EXEC_SALU", simd[w], rank[w], cycle))\n')
h3 = '        lds_starts[simd[r[1]]].append(t)\n'
src = src.replace(h3, h3 + '        LOG.append((t, "LDS_SVC", simd[r[1]], rank[r[1]]))\n')
enc = types.ModuleType("enc_patched"); enc.__dict__["LOG"] = []
exec(compile(src, "sqtt_enc_patched", "exec"), enc.__dict__)
for a in [x for x in sys.argv[2:] if "=" in x]: k, v = a.split("=", 1); setattr(enc, k, eval(v))
emu_mod._make_sqtt_encoder = enc.make_encoder

def real_rows(data):
  for e in data:
    if type(e).__name__ != "ProfileSQTTEvent": continue
    out = []
    for p in decode(e.blob):
      if isinstance(p, IMMEDIATE_MASK):
        for m in range(16):
          if p.mask & (1 << m): out.append((p._time, f"w{m} IMMEDIATE"))
      elif isinstance(p, (IMMEDIATE, VALUINST)): out.append((p._time, f"w{p.wave} {type(p).__name__}"))
      elif isinstance(p, INST):
        k = p.op.name; out.append((p._time, f"{'p' if k.startswith('OTHER_') else 'w'}{p.wave} {k.removeprefix('OTHER_')}"))
      elif isinstance(p, VMEMEXEC): out.append((p._time, f"   EXEC {p.src.name}"))
      elif isinstance(p, ALUEXEC): out.append((p._time, f"   EXEC {p.src.name}"))
      elif isinstance(p, (WAVESTART, WAVEEND)) and p.simd in (0, 2): out.append((p._time, f"{'w' if p.simd == 0 else 'p'}{p.wave} {type(p).__name__}"))
    if any("VALUINST" in r[1] for r in out): return out
  return []

def emu_run(pkl):
  from extra.sqtt.heldout.score import _replay
  data = pickle.load(open(pkl, "rb"))
  real = real_rows(data)
  os.environ["SQTT_TRACE"] = "1"
  import io, contextlib
  # find chosen parity via _replay, then rerun that parity alone with logging
  buf = io.StringIO()
  with contextlib.redirect_stderr(buf): _replay(Path(pkl))
  chosen = int(re.search(r"SQTT_CHOSEN (\d)", buf.getvalue()).group(1))
  os.environ.pop("SQTT_TRACE")
  lib = next(e for e in data if type(e).__name__ == "ProfileProgramEvent" and e.lib).lib
  image, sections, _ = elf_loader(lib)
  text = next(s for s in sections if s.name == ".text")
  code = bytes(image[text.header.sh_addr:text.header.sh_addr + text.header.sh_size])
  bufs = [(ctypes.c_uint8 * (1 << 20))() for _ in range(4)]
  args = (ctypes.c_uint64 * 64)(*[ctypes.addressof(b) for b in bufs])
  cbuf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  os.environ["SQTT_SIMD_PARITY"], os.environ["SQTT_LDS_PARITY"] = str(chosen & 1), str(chosen >> 1); getenv.cache_clear()
  enc.LOG.clear(); sqtt_traces.clear()
  with Context(PROFILE=1, ASM_CALL=0):
    run_asm(ctypes.addressof(cbuf), len(code), 1, 1, 1, 128, 1, 1, ctypes.addressof(args))
  e = real_rows([types.SimpleNamespace(blob=b) for b in sqtt_traces]) if False else None
  rows = []
  for b in sqtt_traces:
    for p in decode(b):
      if isinstance(p, IMMEDIATE_MASK):
        for m in range(16):
          if p.mask & (1 << m): rows.append((p._time, f"w{m} IMMEDIATE"))
      elif isinstance(p, (IMMEDIATE, VALUINST)): rows.append((p._time, f"w{p.wave} {type(p).__name__}"))
      elif isinstance(p, INST):
        k = p.op.name; rows.append((p._time, f"{'p' if k.startswith('OTHER_') else 'w'}{p.wave} {k.removeprefix('OTHER_')}"))
      elif isinstance(p, (WAVESTART, WAVEEND)) and p.simd in (0, 2): rows.append((p._time, f"{'w' if p.simd == 0 else 'p'}{p.wave} {type(p).__name__}"))
    if rows: break
  # emu exec log: only the traced SIMD (0) execs are comparable with ALUEXEC; LDS service of both SIMDs
  for t, k, sm, wv, *_ in enc.LOG:
    if k.startswith("EXEC") and sm == 0: rows.append((t - (2 if k == "EXEC_SALU" else 0), f"   EXEC {k[5:]}"))
    if k == "EXEC_VALU" and sm == 0 and TAG: rows.append((t, f"  ~r{wv} {repr(_[1]).split('(')[0][2:14]}@{_[0]}"))  # real SALU exec token: issue + 2
    if k == "LDS_SVC": rows.append((t, f"   EXEC {'LDS' if sm == 0 else 'LDS_ALT'}"))
  return real, sorted(rows), chosen

if __name__ == "__main__":
  real, emu, chosen = emu_run(sys.argv[1])
  AT = int(os.environ.get("AT", "-1"))
  def first_wait(rows):
    if AT < 0: return next(t for t, k in rows if k == "w0 IMMEDIATE")
    w0 = [t for t, k in rows if k.startswith("w0 ") and "WAVESTART" not in k]
    return w0[AT]
  off = first_wait(real) - first_wait(emu)
  # merge VALU+SALU exec in one cycle into VALU_SALU for emu
  from collections import defaultdict
  def norm(rows, shift):
    d = defaultdict(set)
    for t, k in rows: d[t + shift].add(k)
    for t, ks in d.items():
      if "   EXEC VALU" in ks and "   EXEC SALU" in ks: ks -= {"   EXEC VALU", "   EXEC SALU"}; ks.add("   EXEC VALU_SALU")
    return d
  R, E = norm(real, 0), norm(emu, off)
  t0 = first_wait(real)
  print(f"parity {chosen}, emu shifted by {off}")
  lo, hi = int(os.environ.get("LO", "-5")), int(os.environ.get("HI", "100000"))
  for t in sorted(set(R) | set(E)):
    if not lo <= t - t0 <= hi: continue
    r, e = sorted(R.get(t, ())), sorted(E.get(t, ()))
    mark = "" if r == e else "  <>"
    print(f"{t - t0:>5} | {', '.join(r):<45} | {', '.join(e):<45}{mark}")
