#!/usr/bin/env python3
"""Score the emulator's SQTT timing against real gfx1100 captures of the held-out probe.

Each capture is replayed in the emulator; per traced-SIMD wave the packet kinds must line up, then every
instruction delta (cycles since the wave's previous packet, the first from WAVESTART) is compared exactly.

Run with: python3 extra/sqtt/heldout/score.py [PKL ...]   (default: every capture in captures/)
"""
import ctypes, os, pickle, sys
from collections import defaultdict
from pathlib import Path
from tinygrad.helpers import Context, getenv
from tinygrad.renderer.amd.sqtt import decode, INST, VALUINST, IMMEDIATE, IMMEDIATE_MASK, WAVESTART, WAVEEND
from tinygrad.runtime.support.elf import elf_loader
from test.mockgpu.amd.emu import run_asm, sqtt_traces

CAPTURES = Path(__file__).parent / "captures"

def waves(blob:bytes) -> dict[tuple[int, bool], list[tuple[str, int]]]:
  """{(wave, paired_simd): [(kind, delta)]}. OTHER_* packets belong to the same slot on the paired SIMD."""
  last:dict[tuple[int, bool], int] = {}
  out:dict[tuple[int, bool], list[tuple[str, int]]] = defaultdict(list)
  for p in decode(blob):
    side = {0: False, 2: True}.get(getattr(p, "simd", None))
    if isinstance(p, WAVESTART):
      if side is not None: last[(p.wave, side)] = p._time
    elif isinstance(p, IMMEDIATE_MASK):
      for m in range(16):
        if p.mask & (1 << m) and (m, False) in last:
          out[(m, False)].append(("IMMEDIATE", p._time - last[(m, False)])); last[(m, False)] = p._time
    elif isinstance(p, (INST, VALUINST, IMMEDIATE)):
      kind = p.op.name if isinstance(p, INST) else type(p).__name__
      key = (p.wave, kind.startswith("OTHER_"))
      if key not in last: continue
      out[key].append((kind.removeprefix("OTHER_"), p._time - last[key])); last[key] = p._time
    elif isinstance(p, WAVEEND) and side is not None:
      key = (p.wave, side)
      if key in last and (not side or out[key]): out[key].append(("WAVEEND", p._time - last[key]))
      last.pop(key, None)
  return {w: s for w, s in out.items() if s}

def primary(blobs) -> dict:
  return next(({w: s for w, s in ws.items() if not w[1]} for b in blobs if (ws := waves(b))), {})

def _replay(pkl:Path, local:int=128):
  data = pickle.load(open(pkl, "rb"))
  real = primary(e.blob for e in data if type(e).__name__ == "ProfileSQTTEvent")
  lib = next(e for e in data if type(e).__name__ == "ProfileProgramEvent" and e.lib).lib
  image, sections, _ = elf_loader(lib)
  text = next(s for s in sections if s.name == ".text")
  code = bytes(image[text.header.sh_addr:text.header.sh_addr + text.header.sh_size])
  bufs = [(ctypes.c_uint8 * (1 << 20))() for _ in range(4)]
  args = (ctypes.c_uint64 * 64)(*[ctypes.addressof(b) for b in bufs])
  cbuf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  for parity in (0, 1):  # which SIMD of the pair gets wave 0 is not modeled yet
    os.environ["SQTT_SIMD_PARITY"] = str(parity); getenv.cache_clear()
    sqtt_traces.clear()
    with Context(PROFILE=1, ASM_CALL=0):
      run_asm(ctypes.addressof(cbuf), len(code), 1, 1, 1, local, 1, 1, ctypes.addressof(args))
    emu = primary(list(sqtt_traces))
    n = 0
    for rw, ew in zip(sorted(real), sorted(emu)):
      rs, es = real[rw], emu[ew]
      if [k for k, _ in rs] != [k for k, _ in es]: continue
      n += len(rs)
    if n: return real, emu
  return real, {}

def all_captures() -> list[Path]:
  """Every capture: seed1-11 (randprobe), mul1-10 (randprobe_mul), v2_100-119 (randprobe_v2), in that order."""
  key = lambda p: (p.stem.split("_")[0].rstrip("0123456789"), int(p.stem.split("_")[-1].lstrip("seedmul")))
  return sorted(CAPTURES.glob("*.pkl"), key=key)

DRAM_STALL = 300  # no SRAM latency reaches this; a longer delta is a DRAM access or an instruction fetch from memory

def dram(seq:list[tuple[str, int]], i:int) -> bool:
  """A delta that waits on memory: the s_load, the wait right after it, the end after the global store, a fetch stall."""
  k, d = seq[i]
  return k in ("SMEM_RD", "WAVEEND") or (i > 0 and seq[i - 1][0] == "SMEM_RD") or d >= DRAM_STALL

def score(pkl:Path, local:int=128, sram:bool=False) -> tuple[int, int, int]:
  """(exact deltas, scored deltas, total abs error) for one capture; tries both SIMD parities. sram: skip DRAM deltas."""
  real, emu = _replay(pkl, local)
  hit = n = err = 0
  for rw, ew in zip(sorted(real), sorted(emu)):
    rs, es = real[rw], emu[ew]
    if [k for k, _ in rs] != [k for k, _ in es]: continue
    for i, ((_, rd), (_, ed)) in enumerate(zip(rs, es)):
      if sram and dram(rs, i): continue
      n += 1; hit += rd == ed; err += abs(rd - ed)
  return hit, n, err

def score_detail(pkl:Path, local:int=128) -> dict[str, tuple[int, int, int]]:
  """Return per-packet-kind (exact, scored, absolute error) for one capture."""
  real, emu = _replay(pkl, local)
  out = defaultdict(lambda: [0, 0, 0])
  for rw, ew in zip(sorted(real), sorted(emu)):
    rs, es = real[rw], emu[ew]
    if [k for k, _ in rs] != [k for k, _ in es]: continue
    for (rk, rd), (_, ed) in zip(rs, es):
      out[rk][0] += rd == ed; out[rk][1] += 1; out[rk][2] += abs(rd - ed)
  return {k: tuple(v) for k, v in out.items()}

if __name__ == "__main__":
  sram = "--sram" in sys.argv
  pkls = [Path(p) for p in sys.argv[1:] if p != "--sram"] or all_captures()
  th = tn = 0
  for p in pkls:
    hit, n, err = score(p, sram=sram)
    th, tn = th + hit, tn + n
    print(f"{p.stem:>7}: {hit:>3}/{n:<3} exact ({100 * hit / max(n, 1):.1f}%), mean abs error {err / max(n, 1):.2f} cycles")
  print(f"  total: {th}/{tn} exact ({100 * th / max(tn, 1):.1f}%){' SRAM only' if sram else ''}")
