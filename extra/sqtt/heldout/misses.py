#!/usr/bin/env python3
"""Per-instruction timing table of the held-out captures: real delta, emulator delta, miss, for every scored wave.

Run with: python3 extra/sqtt/heldout/misses.py [--only KIND] [--miss] PKL ...   (default: every fit capture)
"""
import argparse, pickle, sys
from pathlib import Path
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from extra.sqtt.heldout.score import CAPTURES, _replay

def insts(pkl:Path) -> list:
  data = pickle.load(open(pkl, "rb"))
  lib = next(e for e in data if type(e).__name__ == "ProfileProgramEvent" and e.lib).lib
  image, sections, _ = elf_loader(lib)
  text = next(s for s in sections if s.name == ".text")
  code, out, pc = bytes(image[text.header.sh_addr:text.header.sh_addr + text.header.sh_size]), [], 0
  while pc < len(code):
    i = decode_inst(code[pc:pc + 16]); out.append((pc, i)); pc += i.size()
    if i.__class__.__name__ == "SOPP" and i.op.name == "S_ENDPGM": break
  return out

def table(pkl:Path, only:str|None=None, miss_only:bool=False, local:int=128):
  real, emu = _replay(pkl, local)
  code = [(pc, i) for pc, i in insts(pkl) if not (hasattr(i, "op") and i.op.name in ("S_DELAY_ALU", "S_ENDPGM"))]
  for rw, ew in zip(sorted(real), sorted(emu)):
    rs, es = real[rw], emu[ew]
    if [k for k, _ in rs] != [k for k, _ in es]: print(f"{pkl.stem} wave {rw[0]}: sequence differs, not scored"); continue
    print(f"== {pkl.stem} wave {rw[0]}: {sum(rd == ed for (_, rd), (_, ed) in zip(rs, es))}/{len(rs)}")
    ra = ea = 0
    for n, ((k, rd), (_, ed)) in enumerate(zip(rs, es)):
      ra, ea = ra + rd, ea + ed  # cycles since the wave's own WAVESTART
      name = repr(code[n][1]) if n < len(code) else ""
      if (only and k != only) or (miss_only and rd == ed): continue
      print(f"{n:>3} {k:<11} real {rd:>5} emu {ed:>5} {'' if rd == ed else f'{ed - rd:+d}':>5}  @{ra:>5} {ea:>5} pc {code[n][0] if n < len(code) else 0:>3x}  {name}")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("pkls", nargs="*"); ap.add_argument("--only"); ap.add_argument("--miss", action="store_true")
  ap.add_argument("--local", type=int, default=128, help="workgroup size the capture was launched with")
  a = ap.parse_args()
  for p in [Path(p) for p in a.pkls] or sorted(CAPTURES.glob("seed*.pkl"), key=lambda p: int(p.stem[4:])): table(p, a.only, a.miss, a.local)
