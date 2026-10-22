#!/usr/bin/env python3
"""Emulator SQTT timing on held-out kernels: random SRAM probes captured on a 7900 XTX.

seed1-11 (randprobe), mul1-10 (randprobe_mul) and v2_100-119 (randprobe_v2) were each captured once, before the rules
that followed were written; by now every set has been used to fit, so a fresh set must be captured to validate.
Each capture must keep at least its recorded number of exact deltas; a timing change that helps the fitted
examples but costs held-out accuracy fails here. Every measured constant is checked to be falsifiable: moved by
one cycle either way it must lose exact deltas on a named capture.

Run with: DEV=MOCK+AMD python -m pytest test/amd/test_sqtt_heldout.py
"""
import unittest
from extra.sqtt.heldout.score import CAPTURES, all_captures, score
import test.mockgpu.amd.sqtt_enc as sqtt_enc

# capture -> (exact deltas at least, scored deltas)
FLOOR = {"mul1": (76, 94), "mul2": (78, 84), "mul3": (90, 100), "mul4": (88, 100), "mul5": (84, 102),
         "mul6": (77, 90), "mul7": (86, 106), "mul8": (87, 104), "mul9": (96, 116), "mul10": (80, 90),
         "seed1": (85, 94), "seed2": (78, 84), "seed3": (89, 104), "seed4": (93, 108), "seed5": (87, 100),
         "seed6": (76, 90), "seed7": (94, 108), "seed8": (92, 102), "seed9": (102, 112), "seed10": (79, 90),
         "seed11": (96, 116), "v2_100": (85, 94), "v2_101": (86, 102), "v2_102": (89, 96), "v2_103": (96, 116),
         "v2_104": (73, 86), "v2_105": (89, 106), "v2_106": (95, 116), "v2_107": (92, 104), "v2_108": (78, 92),
         "v2_109": (95, 110), "v2_110": (89, 110), "v2_111": (86, 94), "v2_112": (87, 114), "v2_113": (72, 86),
         "v2_114": (85, 96), "v2_115": (77, 100), "v2_116": (91, 100), "v2_117": (86, 102), "v2_118": (85, 98),
         "v2_119": (86, 102)}
# constant -> capture whose exact count drops when the constant moves by one cycle either way
MUTANTS = {"LDS_TO_VALU_LOAD": "mul2", "LDS_TO_VALU_STORE": "mul4", "VALUB_EXEC_EXTRA": "mul3", "SALU_AFTER_VALUB": "mul4",
           "BARRIER_WINDOW": "v2_111", "BARRIER_BUSY": "v2_102", "BARRIER_SERIAL": "v2_109", "MEM_AFTER_EXEC": "seed2"}

class TestSQTTHeldout(unittest.TestCase):
  def test_captures(self):
    self.assertEqual(sorted(p.stem for p in all_captures()), sorted(FLOOR))
    for name, (want, n) in FLOOR.items():
      with self.subTest(capture=name):
        hit, got_n, _ = score(CAPTURES / f"{name}.pkl")
        self.assertEqual(got_n, n, f"{name}: scored deltas changed (a wave's packet sequence moved)")
        self.assertGreaterEqual(hit, want, f"{name}: {hit}/{n} exact, recorded {want}")

  def test_mutants(self):
    """Every measured constant must be falsifiable by the captures."""
    for name, cap in MUTANTS.items():
      base = getattr(sqtt_enc, name)
      want = score(CAPTURES / f"{cap}.pkl")[0]
      for delta in (-1, 1):
        with self.subTest(name=name, delta=delta):
          setattr(sqtt_enc, name, base + delta)
          try: self.assertLess(score(CAPTURES / f"{cap}.pkl")[0], want)
          finally: setattr(sqtt_enc, name, base)

if __name__ == "__main__":
  unittest.main()
