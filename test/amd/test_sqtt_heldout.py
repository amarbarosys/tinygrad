#!/usr/bin/env python3
"""Emulator SQTT timing on held-out kernels: random SRAM probes captured on a 7900 XTX.

seed1-11 (randprobe), mul1-10 (randprobe_mul) and v2_100-139 (randprobe_v2) were each captured once, before the rules
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
FLOOR = {"mul1": (80, 94), "mul2": (76, 84), "mul3": (92, 100), "mul4": (87, 100), "mul5": (88, 102), "mul6": (78, 90),
         "mul7": (92, 106), "mul8": (89, 104), "mul9": (94, 116), "mul10": (82, 90), "seed1": (83, 94),
         "seed2": (76, 84), "seed3": (95, 104), "seed4": (95, 108), "seed5": (90, 100), "seed6": (78, 90),
         "seed7": (98, 108), "seed8": (92, 102), "seed9": (102, 112), "seed10": (81, 90), "seed11": (98, 116),
         "v2_100": (88, 94), "v2_101": (94, 102), "v2_102": (89, 96), "v2_103": (98, 116), "v2_104": (77, 86),
         "v2_105": (91, 106), "v2_106": (98, 116), "v2_107": (96, 104), "v2_108": (82, 92), "v2_109": (99, 110),
         "v2_110": (90, 110), "v2_111": (86, 94), "v2_112": (89, 114), "v2_113": (75, 86), "v2_114": (85, 96),
         "v2_115": (89, 100), "v2_116": (90, 100), "v2_117": (92, 102), "v2_118": (86, 98), "v2_119": (87, 102),
         "v2_120": (105, 116), "v2_121": (82, 94), "v2_122": (95, 114), "v2_123": (76, 84), "v2_124": (86, 102),
         "v2_125": (83, 100), "v2_126": (73, 88), "v2_127": (80, 88), "v2_128": (87, 98), "v2_129": (105, 118),
         "v2_130": (107, 120), "v2_131": (93, 108), "v2_132": (99, 112), "v2_133": (100, 122), "v2_134": (101, 114),
         "v2_135": (97, 116), "v2_136": (92, 106), "v2_137": (75, 86), "v2_138": (86, 96), "v2_139": (66, 76)}
# constant -> capture whose exact count drops when the constant moves by one cycle either way
MUTANTS = {"LDS_TO_VALU_LOAD": "mul2", "LDS_TO_VALU_STORE": "mul4", "SALU_AFTER_VALUB": "mul4",
           "BARRIER_WINDOW": "v2_111", "BARRIER_BUSY": "v2_102", "BARRIER_SERIAL": "v2_109", "MEM_AFTER_EXEC": "seed2", "VALUB_OCC": "v2_104",
           "COLD_READ": "seed3"}

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
