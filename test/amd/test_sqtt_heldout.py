#!/usr/bin/env python3
"""Emulator SQTT timing on held-out kernels: random SRAM probes captured on a 7900 XTX.

seed1-11 (randprobe), mul1-10 (randprobe_mul) and v2_100-159 (randprobe_v2) were each captured once, before the rules
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
FLOOR = {"mul1": (80, 94), "mul2": (76, 84), "mul3": (90, 100), "mul4": (90, 100), "mul5": (90, 102), "mul6": (80, 90),
         "mul7": (94, 106), "mul8": (89, 104), "mul9": (96, 116), "mul10": (84, 90), "seed1": (85, 94),
         "seed2": (76, 84), "seed3": (95, 104), "seed4": (97, 108), "seed5": (90, 100), "seed6": (80, 90),
         "seed7": (100, 108), "seed8": (92, 102), "seed9": (102, 112), "seed10": (82, 90), "seed11": (103, 116),
         "v2_100": (88, 94), "v2_101": (96, 102), "v2_102": (89, 96), "v2_103": (98, 116), "v2_104": (77, 86),
         "v2_105": (91, 106), "v2_106": (100, 116), "v2_107": (96, 104), "v2_108": (80, 92), "v2_109": (99, 110),
         "v2_110": (92, 110), "v2_111": (88, 94), "v2_112": (90, 114), "v2_113": (79, 86), "v2_114": (87, 96),
         "v2_115": (89, 100), "v2_116": (90, 100), "v2_117": (92, 102), "v2_118": (90, 98), "v2_119": (88, 102),
         "v2_120": (105, 116), "v2_121": (88, 94), "v2_122": (96, 114), "v2_123": (76, 84), "v2_124": (91, 102),
         "v2_125": (86, 100), "v2_126": (77, 88), "v2_127": (78, 88), "v2_128": (89, 98), "v2_129": (105, 118),
         "v2_130": (103, 120), "v2_131": (93, 108), "v2_132": (101, 112), "v2_133": (102, 122), "v2_134": (101, 114),
         "v2_135": (98, 116), "v2_136": (92, 106), "v2_137": (77, 86), "v2_138": (86, 96), "v2_139": (66, 76),
         "v2_140": (86, 94), "v2_141": (108, 118), "v2_142": (99, 120), "v2_143": (86, 94), "v2_144": (113, 124),
         "v2_145": (98, 108), "v2_146": (90, 96), "v2_147": (90, 96), "v2_148": (99, 114), "v2_149": (83, 92),
         "v2_150": (92, 106), "v2_151": (90, 108), "v2_152": (106, 114), "v2_153": (114, 124), "v2_154": (86, 96),
         "v2_155": (100, 112), "v2_156": (99, 116), "v2_157": (102, 116), "v2_158": (90, 100), "v2_159": (98, 110)}
# constant -> capture whose exact count drops when the constant moves by one cycle either way
MUTANTS = {"LDS_TO_VALU_LOAD": "mul2", "LDS_TO_VALU_STORE": "mul4", "SALU_AFTER_VALUB": "mul4",
           "BARRIER_WINDOW": "v2_111", "BARRIER_BUSY": "v2_102", "BARRIER_SERIAL": "v2_109", "MEM_AFTER_EXEC": "seed2", "VALUB_OCC": "v2_104",
           "COLD_READ": "seed3", "VMEM_AFTER_SALU": "mul6"}

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
