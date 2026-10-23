#!/usr/bin/env python3
"""Emulator SQTT timing on held-out kernels: random SRAM probes captured on a 7900 XTX.

seed1-11 (randprobe), mul1-10 (randprobe_mul) and v2_100-199 (randprobe_v2) were each captured once, before the rules
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
FLOOR = {"mul1": (85, 94), "mul2": (77, 84), "mul3": (94, 100), "mul4": (94, 100), "mul5": (90, 102), "mul6": (84, 90),
         "mul7": (96, 106), "mul8": (91, 104), "mul9": (108, 116), "mul10": (84, 90), "seed1": (87, 94),
         "seed2": (78, 84), "seed3": (93, 104), "seed4": (99, 108), "seed5": (88, 100), "seed6": (84, 90),
         "seed7": (102, 108), "seed8": (96, 102), "seed9": (99, 112), "seed10": (82, 90), "seed11": (103, 116),
         "v2_100": (88, 94), "v2_101": (96, 102), "v2_102": (90, 96), "v2_103": (101, 116), "v2_104": (77, 86),
         "v2_105": (97, 106), "v2_106": (107, 116), "v2_107": (96, 104), "v2_108": (84, 92), "v2_109": (101, 110),
         "v2_110": (95, 110), "v2_111": (88, 94), "v2_112": (97, 114), "v2_113": (79, 86), "v2_114": (87, 96),
         "v2_115": (91, 100), "v2_116": (94, 100), "v2_117": (96, 102), "v2_118": (92, 98), "v2_119": (91, 102),
         "v2_120": (106, 116), "v2_121": (88, 94), "v2_122": (101, 114), "v2_123": (78, 84), "v2_124": (92, 102),
         "v2_125": (89, 100), "v2_126": (78, 88), "v2_127": (82, 88), "v2_128": (89, 98), "v2_129": (106, 118),
         "v2_130": (104, 120), "v2_131": (99, 108), "v2_132": (102, 112), "v2_133": (105, 122), "v2_134": (102, 114),
         "v2_135": (98, 116), "v2_136": (98, 106), "v2_137": (80, 86), "v2_138": (90, 96), "v2_139": (70, 76),
         "v2_140": (88, 94), "v2_141": (110, 118), "v2_142": (108, 120), "v2_143": (86, 94), "v2_144": (113, 124),
         "v2_145": (100, 108), "v2_146": (88, 96), "v2_147": (90, 96), "v2_148": (105, 114), "v2_149": (83, 92),
         "v2_150": (94, 106), "v2_151": (100, 108), "v2_152": (106, 114), "v2_153": (116, 124), "v2_154": (88, 96),
         "v2_155": (104, 112), "v2_156": (104, 116), "v2_157": (108, 116), "v2_158": (94, 100), "v2_159": (99, 110),
         "v2_160": (80, 86), "v2_161": (97, 112), "v2_162": (90, 96), "v2_163": (99, 108), "v2_164": (82, 90),
         "v2_165": (74, 84), "v2_166": (90, 96), "v2_167": (88, 98), "v2_168": (80, 88), "v2_169": (79, 86),
         "v2_170": (95, 106), "v2_171": (91, 100), "v2_172": (94, 104), "v2_173": (90, 106), "v2_174": (84, 92),
         "v2_175": (102, 112), "v2_176": (78, 86), "v2_177": (91, 98), "v2_178": (80, 90), "v2_179": (93, 100),
         "v2_180": (89, 96), "v2_181": (88, 94), "v2_182": (96, 114), "v2_183": (82, 90), "v2_184": (100, 108),
         "v2_185": (84, 94), "v2_186": (100, 108), "v2_187": (100, 110), "v2_188": (96, 106), "v2_189": (102, 112),
         "v2_190": (84, 102), "v2_191": (105, 116), "v2_192": (104, 114), "v2_193": (97, 114), "v2_194": (81, 90),
         "v2_195": (103, 114), "v2_196": (100, 124), "v2_197": (76, 86), "v2_198": (79, 86), "v2_199": (98, 104)}
# constant -> capture whose exact count drops when the constant moves by one cycle either way
MUTANTS = {"LDS_TO_VALU_LOAD": "mul2", "LDS_TO_VALU_STORE": "mul4", "SALU_AFTER_VALUB": "mul4",
           "BARRIER_WINDOW": "v2_111", "BARRIER_BUSY": "v2_102", "BARRIER_SERIAL": "v2_109", "MEM_AFTER_EXEC": "seed2", "VALUB_OCC": "v2_104",
           "COLD_READ": "seed3", "VMEM_AFTER_SALU": "mul6", "STORE_LOAD_DELAY": "v2_125", "ICACHE_WAVE_GAP": "v2_133", "SCC_HOLD": "v2_188"}

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
