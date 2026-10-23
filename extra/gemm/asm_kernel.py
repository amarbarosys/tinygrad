# RDNA3 assembly builder used by the asm DSL tests: collects instructions, resolves branch labels, emits s_waitcnt
from tinygrad.runtime.autogen.amd.rdna3.ins import s_waitcnt

class Kernel:
  def __init__(self): self.instructions, self.labels, self.pos = [], {}, 0
  def label(self, name): self.labels[name] = self.pos

  def emit(self, inst, target=None):
    self.instructions.append(inst)
    inst._target, inst._pos = target, self.pos
    self.pos += inst.size()
    return inst

  def waitcnt(self, lgkm=None, vm=None):
    """Wait for memory operations. lgkm=N waits until N lgkm ops remain, vm=N waits until N vmem ops remain."""
    vmcnt, lgkmcnt, expcnt = vm if vm is not None else 63, lgkm if lgkm is not None else 63, 7
    waitcnt = (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)
    self.emit(s_waitcnt(simm16=waitcnt))

  def finalize(self):
    """Patch branch offsets and return the finalized instruction list."""
    for inst in self.instructions:
      if inst._target is None: continue
      offset_dwords = (self.labels[inst._target] - inst._pos - inst.size()) // 4
      if not -32768 <= offset_dwords <= 32767: raise ValueError(f"branch to '{inst._target}' offset {offset_dwords} exceeds simm16 range")
      inst.simm16 = offset_dwords
    return self.instructions
