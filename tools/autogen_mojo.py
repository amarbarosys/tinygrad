#!/usr/bin/env python3
# Emit Mojo bindings from tinygrad's generated ctypes modules (tinygrad/runtime/autogen/*.py).
# The Python module is the oracle: sizes, offsets, constants and signatures are read from it, never re-derived.
# usage: python tools/autogen_mojo.py OUT_DIR name [name ...]
import ctypes, functools, importlib, keyword, sys, pathlib
from tinygrad.runtime.support import c

SCALARS = {ctypes.c_uint8:"UInt8", ctypes.c_int8:"Int8", ctypes.c_uint16:"UInt16", ctypes.c_int16:"Int16", ctypes.c_uint32:"UInt32",
           ctypes.c_int32:"Int32", ctypes.c_uint64:"UInt64", ctypes.c_int64:"Int64", ctypes.c_float:"Float32", ctypes.c_double:"Float64",
           ctypes.c_char:"Int8", ctypes.c_bool:"UInt8"}
MOJO_KW = set(keyword.kwlist) | {"var", "ref", "out", "mut", "owned", "deinit", "inout", "read", "struct", "trait", "comptime", "fn", "def", "alias",
                                 "self", "Self", "raises", "unsafe", "imm", "borrowed"}
PTRS = (ctypes._Pointer, ctypes._CFuncPtr)

# field access goes through these so each generated accessor is one line
HELPERS = """
@always_inline
def _ld[d: DType](p: UnsafePointer[UInt8, _], off: Int) -> Scalar[d]:
    return (p + off).bitcast[Scalar[d]]()[]

@always_inline
def _st[d: DType](p: UnsafePointer[mut=True, UInt8, _], off: Int, x: Scalar[d]):
    (p + off).bitcast[Scalar[d]]()[] = x

@always_inline
def _mask(w: Int) -> UInt64:
    return ~UInt64(0) if w >= 64 else (UInt64(1) << UInt64(w)) - 1

@always_inline
def _bget(p: UnsafePointer[UInt8, _], off: Int, w: Int, bo: Int) -> UInt64:
    var v: UInt64 = 0
    for i in range((w + bo + 7) // 8):
        v |= UInt64(p[off + i]) << UInt64(8 * i)
    return (v >> UInt64(bo)) & _mask(w)

@always_inline
def _bset(p: UnsafePointer[mut=True, UInt8, _], off: Int, w: Int, bo: Int, x: UInt64):
    var m = _mask(w) << UInt64(bo)
    var v: UInt64 = 0
    for i in range((w + bo + 7) // 8):
        v |= UInt64(p[off + i]) << UInt64(8 * i)
    v = (v & ~m) | ((x << UInt64(bo)) & m)
    for i in range((w + bo + 7) // 8):
        p[off + i] = UInt8((v >> UInt64(8 * i)) & 0xFF)
"""

def ident(n:str) -> str: return n + "_" if n in MOJO_KW else n
def mty(t) -> str|None:
  # pointers and function pointers cross the boundary as addresses
  if t in SCALARS: return SCALARS[t]
  if t is ctypes.c_void_p or t is ctypes.c_char_p or (isinstance(t, type) and issubclass(t, PTRS)): return "Int"
  return None

def emit_struct(name:str, cls) -> list[str]:
  out = [f"struct {name}(Copyable, Movable):", f"    comptime SIZE = {cls.SIZE}", f"    var _mem: Array[UInt8, {max(cls.SIZE, 1)}]",
         "    def __init__(out self):", f"        self._mem = Array[UInt8, {max(cls.SIZE, 1)}](fill=0)",
         "    def _addr(self) -> Int:", "        return Int(self._mem.unsafe_ptr())"]
  for fname, typ, off, *bits in cls._real_fields_:
    f = ident(fname)
    if bits:
      w, bo = bits
      out += [f"    def {f}(self) -> UInt64:", f"        return _bget(self._mem.unsafe_ptr(), {off}, {w}, {bo})",
              f"    def set_{f}(mut self, x: UInt64):", f"        _bset(self._mem.unsafe_ptr(), {off}, {w}, {bo}, x)"]
    elif (t:=mty(typ)) is not None:
      st = "DType." + ("UInt64" if t == "Int" else t).lower()
      get = f"_ld[{st}](self._mem.unsafe_ptr(), {off})"
      out += [f"    def {f}(self) -> {t}:", f"        return {f'Int({get})' if t == 'Int' else get}",
              f"    def set_{f}(mut self, x: {t}):", f"        _st[{st}](self._mem.unsafe_ptr(), {off}, {'UInt64(x)' if t == 'Int' else 'x'})"]
    # arrays, nested records, unions: address of the field, the caller reinterprets
    else: out += [f"    def _addr_{f}(self) -> Int:", f"        return Int(self._mem.unsafe_ptr()) + {off}"]
  names = [l.split("(")[0].split()[-1] for l in out if l.startswith("    def ")] + ["SIZE", "_mem"]
  if len(names) != len(set(names)): raise ValueError(f"{name}: generated member names collide: {sorted({n for n in names if names.count(n) > 1})}")
  return out

def gen(name:str) -> str:
  mod = importlib.import_module(f"tinygrad.runtime.autogen.{name}")
  consts, structs, ioctls, fns, skipped = [], [], [], [], []
  for k, v in vars(mod).items():
    if k.startswith("__") and k.endswith("__"): continue
    if isinstance(v, type) and issubclass(v, c.Struct) and v.__module__ == mod.__name__ and hasattr(v, "_real_fields_") and k == v.__name__:
      structs += emit_struct(ident(k), v) + [""]
    elif isinstance(v, int) and not isinstance(v, bool):
      consts.append(f"comptime {ident(k)}{': UInt64' if v >= 1 << 63 else ''} = {v}")
    elif isinstance(v, functools.partial) and v.func is c._do_ioctl:
      idir, base, nr, st = v.args
      code = (base << 8) | nr if st is None else (idir << 30) | (ctypes.sizeof(st) << 16) | (base << 8) | nr
      ioctls.append(f"comptime {ident(k)} = {code}")
    elif callable(v) and hasattr(v, "argtypes") and hasattr(v, "__wrapped__"):
      rt = "NoneType" if v.restype is None else mty(v.restype)
      ats = [mty(a) for a in v.argtypes]
      if rt is None or None in ats:
        skipped.append(f"# skipped {k}: struct by value or unsupported type")
        continue
      ps = ", ".join(f"a{i}: {t}" for i, t in enumerate(ats))
      args = ", ".join(f"a{i}" for i in range(len(ats)))
      fns += [f"    def {ident(k)}(self{', ' if ps else ''}{ps}) -> {rt}:", f"        return self.h.call[\"{k}\", {rt}]({args})"]
  lines = [f"# generated by tools/autogen_mojo.py from tinygrad/runtime/autogen/{name}.py, do not edit", "from std.ffi import OwnedDLHandle", HELPERS]
  lines += consts + ioctls + [""] + structs
  if fns: lines += ["struct Lib(Movable):", "    var h: OwnedDLHandle", "    def __init__(out self, path: String) raises:",
                    "        self.h = OwnedDLHandle(path)"] + fns + [""]
  return "\n".join(lines + skipped) + "\n"

if __name__ == "__main__":
  out = pathlib.Path(sys.argv[1])
  out.mkdir(parents=True, exist_ok=True)
  for n in sys.argv[2:]:
    try: (out / f"{n}.mojo").write_text(gen(n))
    except Exception as e: print(f"FAIL {n}: {type(e).__name__}: {e}", file=sys.stderr)
