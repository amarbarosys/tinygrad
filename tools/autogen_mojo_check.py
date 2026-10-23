#!/usr/bin/env python3
# Oracle check for tools/autogen_mojo.py: fill random structs with random bytes, read every scalar and bitfield
# from ctypes and from the generated Mojo, compare.
# usage: python tools/autogen_mojo_check.py BINDINGS_DIR WORK_DIR module [module ...]
import ctypes, random, subprocess, sys, pathlib, importlib
from tinygrad.runtime.support import c
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from autogen_mojo import ident, mty

def main():
  bdir, wdir, mods = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
  wdir.mkdir(parents=True, exist_ok=True)
  for name in mods:
    # per-module seed: a module's sample never depends on which modules ran before it in this process
    rng = random.Random(name)
    mod = importlib.import_module(f"tinygrad.runtime.autogen.{name}")
    sts = [(k, v) for k, v in vars(mod).items() if isinstance(v, type) and issubclass(v, c.Struct) and hasattr(v, "_real_fields_")
           and k == v.__name__ and v.__module__ == mod.__name__ and v.SIZE > 0]
    # all bitfield structs (likeliest to break) plus a random sample of the rest
    bf = [kv for kv in sts if any(len(f) > 3 for f in kv[1]._real_fields_)][:60]
    sts = bf + rng.sample([kv for kv in sts if kv not in bf], min(40, len(sts) - len(bf)))
    if not sts:
      print(f"OK {name}: 0 structs, 0 fields")
      continue
    body, expect = [], []
    for k, cls in sts:
      raw = bytes(rng.randrange(256) for _ in range(cls.SIZE))
      obj = cls.from_buffer_copy(raw)
      body += ["    if True:", f"        var s = {ident(k)}()"] + [f"        s._mem[{i}] = {b}" for i, b in enumerate(raw)]
      for fname, typ, off, *bits in cls._real_fields_:
        if not bits and mty(typ) is None: continue
        if not bits and typ in (ctypes.c_float, ctypes.c_double): continue  # NaN payloads make float text compare meaningless
        v = getattr(obj, fname)
        if not bits and mty(typ) == "Int": v = int.from_bytes(raw[off:off+8], sys.byteorder, signed=True)
        elif not bits and typ is ctypes.c_char: v = int.from_bytes(v, sys.byteorder, signed=True)
        elif not bits and typ is ctypes.c_bool: v = raw[off]
        body.append(f"        print(\"{k}.{fname}\", s.{ident(fname)}())")
        expect.append(f"{k}.{fname} {int(v)}")
    src = wdir / f"check_{name}.mojo"
    # explicit names: `import *` skips names starting with _ (eg. __mbstate_t)
    src.write_text(f"from {name} import {', '.join(ident(k) for k, _ in sts)}\n\ndef main():\n" + "\n".join(body) + "\n    pass\n")
    exe = wdir / f"check_{name}"
    r = subprocess.run(["mojo", "build", "-I", str(bdir), str(src), "-o", str(exe)], capture_output=True, text=True, timeout=600)
    if r.returncode: sys.exit(f"FAIL build {name}: {r.stderr[-800:]}")
    got = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
    bad = [(e, g) for e, g in zip(expect, got) if e != g] + ([("len", f"{len(expect)} vs {len(got)}")] if len(expect) != len(got) else [])
    if bad: sys.exit(f"FAIL {name}: {len(bad)} mismatches, first {bad[:3]}")
    print(f"OK {name}: {len(sts)} structs, {len(expect)} fields")

if __name__ == "__main__": main()
