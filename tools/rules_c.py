#!/usr/bin/env python3
# Role c: classify PatternMatcher rewrite fns as PURE / HELPER / DYNAMIC (AST heuristics; spot-checked by hand in c-verdict.md).
# Rules: the 10 biggest matchers in a-census.tsv + every rule at a b-fired.tsv top-90% time location.
# usage: python tools/rules_c.py   (reads .work/team-rules/{a-census,b-fired}.tsv; writes c-classes.tsv, c-b-share.tsv)
import ast, builtins, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / ".work" / "team-rules"
PURE_TYPES = {"Ops", "UOp", "dtypes", "GroupOp", "AddrSpace", "UPat", "sint", "PtrDType", "DType", "Invalid", "UPat"}
NOT_HELPER = set(dir(builtins)) | {"self", "math", "struct", "functools", "itertools", "operator"}  # stdlib, not repo helpers
# ctx = Python renderer/dict, not a UOp
DYN_RE = re.compile(r"\b(getattr|setattr|hasattr|eval|exec|globals|locals|vars)\(|\bnonlocal\b|\bglobal\b|__dict__|__class__|"
                    r"KernelInfo|CallInfo|ParamArg|TracingKey|graph_rewrite|\bctx\[|\bctx\.get\(|\bctx\.(render|exec|copy|emit|lower)\w*")
MUT_CALLS = {"dict", "defaultdict", "list", "set", "Counter", "OrderedDict"}

def die(step, why):
  print(f"FAIL {step}: {why}", file=sys.stderr)
  sys.exit(1)

def read_tsv(p):
  rows = [l.split("\t") for l in p.read_text().splitlines()]
  return rows[0], rows[1:]

class Mod:
  """One parsed source file: module-level matchers, rule tuples, defs, mutable globals, tinygrad imports."""
  def __init__(self, path):
    self.rel, self.src = str(path.relative_to(ROOT)), path.read_text()
    self.tree = ast.parse(self.src)
    self.defs = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}
    self.assigns, self.mutable, self.imported = {}, set(), set()
    for n in self.tree.body:
      if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
        nm, v = n.targets[0].id, n.value
        self.assigns[nm] = n
        if isinstance(v, (ast.Dict, ast.List, ast.Set)) or (isinstance(v, ast.Call) and getattr(v.func, "id", "") in MUT_CALLS):
          self.mutable.add(nm)
      if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("tinygrad"):
        self.imported |= {a.asname or a.name for a in n.names}
  def tuples_of(self, expr, out):
    if isinstance(expr, ast.BinOp):
      self.tuples_of(expr.left, out)
      self.tuples_of(expr.right, out)
    elif isinstance(expr, ast.Call) and getattr(expr.func, "id", "") == "PatternMatcher" and expr.args: self.tuples_of(expr.args[0], out)
    elif isinstance(expr, (ast.List, ast.Tuple)):
      for e in expr.elts:
        if isinstance(e, ast.Tuple) and len(e.elts) >= 2: out.append((e.elts[0], e.elts[1], e.lineno))
    return out
  def rule_starts(self):
    """(lineno, matcher_name, fn_node) for every rule tuple in any PatternMatcher([...]) in the file.
    Name = nearest enclosing assignment target (`x = `, `Cls.x += `), else (inline)."""
    res, owner = [], {}
    for n in ast.walk(self.tree):
      if isinstance(n, (ast.Assign, ast.AugAssign)):
        t = n.targets[0] if isinstance(n, ast.Assign) else n.target
        nm = t.id if isinstance(t, ast.Name) else getattr(t, "attr", "(inline)")
        for c in ast.walk(n.value): owner.setdefault(id(c), nm)
    for n in ast.walk(self.tree):
      if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "PatternMatcher" and n.args:
        for _, fn, ln in self.tuples_of(n.args[0], []): res.append((ln, owner.get(id(n), "(inline)"), fn))
    return res

def classify(mod, fn):
  if isinstance(fn, ast.Lambda): body, args = fn.body, {a.arg for a in fn.args.args}
  elif isinstance(fn, ast.Name) and fn.id in mod.defs: body, args = mod.defs[fn.id], {a.arg for a in mod.defs[fn.id].args.args}
  elif isinstance(fn, ast.Name): return "HELPER", f"rewrite fn `{fn.id}` lives outside this file; port needs it too"
  else: return "DYNAMIC", "rewrite is a non-name, non-lambda expression (partial / callable object)"
  seg = ast.get_source_segment(mod.src, body) or ""
  if (m := DYN_RE.search(seg)): return "DYNAMIC", f"python-only token `{m.group(0).strip()}` in rewrite body"
  bound = set(args)
  for n in ast.walk(body):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): bound.add(n.id)
  calls = {n.func.id for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
  free = {n.id for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - bound
  if (mut := free & mod.mutable): return "DYNAMIC", f"closure over mutable module state {sorted(mut)}"
  objs = {c for c in calls if c[:1].isupper() and c not in PURE_TYPES}
  if objs: return "DYNAMIC", f"constructs Python runtime object(s) {sorted(objs)}"
  helpers = {c for c in calls | free if c not in bound and c not in PURE_TYPES and c not in NOT_HELPER
             and (c in mod.defs or c in mod.imported or c[:1].islower())}
  helpers = {h for h in helpers if h not in ("lambda",)}
  if helpers: return "HELPER", f"calls repo helper(s) {sorted(helpers)}"
  return "PURE", "builds UOps from matched UOps/consts only"

def main():
  _, cen = read_tsv(TR / "a-census.tsv")
  cen = [r for r in cen if len(r) >= 7 and r[0] != "SUM"]
  # n_rules is blank for the 4 `dynamic-fn` rows (matchers built by a call, not a literal): census says unresolved, count as 0
  top10 = sorted(cen, key=lambda r: -int(r[3] or 0))[:10]
  mods = {}
  def getmod(rel):
    if rel not in mods:
      p = ROOT / rel
      if not p.exists(): die("load", f"{rel} missing")
      mods[rel] = Mod(p)
    return mods[rel]
  rows = {}  # (rel, lineno) -> (matcher, class, reason)
  for rel, ln0, nm, *_ in top10:
    m = getmod(rel)
    for ln, mname, fn in m.rule_starts():
      if mname == nm: rows.setdefault((rel, ln), (nm,) + classify(m, fn))
  bh, b = read_tsv(TR / "b-fired.tsv")
  # exclusive time only: incl_ms double-counts nested graph_rewrite/realize work under driver rules
  ms_col = bh.index("excl_ms") if "excl_ms" in bh else bh.index("time_matched_ms")
  for r in b: r[5] = r[ms_col]
  total = sum(float(r[5]) for r in b)
  top, cum = [], 0.0
  for r in b:
    top.append(r)
    cum += float(r[5])
    if cum >= 0.9 * total: break
  b_map = []
  for r in top:
    rel, Ls = r[1].rsplit(":", 1)  # b writes repo-relative paths
    L = int(Ls)
    hit = None
    if (ROOT / rel).exists():
      m = getmod(rel)
      near = [s for s in m.rule_starts() if abs(s[0] - L) <= 3]
      if near: hit = (m, min(near, key=lambda s: abs(s[0] - L)))
    if hit is None:
      b_map.append((r, None, None))
      continue
    m, (ln, mname, fn) = hit
    rows.setdefault((m.rel, ln), (mname,) + classify(m, fn))
    b_map.append((r, m.rel, ln))
  out = ["file:line\tmatcher\tclass\treason"] + [f"{rel}:{ln}\t{v[0]}\t{v[1]}\t{v[2]}" for (rel, ln), v in sorted(rows.items())]
  (TR / "c-classes.tsv").write_text("\n".join(out) + "\n")
  ms_by_rule, unmapped = defaultdict(float), 0.0
  for r, rel, ln in b_map:
    if rel is None: unmapped += float(r[5])
    else: ms_by_rule[(rel, ln)] += float(r[5])
  share = defaultdict(float)
  for key, ms in ms_by_rule.items(): share[rows[key][1]] += ms
  top90 = sum(float(r[5]) for r in top)
  lines = ["class\trules\tb_ms\tpct_of_all_rewrite_ms\tpct_of_top90_ms"]
  for cls in ("PURE", "HELPER", "DYNAMIC"):
    n = sum(1 for v in rows.values() if v[1] == cls)
    lines.append(f"{cls}\t{n}\t{share[cls]:.1f}\t{100*share[cls]/total:.1f}\t{100*share[cls]/top90:.1f}")
  lines.append(f"UNMAPPED\t-\t{unmapped:.1f}\t{100*unmapped/total:.1f}\t{100*unmapped/top90:.1f}")
  lines.append(f"ALL_B_REWRITE_MS\t-\t{total:.1f}\t100.0\t-")
  (TR / "c-b-share.tsv").write_text("\n".join(lines) + "\n")
  per, nrules = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
  for key, v in rows.items(): nrules[v[0]][v[1]] += 1
  for key, ms in ms_by_rule.items(): per[rows[key][0]][rows[key][1]] += ms
  ml = ["matcher\tPURE_ms\tHELPER_ms\tDYNAMIC_ms\tPURE_n\tHELPER_n\tDYNAMIC_n"]
  for mname in sorted(per, key=lambda k: -sum(per[k].values())):
    cls = ("PURE", "HELPER", "DYNAMIC")
    ml.append(f"{mname}\t" + "\t".join(f"{per[mname][c]:.1f}" for c in cls) + "\t" + "\t".join(str(nrules[mname][c]) for c in cls))
  (TR / "c-by-matcher.tsv").write_text("\n".join(ml) + "\n")
  print("\n".join(lines))
  print(f"rules={len(rows)} top10={[r[2] for r in top10]} b_top90_rows={len(top)} unmapped_b_rows={sum(1 for x in b_map if x[1] is None)}")

if __name__ == "__main__":
  main()
