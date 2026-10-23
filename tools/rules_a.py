#!/usr/bin/env python3
# Role a: static census of PatternMatcher(...) constructions under tinygrad/ (skips tinygrad/runtime/autogen).
# AST pass finds every construction site and counts literal rules; import pass reads len(pm.patterns) of module-level matchers.
# usage: python tools/rules_a.py   (writes .work/team-rules/a-census.tsv)
import ast, importlib, sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "tinygrad"
OUT = ROOT / ".work" / "team-rules" / "a-census.tsv"
SKIP = ("runtime/autogen",)
# matchers built inside a function: (file, function) -> rule count depends on call args, so no static count
DYNAMIC = {("tinygrad/codegen/decomp/op.py", "get_simplifying_rewrite_patterns()"), ("tinygrad/codegen/decomp/op.py", "get_late_rewrite_patterns()"),
           ("tinygrad/codegen/decomp/transcendental.py", "get_transcendental_patterns()"), ("tinygrad/uop/ops.py", "__add__()")}

def fail(step:str, why:str) -> None:
  print(f"FAIL {step}: {why}", file=sys.stderr)
  sys.exit(1)

def py_files() -> list[Path]:
  out = []
  for p in sorted(PKG.rglob("*.py")):
    rel = p.relative_to(ROOT).as_posix()
    if not any(s in rel for s in SKIP): out.append(p)
  return out

def rule_kind(fn:ast.expr) -> str:
  if isinstance(fn, ast.Lambda): return "lambda"
  if isinstance(fn, (ast.Name, ast.Attribute)): return "def"
  return "other"

class Site:
  def __init__(self, file:str, line:int, name:str, n_rules:int|None, n_lambda:int, n_def:int, method:str, module:str|None):
    self.file, self.line, self.name, self.n_rules = file, line, name, n_rules
    self.n_lambda, self.n_def, self.method, self.module = n_lambda, n_def, method, module

def ast_sites(path:Path) -> list[Site]:
  rel = path.relative_to(ROOT).as_posix()
  module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
  tree = ast.parse(path.read_text(), filename=rel)
  # map each PatternMatcher(...) call to the name it is bound to (module-level assign target, or enclosing def)
  owner: dict[int, tuple[str, bool]] = {}
  for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
      for sub in ast.walk(node.value): owner[id(sub)] = (node.targets[0].id, True)
  for fdef in ast.walk(tree):
    if isinstance(fdef, (ast.FunctionDef, ast.AsyncFunctionDef)):
      for sub in ast.walk(fdef): owner.setdefault(id(sub), (f"{fdef.name}()", False))
  sites: list[Site] = []
  for node in ast.walk(tree):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "PatternMatcher"): continue
    name, module_level = owner.get(id(node), ("<module-expr>", False))
    arg = node.args[0] if node.args else None
    if isinstance(arg, ast.List):
      n_lambda = n_def = 0
      other = 0
      for elt in arg.elts:
        if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
          k = rule_kind(elt.elts[1])
          if k == "lambda": n_lambda += 1
          elif k == "def": n_def += 1
          else: other += 1
        else: other += 1
      method = "ast-literal" if not other else "ast-literal+other"
      sites.append(Site(rel, node.lineno, name, len(arg.elts), n_lambda, n_def, method, module if module_level else None))
    else:
      # non-literal argument (variable, comprehension, concatenation): resolved by import pass
      sites.append(Site(rel, node.lineno, name, None, 0, 0, "unresolved", module if module_level else None))
  return sites

def import_counts() -> dict[tuple[str, str], tuple[int, int, int]]:
  # (module, global name) -> (len(patterns), lambda count, def count) for every PatternMatcher global
  sys.path.insert(0, str(ROOT))
  from tinygrad.uop.ops import PatternMatcher
  res: dict[tuple[str, str], tuple[int, int, int]] = {}
  for path in py_files():
    mod_name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    if mod_name.endswith(".__init__"): mod_name = mod_name[: -len(".__init__")]
    try:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = importlib.import_module(mod_name)
    except Exception as e:
      print(f"WARN import {mod_name}: {type(e).__name__}: {e}", file=sys.stderr)
      continue
    for gname, val in vars(mod).items():
      if not isinstance(val, PatternMatcher): continue
      lam = dfn = 0
      for _, fn in val.patterns:
        if getattr(fn, "__name__", "") == "<lambda>": lam += 1
        elif callable(fn): dfn += 1
      res[(mod_name, gname)] = (len(val.patterns), lam, dfn)
  return res

def main() -> None:
  sites: list[Site] = []
  for path in py_files(): sites.extend(ast_sites(path))
  if not sites: fail("ast", "no PatternMatcher( constructions found")
  print(f"ast sites: {len(sites)}  literal: {sum(s.n_rules is not None for s in sites)}  unresolved: {sum(s.n_rules is None for s in sites)}")

  imp = import_counts()
  print(f"import globals: {len(imp)}")

  # cross-check: module-level literal matchers vs import len(patterns); differences are composed matchers (a + b)
  agree = disagree = 0
  for s in sites:
    if s.module and s.n_rules is not None and (s.module, s.name) in imp:
      n_imp, _, _ = imp[(s.module, s.name)]
      if n_imp == s.n_rules: agree += 1
      else:
        disagree += 1
        print(f"  note {s.file}:{s.line} {s.name}: ast {s.n_rules} vs import {n_imp} (composed with +)")
  print(f"cross-check literal==import: agree {agree} / disagree {disagree}")

  # resolve unresolved sites from the import pass; the rest are matchers built inside a function (count depends on args)
  for s in sites:
    if s.n_rules is None and s.module and (s.module, s.name) in imp:
      n, lam, dfn = imp[(s.module, s.name)]
      s.n_rules, s.n_lambda, s.n_def, s.method = n, lam, dfn, "import-len"
    elif s.n_rules is None and (s.file, s.name) in DYNAMIC:
      s.method = "dynamic-fn"
    elif s.n_rules is None:
      s.method = "unresolved"

  sites.sort(key=lambda s: (s.file, s.line))
  OUT.parent.mkdir(parents=True, exist_ok=True)
  rows = ["file\tline\tname\tn_rules\tn_lambda\tn_def\tmethod"]
  for s in sites:
    rows.append(f"{s.file}\t{s.line}\t{s.name}\t{'' if s.n_rules is None else s.n_rules}\t{s.n_lambda}\t{s.n_def}\t{s.method}")
  resolved = [s for s in sites if s.n_rules is not None]
  tot_rules = sum(s.n_rules for s in resolved)
  tot_lam = sum(s.n_lambda for s in resolved)
  tot_def = sum(s.n_def for s in resolved)
  rows.append(f"SUM\t{len(sites)} sites\t\t{tot_rules}\t{tot_lam}\t{tot_def}\t{sum(s.method == 'dynamic-fn' for s in sites)} dynamic-fn excluded")
  OUT.write_text("\n".join(rows) + "\n")
  print(f"wrote {OUT}")
  print(f"SUM sites={len(sites)} n_rules={tot_rules} lambda={tot_lam} def={tot_def} unresolved={len(sites)-len(resolved)}")
  unknown = [s for s in sites if s.method == "unresolved"]
  if unknown: fail("resolve", f"{len(unknown)} sites neither resolved nor allowlisted: " + ", ".join(f"{s.file}:{s.line}" for s in unknown))
  print(f"dynamic-fn sites excluded from SUM: {sum(s.method == 'dynamic-fn' for s in sites)}")

if __name__ == "__main__":
  main()
