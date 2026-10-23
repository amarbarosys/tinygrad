#!/usr/bin/env python3
# Generate the Brain copy of the SQTT timing model from the code: one note (neuron) per constant of
# test/mockgpu/amd/sqtt_enc.py with its value (strength), comment, location and introducing commit; links (synapses) to the
# constants it shares a code line with. Output is rewritten whole each run: never edit it by hand.
# usage: python3 rules2brain.py REPO OUTDIR
import re, subprocess, sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
src_path = repo / "test/mockgpu/amd/sqtt_enc.py"
lines = src_path.read_text().split("\n")
ASSIGN = re.compile(r"^(\s*)([A-Z][A-Z0-9_]*(?:\s*,\s*[A-Z][A-Z0-9_]*)*)\s*=\s*(.+?)\s*(?:#\s*(.*))?$")

def git(*a): return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()

consts: dict[str, dict] = {}
for i, l in enumerate(lines):
  m = ASSIGN.match(l)
  if not m or "(" in m.group(3).split("#")[0][:1]: continue
  names = [n.strip() for n in m.group(2).split(",")]
  vals = [v.strip() for v in re.split(r",(?![^{\[(]*[}\])])", m.group(3))]
  if len(vals) != len(names) and len(vals) != 1: continue
  comment = [m.group(4)] if m.group(4) else []
  j = i - 1
  while j >= 0 and lines[j].strip().startswith("#"):
    comment.insert(0, lines[j].strip().lstrip("# ")); j -= 1
  for k, n in enumerate(names):
    consts[n] = dict(value=vals[k] if len(vals) == len(names) else vals[0], line=i + 1, comment=" ".join(comment),
                     scope="module" if not m.group(1) else "encoder")

# synapses: constants used together on a code line (outside their own definition)
links: dict[str, set] = {n: set() for n in consts}
defs = {c["line"] - 1 for c in consts.values()}
for i in range(len(lines)):
  if i in defs: continue
  code = " ".join(lines[j].split("#")[0] for j in range(i, min(i + 3, len(lines))) if j not in defs)  # one rule spans a few lines
  used = [n for n in consts if re.search(rf"\b{n}\b", code)]
  for a in used:
    links[a] |= {b for b in used if b != a}

out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.md"): f.unlink()
head = git("rev-parse", "--short", "HEAD")
for n, c in sorted(consts.items()):
  intro = git("log", "--reverse", "--format=%h %s", "-S", n, "--", "test/mockgpu/amd/sqtt_enc.py").split("\n")[0]
  body = git("log", "-1", "--format=%b", intro.split(" ")[0]) if intro else ""
  syn = " ".join(f"[[sqtt-{x}]]" for x in sorted(links[n]))
  (out / f"sqtt-{n}.md").write_text(
    f"---\ntype: neuron\nmodel: sqtt-timing\nstrength: {c['value']}\nscope: {c['scope']}\nsource: tinygrad/{src_path.relative_to(repo)}:{c['line']}\n"
    f"head: {head}\nintroduced: \"{intro}\"\ngenerated: rules2brain.py (do not edit)\n---\n\n# {n} = {c['value']}\n\n{c['comment']}\n\n"
    f"Synapses: {syn or '-'}\n\nHub: [[sqtt-model]]\n\n## Evidence ({intro or 'no commit found'})\n\n{body}\n")
hub = "\n".join(f"- [[sqtt-{n}]] = {c['value']} ({c['scope']}, line {c['line']})" for n, c in sorted(consts.items(), key=lambda kv: kv[1]["line"]))
(out / "sqtt-model.md").write_text(f"---\ntype: hub\nmodel: sqtt-timing\nhead: {head}\ngenerated: rules2brain.py (do not edit)\n---\n\n"
                                  f"# SQTT timing model, {len(consts)} neurons at {head}\n\nGenerated from tinygrad test/mockgpu/amd/sqtt_enc.py.\n\n{hub}\n")
print(f"{len(consts)} neurons, {sum(len(v) for v in links.values()) // 2} synapses -> {out}")
