# rewrite FLOOR in test/amd/test_sqtt_heldout.py to the current scores; print captures that dropped below the old floor
import re, multiprocessing as mp
from pathlib import Path
T = Path("test/amd/test_sqtt_heldout.py")
def one(p):
  from extra.sqtt.heldout.score import score
  h, n, _ = score(p); return p.stem, h, n
if __name__ == "__main__":
  from extra.sqtt.heldout.score import all_captures
  src = T.read_text()
  m = re.search(r"FLOOR = \{.*?\}\n", src, re.S)
  old = {k: (int(a), int(b)) for k, a, b in re.findall(r'"(\w+)": \((\d+), (\d+)\)', m.group(0))}
  with mp.Pool(14) as pool: res = pool.map(one, all_captures())
  for st, h, n in res:
    if st in old and (h < old[st][0] or n != old[st][1]): print(f"DROP {st}: {h}/{n} < floor {old[st]}")
  items = [f'"{st}": ({h}, {n})' for st, h, n in res]
  lines, cur = [], "FLOOR = {"
  for it in items:
    if len(cur) + len(it) + 2 > 120: lines.append(cur.rstrip()); cur = "         "
    cur += it + ", "
  lines.append(cur.rstrip(", ") + "}")
  T.write_text(src.replace(m.group(0), "\n".join(lines) + "\n"))
  print(f"total {sum(h for _, h, _ in res)}/{sum(n for *_, n in res)}")
