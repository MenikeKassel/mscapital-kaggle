# -*- coding: utf-8 -*-
"""生成 docs/experiment-lineage.md (研究谱系树, 任务书 §43)."""
import csv, os, sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
REG = os.path.join(BASE, "experiments", "registry.csv")
OUT = os.path.join(BASE, "docs", "experiment-lineage.md")

rows = list(csv.DictReader(open(REG, encoding="utf-8-sig")))
by_id = {r["experiment_id"]: r for r in rows}
children = defaultdict(list)
for r in rows:
    if r["parent"]:
        children[r["parent"]].append(r["experiment_id"])

def fmt(r):
    d = r["decision"]
    mark = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "NA": "⚪"}.get(d, "⚪")
    s = r["status"]
    note = f" [{d}]" if d not in ("NA",) else ""
    if s in ("planned",):
        note += " (planned)"
    if s in ("superseded", "deprecated"):
        note += f" ({s})"
    return f"`{r['experiment_id']}` {r['title']}{note}"

lines = ["# MSCapital 实验研究谱系 (Research Lineage)", "",
         "> 自动生成: 由 registry parent/successor 构建 (任务书 §43) | 重跑: `python experiments/_tools/build_lineage.py`", ""]

# 根 (无 parent)
roots = [r["experiment_id"] for r in rows if not r["parent"]]

def walk(cid, depth):
    r = by_id[cid]
    lines.append("  " * depth + ("├── " if depth else "") + fmt(r))
    for c in sorted(children.get(cid, [])):
        walk(c, depth + 1)

for ph in ["baseline", "P0_protocol", "P1_representation", "P2_calibration", "C_clean",
           "P3_nextgen", "P4_hidden", "M_representation", "E_conditional",
           "P5_market", "P6_production", "P7_amplitude", "S_submissions"]:
    ph_rows = [r for r in rows if r["phase"] == ph]
    if not ph_rows:
        continue
    lines.append(f"## {ph}\n")
    ph_ids = {r["experiment_id"] for r in ph_rows}
    for r in ph_rows:
        if not r["parent"] or r["parent"] not in ph_ids:
            walk(r["experiment_id"], 0)
    lines.append("")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("lineage written:", len(lines), "lines")
