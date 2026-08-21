# -*- coding: utf-8 -*-
"""Parse Appendix D per-dataset tables (field-per-line format). v3"""
import re

SRC = r"D:\mscapital-kaggle\research\paper-reading-2026-08\realmlp.txt"
lines = open(SRC, encoding="utf-8").read().splitlines()

def strip_field(ln):
    s = ln.strip()
    if s.startswith("|"):
        s = s[1:].strip()
    if s.endswith("$"):
        s = s[:-1].strip()
    return s

table_starts = {}
for i, ln in enumerate(lines):
    m = re.match(r"Table (D\.\d+):", ln)
    if m:
        table_starts[m.group(1)] = i

VAL_RE = re.compile(r"([\d\.]+)±\\pm([\d\.]+)?")

def extract_table(name, n_cols):
    start = table_starts[name]
    end = len(lines)
    for other, pos in table_starts.items():
        if pos > start and pos < end:
            end = pos
    fields = []
    for ln in lines[start + 1:end]:
        s = strip_field(ln)
        if not s:
            continue
        if re.match(r"^(Table (D\.|C\.)|Figure |Appendix)", s):
            break
        fields.append(s)
    # locate 'Dataset' header
    try:
        k = fields.index("Dataset")
    except ValueError:
        print(f"  [WARN] {name}: no Dataset header, first fields: {fields[:5]}")
        return [], []
    header = fields[k + 1:k + 1 + n_cols]
    rest = fields[k + 1 + n_cols:]
    rows = []
    i = 0
    while i < len(rest):
        name_field = rest[i]
        if VAL_RE.search(name_field):  # a value where name expected -> alignment broken
            i += 1
            continue
        vals = {}
        ok = True
        for j in range(n_cols):
            if i + 1 + j >= len(rest):
                ok = False
                break
            m = VAL_RE.search(rest[i + 1 + j])
            if m:
                vals[j] = float(m.group(1))
        if ok:
            rows.append((name_field, vals))
        i += 1 + n_cols
    return header, rows

def win_summary(rows, rmlp_col, gbdt_cols):
    wins, losses = {}, {}
    for name, vals in rows:
        r = vals.get(rmlp_col)
        if r is None:
            continue
        for col, meth in gbdt_cols:
            v = vals.get(col)
            if v is None:
                continue
            wins.setdefault(meth, 0); losses.setdefault(meth, 0)
            if r < v: wins[meth] += 1
            elif r > v: losses[meth] += 1
    return wins, losses

TD_GBDT = [(5, "CatBoost-TD"), (6, "LGBM-TD"), (7, "XGB-TD")]
HPO_GBDT = [(6, "CatBoost-HPO"), (7, "LGBM-HPO"), (8, "XGB-HPO")]

print("=" * 74)
print("RealMLP-TD vs GBDT-TD 逐数据集胜负（均值误差，越小越好）")
print("=" * 74)
for t, label, ncols, rmlp_col, gcols in [
    ("D.1", "meta-train CLASS", 9, 0, TD_GBDT),
    ("D.3", "meta-train REG", 9, 0, TD_GBDT),
    ("D.5", "meta-test CLASS", 9, 0, TD_GBDT),
    ("D.7", "meta-test REG", 9, 0, TD_GBDT),
    ("D.9", "Grinsztajn CLASS", 9, 0, TD_GBDT),
    ("D.11", "Grinsztajn REG", 9, 0, TD_GBDT),
]:
    header, rows = extract_table(t, ncols)
    print(f"\n--- {label} (Table {t}, {len(rows)} datasets) ---")
    wins, losses = win_summary(rows, rmlp_col, gcols)
    for meth in [c for _, c in gcols]:
        w, l = wins.get(meth, 0), losses.get(meth, 0)
        wr = 100 * w / (w + l) if (w + l) else 0
        print(f"  vs {meth:12s} 赢 {w:3d} / 输 {l:3d}   胜率 {wr:5.1f}%")
    # worst losses vs CatBoost
    worst = []
    for name, vals in rows:
        r = vals.get(0)
        cb = vals.get(5)
        if r is not None and cb is not None and r > cb:
            worst.append((name, r, cb, (r - cb) / cb))
    for n, a, b, d in sorted(worst, key=lambda x: -x[3])[:5]:
        print(f"    输CB最惨: {n:34s} RMLP={a:.3f} CB={b:.3f} (相对差{d:+.0%})")

print()
print("=" * 74)
print("RealMLP-HPO vs GBDT-HPO（Grinsztajn 基准）")
print("=" * 74)
for t, label in [("D.10", "Grinsztajn CLASS"), ("D.12", "Grinsztajn REG")]:
    header, rows = extract_table(t, 10)
    print(f"\n--- {label} (Table {t}, {len(rows)} datasets) ---")
    wins, losses = win_summary(rows, 0, HPO_GBDT)
    for meth in [c for _, c in HPO_GBDT]:
        w, l = wins.get(meth, 0), losses.get(meth, 0)
        wr = 100 * w / (w + l) if (w + l) else 0
        print(f"  vs {meth:13s} 赢 {w:3d} / 输 {l:3d}   胜率 {wr:5.1f}%")
