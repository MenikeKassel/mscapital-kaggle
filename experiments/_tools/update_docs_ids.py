# -*- coding: utf-8 -*-
"""一次性: 更新 docs 中旧实验 ID → canonical 新 ID (词边界精确替换).
历史报告正文不动 (任务书 §12); 只更新当前索引/汇总类文档.
"""
import os, re, glob

REPO = r"D:\mscapital-kaggle"

# 长 token 优先, 防子串误伤
MAP = [
    ("P0.5-B", "P0-04"), ("P0.5-C", "P0-05"), ("P0.5-D", "P0-06"),
    ("P1-01a", "P1-01"), ("P1-01b", "P1-02"), ("P1-01c", "P1-03"), ("P1-01e", "P1-04"),
    ("P5-02I", "P5-02"), ("P5-A", "P5-03"), ("P5-B", "P5-04"), ("P5-C", "P5-05"),
    ("P5-D", "P5-06"), ("P5-E", "P5-07"),
    ("P4-06A", "P4-08"), ("P4-08A", "P4-10"), ("P4-H1H2", "P4-15"),
    ("P4-LB142", "P4-16"), ("P4-MH", "P4-17"),
    ("P7-AMP", "P7-01"), ("SUB-v4", "S-04"), ("SUB-v5", "S-05"),
    ("SUB-v7", "S-07"), ("SUB-v8", "S-08"),
    ("P2-cal", "P2-01"), ("P6R-00", "P6R-00"), ("P6R-01", "P6R-01"),
    ("M01-A", "M-01"), ("M02-T", "M-03"), ("E01", "E-01"), ("E02", "E-02"), ("E03", "E-03"),
    ("P4-02c", "P4-04"), ("P4-02b", "P4-03"), ("P4-02a", "P4-02"),
]
# C/M 系列 (需词边界: C1 不能碰 C1-FE, C-01 已有; M02 不能碰 M02-T)
C_MAP = [("C1", "C-01"), ("C2", "C-02"), ("C3", "C-03"), ("C4", "C-04"),
         ("M02", "M-02"), ("M03", "M-04"), ("M04", "M-05"), ("M05", "M-06"), ("M06", "M-07")]

FILES = [
    "docs/experiment-index.md", "docs/failed-experiments.md",
    "docs/research-findings.md", "docs/method-map.md", "docs/reorganization-report.md",
    "docs/experiment-inventory.md", "docs/README.md", "README.md",
    "submissions/README.md", "docs/gpt-review-experiment-id-2026-08-15.md",
]

def repl(text):
    for old, new in MAP:
        text = re.sub(rf"(?<![A-Za-z0-9-]){re.escape(old)}(?![A-Za-z0-9-])", new, text)
    for old, new in C_MAP:
        # 词边界但允许 "-" 后缀 (C1-FE 不匹配); 禁止 C-01 已存在被二次替换
        text = re.sub(rf"(?<![A-Za-z0-9-]){re.escape(old)}(?![A-Za-z0-9-])", new, text)
    return text

total = 0
for f in FILES:
    p = os.path.join(REPO, f)
    if not os.path.exists(p):
        continue
    t = open(p, encoding="utf-8").read()
    t2 = repl(t)
    if t2 != t:
        open(p, "w", encoding="utf-8").write(t2)
        n = sum(1 for _ in re.finditer(r"\|", t2)) - sum(1 for _ in re.finditer(r"\|", t))
        total += 1
        print("updated:", f)
print("files updated:", total)
