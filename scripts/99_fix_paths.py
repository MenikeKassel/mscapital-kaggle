# -*- coding: utf-8 -*-
"""批量替换脚本中的旧路径 (E:→D:, F:→D:)"""
import glob, os

PAIRS = [
    (r"D:\mscapital-kaggle", r"D:\mscapital-kaggle"),
    (r"D:/mscapital-kaggle", r"D:/mscapital-kaggle"),
    (r"D:\mscapital-forecasting", r"D:\mscapital-forecasting"),
    (r"D:/mscapital-forecasting", r"D:/mscapital-forecasting"),
]

files = (glob.glob(r"D:\mscapital-kaggle\scripts\*.py")
         + glob.glob(r"D:\mscapital-kaggle\docs\*.md")
         + [r"D:\mscapital-kaggle\README.md", r"D:\mscapital-kaggle\RESULTS.md"])

changed = 0
for fp in files:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    orig = s
    for a, b in PAIRS:
        s = s.replace(a, b)
    if s != orig:
        with open(fp, "w", encoding="utf-8", newline="\n") as f:
            f.write(s)
        changed += 1
        print(f"updated: {os.path.basename(fp)}")

# 验证残留
leftover = []
for fp in files:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        if "E:\\aiworkspace" in f.read() or "F:\\mscapital" in f.read():
            leftover.append(fp)
print(f"\n{changed} files updated; leftover: {len(leftover)}")
for l in leftover:
    print(f"  REMAINS: {l}")
