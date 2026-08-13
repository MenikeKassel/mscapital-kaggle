# -*- coding: utf-8 -*-
"""自动合并 cherry-pick 冲突: 每个冲突块保留 HEAD + 9dacdc9 两侧内容"""
import re, sys

for path in [r"D:\mscapital-revol-integration\src\mscapital\cli.py",
             r"D:\mscapital-revol-integration\src\mscapital\features\__init__.py"]:
    src = open(path, encoding="utf-8").read()
    n = 0

    def repl(m):
        global n
        n += 1
        head, theirs = m.group(1), m.group(2)
        return head.rstrip() + "\n\n" + theirs.strip() + "\n"

    out, cnt = re.subn(r"<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> 9dacdc9[^\n]*\n",
                       repl, src, flags=re.S)
    open(path, "w", encoding="utf-8").write(out)
    print(f"{path}: resolved {cnt} conflicts (was {n})")

# 语法检查
import py_compile
for path in [r"D:\mscapital-revol-integration\src\mscapital\cli.py",
             r"D:\mscapital-revol-integration\src\mscapital\features\__init__.py"]:
    py_compile.compile(path, doraise=True)
    print(f"compile OK: {path}")
