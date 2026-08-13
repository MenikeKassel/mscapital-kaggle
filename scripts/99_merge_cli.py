# -*- coding: utf-8 -*-
"""Resolve cli.py merge conflicts: keep both HEAD and origin blocks."""
import re, sys

path = r"D:\mscapital-kaggle\src\mscapital\cli.py"
text = open(path, encoding="utf-8").read()

pattern = re.compile(
    r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> origin/main\n",
    re.S,
)
matches = pattern.findall(text)
print("conflict blocks:", len(matches))

def merge(m):
    head, origin = m.group(1), m.group(2)
    # both kept, HEAD first
    return head + origin

new = pattern.sub(merge, text)
assert "<<<<<<<" not in new and "=======" not in new and ">>>>>>>" not in new, "unresolved markers remain"
open(path, "w", encoding="utf-8", newline="\n").write(new)
print("merged, markers left:", new.count("<<<<<<<"))
