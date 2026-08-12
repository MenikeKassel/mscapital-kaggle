# -*- coding: utf-8 -*-
"""核验 GPT1 引用的新论文: 2407.04491 (Better by Default = RealMLP 来源)"""
import subprocess, re

r = subprocess.run(["curl", "-s", "--max-time", "30", "https://arxiv.org/abs/2407.04491"],
                   capture_output=True, text=True)
html = r.stdout
m = re.search(r"<title>(.*?)</title>", html, re.S)
print("title:", m.group(1).strip() if m else "NOT FOUND")
# 摘要前 200 字
m2 = re.search(r'<blockquote class="abstract[^"]*">\s*<p>(.*?)</p>', html, re.S)
if m2:
    abstract = re.sub(r"<[^>]+>", "", m2.group(1)).strip()
    print("abstract:", abstract[:300])
else:
    print("abstract: not found")
