# -*- coding: utf-8 -*-
"""核验 GPT 评审引用的 3 篇新论文"""
import re, subprocess, time

IDS = ["1907.06230", "2505.02139", "2403.09267"]
for aid in IDS:
    r = subprocess.run(["curl", "-s", "--max-time", "25", f"https://arxiv.org/abs/{aid}"],
                       capture_output=True, text=True)
    m = re.search(r"<title>\[([^\]]+)\] ([^<]+)</title>", r.stdout)
    if m:
        print(f"[OK] {aid} ({m.group(1)[:4]}) {m.group(2).strip()[:100]}")
    else:
        print(f"[FAIL] {aid}")
    time.sleep(3)
