# -*- coding: utf-8 -*-
"""核验通用模型经典论文的 arXiv ID (网页版, 3s间隔)"""
import json, os, re, subprocess, time

IDS = ["1803.01271", "1706.03762", "1603.02754", "1703.09831", "1908.07442"]
out = {}
for aid in IDS:
    r = subprocess.run(["curl", "-s", "--max-time", "25", f"https://arxiv.org/abs/{aid}"],
                       capture_output=True, text=True)
    html = r.stdout
    m = re.search(r"<title>\[([^\]]+)\] ([^<]+)</title>", html)
    if m:
        out[aid] = {"id": aid, "title": m.group(2).strip(), "year": m.group(1)[:4]}
        print(f"[OK] {aid} ({m.group(1)[:4]}) {m.group(2).strip()[:90]}", flush=True)
    else:
        out[aid] = {"id": aid, "title": "VERIFY_FAILED", "year": "?"}
        print(f"[FAIL] {aid}", flush=True)
    time.sleep(3)

with open(r"D:\mscapital-kaggle\research\arxiv_classics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("done")
