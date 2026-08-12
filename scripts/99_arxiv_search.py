# -*- coding: utf-8 -*-
"""arXiv 网页搜索 v2: curl抓HTML到项目tmp, python正则解析结果块"""
import json, os, re, subprocess, time

TMP = r"D:\mscapital-kaggle\research\tmp"
os.makedirs(TMP, exist_ok=True)

QUERIES = {
    "t1_ofi": "order flow imbalance price impact limit order book",
    "t2_lob_deep": "limit order book deep learning prediction",
    "t3_dist_shift": "distribution shift financial time series forecasting generalization",
    "t4_micro_feat": "high frequency trading microstructure features machine learning",
    "t5_gbdt_seq": "gradient boosting machine learning financial market prediction",
    "t6_ensemble": "ensemble learning deep learning financial time series",
}

def fetch(url, path):
    r = subprocess.run(["curl", "-s", "--max-time", "30", url, "-o", path], capture_output=True, text=True)
    return r.returncode

out = {}
for key, q in QUERIES.items():
    path = os.path.join(TMP, f"{key}.html")
    url = "https://arxiv.org/search/?searchtype=all&query=" + q.replace(" ", "+") + "&start=0"
    rc = fetch(url, path)
    if rc != 0 or not os.path.exists(path):
        print(f"=== {key}: FETCH FAILED ===", flush=True)
        continue
    html = open(path, encoding="utf-8", errors="ignore").read()
    items = []
    # 按 li.arxiv-result 分块
    blocks = re.split(r'<li class="arxiv-result">', html)[1:]
    for b in blocks[:10]:
        m = re.search(r'href="https://arxiv\.org/abs/(\d{4}\.\d{4,5})"', b)
        if not m:
            continue
        aid = m.group(1)
        tm = re.search(r'<p class="title[^"]*">\s*<a[^>]*>(.*?)</a>', b, re.S)
        title = re.sub(r"\s+", " ", tm.group(1)).strip() if tm else "?"
        ym = re.search(r"\[Submitted on (.*?)\]", b)
        year = ym.group(1)[-4:] if ym else "?"
        items.append({"id": aid, "title": title, "year": year})
    out[key] = items
    print(f"=== {key}: {len(items)} results ===", flush=True)
    for it in items:
        print(f"  {it['id']} ({it['year']}) {it['title'][:85]}", flush=True)
    time.sleep(5)

with open(r"D:\mscapital-kaggle\research\arxiv_search.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved arxiv_search.json")
