# -*- coding: utf-8 -*-
"""解析已抓取的 arxiv 搜索页 HTML -> 结构化清单 (id/title/year)"""
import json, os, re

TMP = r"D:\mscapital-kaggle\research\tmp"
OUT = r"D:\mscapital-kaggle\research\arxiv_search.json"

QUERIES = {
    "t1_ofi": "order flow imbalance price impact limit order book",
    "t2_lob_deep": "limit order book deep learning prediction",
    "t3_dist_shift": "distribution shift financial time series forecasting generalization",
    "t4_micro_feat": "high frequency trading microstructure features machine learning",
    "t5_gbdt_seq": "gradient boosting machine learning financial market prediction",
    "t6_ensemble": "ensemble learning deep learning financial time series",
}

out = {}
for key in QUERIES:
    path = os.path.join(TMP, f"{key}.html")
    if not os.path.exists(path):
        continue
    html = open(path, encoding="utf-8", errors="ignore").read()
    blocks = re.split(r'<li class="arxiv-result">', html)[1:]
    items = []
    for b in blocks[:10]:
        m = re.search(r'href="https://arxiv\.org/abs/(\d{4}\.\d{4,5})"', b)
        if not m:
            continue
        aid = m.group(1)
        # 标题: 在 "list-title" 链接之后找 <p class="title ..."> 块
        tm = re.search(r'<p class="title[^"]*">\s*(.*?)\s*</p>', b, re.S)
        title = "?"
        if tm:
            title = re.sub(r"<[^>]+>", "", tm.group(1))
            title = re.sub(r"\s+", " ", title).strip()
        ym = re.search(r"\[Submitted on ([^]]+)\]", b)
        year = ym.group(1)[-4:] if ym else "?"
        items.append({"id": aid, "title": title, "year": year})
    out[key] = items
    print(f"=== {key}: {len(items)} ===", flush=True)
    for it in items:
        print(f"  {it['id']} ({it['year']}) {it['title'][:85]}", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\nsaved {OUT}")
