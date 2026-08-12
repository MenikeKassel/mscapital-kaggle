# -*- coding: utf-8 -*-
"""文献候选核验: 用 arxiv id_list API 批量核验候选论文, 输出 JSON 清单"""
import json, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET

# 候选 arXiv ID (凭领域知识列出, 待API核验; 核验失败自动剔除)
CANDIDATES = [
    # 主题1: OFI / order flow
    "1808.03668",  # DeepLOB
    "1803.06917",  # Sirignano & Cont universal features
    "1704.02645",  # ??
    "1512.06665",  # ??
    # 主题2: LOB deep learning
    "1905.04388",  # ??
    "2006.14425",  # ??
    "2201.12293",  # ??
    "2107.03637",  # ??
    # 主题3: distribution shift / temporal
    "2109.09114",  # ??
    "2205.13547",  # ??
    # 主题4: microstructure
    "1405.7350",   # ??
    "1202.2513",   # ??
    # 主题5: tabular/GBDT
    "2207.08815",  # ??
    "2110.14051",  # ??
]

NS = {"a": "http://www.w3.org/2005/Atom"}
out = []
for i in range(0, len(CANDIDATES), 5):
    batch = CANDIDATES[i:i+5]
    url = "http://export.arxiv.org/api/query?id_list=" + ",".join(batch) + "&max_results=10"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            xml = r.read().decode("utf-8", "ignore")
        root = ET.fromstring(xml)
        for e in root.findall("a:entry", NS):
            aid = e.find("a:id", NS).text.split("/abs/")[-1]
            title = " ".join(e.find("a:title", NS).text.split())
            pub = e.find("a:published", NS).text[:10]
            out.append({"id": aid, "title": title, "year": pub[:4]})
            print(f"[OK] {aid} ({pub}) {title[:90]}", flush=True)
    except Exception as ex:
        print(f"[ERR] batch {batch}: {ex}", flush=True)
    time.sleep(4)

with open(r"D:\mscapital-kaggle\research\arxiv_verify.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\nverified {len(out)}/{len(CANDIDATES)}")
