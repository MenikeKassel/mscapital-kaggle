# -*- coding: utf-8 -*-
"""下载并解析排行榜, 定位自己的分数排名"""
import csv, subprocess, glob, os

os.chdir(r"D:\mscapital-kaggle")
r = subprocess.run(["kaggle", "competitions", "leaderboard", "-c",
                    "ms-capital-real-financial-market-forecasting", "--download"],
                   capture_output=True, text=True, env=os.environ)
print(r.stdout[-200:])
# 解压下载的 zip
import zipfile
zips = glob.glob("ms-capital-real-financial-market-forecasting*.zip")
print("zip:", zips)
if zips:
    z = max(zips, key=os.path.getmtime)
    with zipfile.ZipFile(z) as zf:
        zf.extractall("_lb_tmp")
        csvs = glob.glob("_lb_tmp/*.csv")
print("csv files:", csvs)
if not csvs:
    print("STDERR:", r.stderr[-500:])
    raise SystemExit(1)
f = max(csvs, key=os.path.getmtime)
rows = list(csv.DictReader(open(f, encoding="utf-8")))
scored = [(i + 1, r["TeamName"], float(r["Score"])) for i, r in enumerate(rows)
          if r.get("Score") not in (None, "") and r["Score"] != ""]
scored.sort(key=lambda x: -x[2])
print(f"total teams: {len(scored)}")
for rank, name, s in scored:
    if abs(s - 0.123) < 0.0006:
        print(f">>> 0.123 排名: #{rank} {name} ({s})")
# 打印 0.123 前后 5 名
for i, (rank, name, s) in enumerate(scored):
    if abs(s - 0.123) < 0.0006:
        for j in range(max(0, i - 3), min(len(scored), i + 4)):
            print(f"  #{scored[j][0]} {scored[j][1]} {scored[j][2]}")
        break
