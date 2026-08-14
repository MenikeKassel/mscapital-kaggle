# -*- coding: utf-8 -*-
"""去重: 三份相同 parquet → hardlink 到同一 inode (零破坏, 路径不变)"""
import os, hashlib

base = "D:/mscapital-kaggle/scripts"
src_dir = f"{base}/kaggle_0726ds"
dup_dirs = [f"{base}/kaggle_f0726ds", f"{base}/kaggle_f0726pq"]

for fname in ["f0726_train_f32.parquet", "f0726_test_f32.parquet"]:
    src = f"{src_dir}/{fname}"
    if not os.path.exists(src):
        print(f"SKIP (no source): {fname}")
        continue
    with open(src, "rb") as f:
        h_src = hashlib.md5(f.read()).hexdigest()
    for dd in dup_dirs:
        tgt = f"{dd}/{fname}"
        if not os.path.exists(tgt):
            print(f"SKIP (missing): {tgt}")
            continue
        with open(tgt, "rb") as f:
            h_tgt = hashlib.md5(f.read()).hexdigest()
        if h_src != h_tgt:
            print(f"DIFFERS (skip): {tgt}")
            continue
        os.remove(tgt)
        os.link(src, tgt)
        s_ino = os.stat(src).st_ino
        t_ino = os.stat(tgt).st_ino
        assert s_ino == t_ino, f"hardlink failed: {tgt}"
        print(f"LINKED: {tgt} (inode {s_ino})")

print("--- 结果 ---")
for fname in ["f0726_train_f32.parquet", "f0726_test_f32.parquet"]:
    inos = set()
    for dd in [src_dir] + dup_dirs:
        p = f"{dd}/{fname}"
        inos.add(os.stat(p).st_ino if os.path.exists(p) else None)
    print(f"{fname}: {len(inos - {None})} 个独立 inode, 大小 {os.path.getsize(f'{src_dir}/{fname}')/1e9:.2f} GB")
