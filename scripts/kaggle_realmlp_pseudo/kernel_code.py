# -*- coding: utf-8 -*-
"""
RealMLP PSEUDO 训练 (云端): 计算 v7 融合的本地分
exec dataset 里的 realmlp_pseudo.py (PSEUDO 切分 + 训练 + valid pred 输出)
"""
import os, glob

print("=== input listing ===", flush=True)
print(glob.glob("/kaggle/input/**/*", recursive=True)[:20], flush=True)

code = open("/kaggle/input/msc-f0726-data/realmlp_pseudo.py", encoding="utf-8").read()
exec(compile(code, "realmlp_pseudo.py", "exec"))
print("DONE")
