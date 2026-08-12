# -*- coding: utf-8 -*-
"""本地全量跑 P1-2 (v3 修复版, 保险)"""
import sys
sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
import kaggle_p12_tcn as k

k.DATA = r"D:\mscapital-forecasting\data\raw"
k.OUT = r"D:\mscapital-forecasting\data\processed\p12_out"
k.main()
