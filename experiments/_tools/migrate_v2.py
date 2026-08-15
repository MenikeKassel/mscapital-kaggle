# -*- coding: utf-8 -*-
"""Experiment ID v1.0 迁移增量表 (2026-08-15).

每个条目: (old_id, new_id, id_status, title, name_slug, parent, aliases, tags, decision, status)
- id_status: canonical(迁移/合规) | legacy(冻结保留)
- parent/successor 用新 ID 表达研究谱系 (任务书 §24)
- aliases: 旧 ID 全部保留 (| 分隔), 一步解析 (任务书 §10)
- decision ∈ {GREEN, YELLOW, RED, NA}; status ∈ {planned, running, completed, aborted, deprecated, superseded}
"""
# (old_id, new_id, id_status, title, name, parent, aliases, tags, decision, status)
MIGRATE = [
# ===== Legacy 冻结区 (baseline, 任务书 §8/§9) =====
("B0", "B0", "legacy", "全0/常数基线 sanity check", "constant-baseline", "", "", "sanity", "NA", "superseded"),
("B1", "B1", "legacy", "LightGBM 官方参数 90特征复刻", "lgb-official", "", "", "baseline|lightgbm", "GREEN", "completed"),
("A1", "A1", "legacy", "CV 切分敏感性", "cv-sensitivity", "B1", "", "validation", "GREEN", "completed"),
("A2", "A2", "legacy", "特征数量 17 vs 90", "feature-count", "B1", "", "features", "GREEN", "completed"),
("B1-LGO", "B1-LGO", "legacy", "特征组消融 LGO", "feature-ablation", "A2", "", "features", "GREEN", "completed"),
("B2", "B2", "legacy", "特征精简 90/84/77", "feature-prune", "B1-LGO", "", "features", "GREEN", "completed"),
("C1-FE", "C1-FE", "legacy", "增强窗口特征 90→109", "enhanced-windows", "B2", "", "features", "RED", "completed"),
("D1", "D1", "legacy", "超参数单变量", "hp-sweep", "B1", "", "lightgbm", "YELLOW", "completed"),
("E1-TW", "E1-TW", "legacy", "时间衰减加权", "time-decay", "B1", "", "validation", "RED", "completed"),
("F1", "F1", "legacy", "轻量 MLP (90特征)", "mlp-tabular", "B1", "", "mlp", "YELLOW", "completed"),
("F2", "F2", "legacy", "MLP 30ep×3seed 集成", "mlp-ensemble", "F1", "", "mlp", "GREEN", "completed"),
("G1", "G1", "legacy", "LGBM+MLP 融合", "lgb-mlp-blend", "F1", "", "blend", "GREEN", "completed"),
("G2", "G2", "legacy", "三模型融合 +XGBoost", "3model-blend", "G1", "", "blend", "YELLOW", "completed"),
("G3", "G3", "legacy", "三模型融合v2 (MLP-ens)", "3model-blend-v2", "G2", "", "blend", "GREEN", "completed"),
("H1", "H1", "legacy", "五模型融合 (CatBoost 惊喜)", "5model-blend", "G3", "", "blend", "RED", "completed"),

# ===== P0 协议 =====
("P0-01", "P0-01", "canonical", "Temporal Matrix 时序矩阵", "temporal-matrix", "B1", "", "validation", "GREEN", "completed"),
("P0-02", "P0-02", "canonical", "Adversarial Validation 对抗验证", "adversarial-validation", "P0-01", "", "validation|drift", "GREEN", "completed"),
("P0-03", "P0-03", "canonical", "权重重估 temporal-mean", "weight-reopt", "P0-01", "P0-3", "validation|blend", "RED", "completed"),
("P0.5-B", "P0-04", "canonical", "MLP Fairness Check (last-epoch bug)", "mlp-fairness", "P0-01", "P0.5-B", "protocol|mlp", "GREEN", "superseded"),
("P0.5-C", "P0-05", "canonical", "Drift Intervention (R2 归一化)", "drift-intervention", "P0-02", "P0.5-C", "drift|features", "GREEN", "completed"),
("P0.5-D", "P0-06", "canonical", "R2 × 全融合", "r2-blend", "P0-05", "P0.5-D", "blend", "GREEN", "completed"),

# ===== S 提交系列 (任务书 §28) =====
("SUB-v4", "S-04", "canonical", "v4 提交 (R2+temporal)", "submission-v4", "P0-06", "SUB-v4", "submission", "GREEN", "completed"),
("SUB-v5", "S-05", "canonical", "v5 提交 (R2+22微观)", "submission-v5", "P1-03", "SUB-v5", "submission", "GREEN", "completed"),
("SUB-v7", "S-07", "canonical", "v7 提交 (RealMLP 复刻融合)", "submission-v7", "P1-03", "SUB-v7", "submission", "GREEN", "completed"),
("SUB-v8", "S-08", "canonical", "v8/v8b 提交 (lb142 融合)", "submission-v8b", "S-07", "SUB-v8", "submission", "GREEN", "completed"),
# 补录提交事件 (registry 原缺)
("S-01", "S-01", "canonical", "v1 提交 (三模型融合)", "submission-v1", "G3", "", "submission", "GREEN", "completed"),
("S-02", "S-02", "canonical", "v2 提交 (五模型)", "submission-v2", "H1", "", "submission", "YELLOW", "completed"),
("S-03", "S-03", "canonical", "v3 提交 (temporal 权重)", "submission-v3", "P0-03", "", "submission", "YELLOW", "completed"),
("S-06", "S-06", "canonical", "v6 提交 (TCN 融合, 灾难)", "submission-v6", "P1-05", "", "submission|negative", "RED", "completed"),

# ===== P1 表示 =====
("P1-01a", "P1-01", "canonical", "微观结构特征构建 (22 primitive)", "micro-features-build", "P0-06", "P1-1a|P1-01a", "features|order-flow", "NA", "completed"),
("P1-01b", "P1-02", "canonical", "双轴筛选 (alpha+drift)", "micro-features-screen", "P1-01", "P1-1b|P1-01b", "features|validation", "YELLOW", "completed"),
("P1-01c", "P1-03", "canonical", "全融合验证 (R2+micro)", "micro-features-blend", "P1-02", "P1-1c|P1-01c", "blend", "GREEN", "completed"),
("P1-01e", "P1-04", "canonical", "特征相对化第二轮", "micro-features-rel2", "P1-03", "P1-1e|P1-01e", "features|negative", "RED", "completed"),
("P1-02", "P1-05", "canonical", "TCN 双塔序列模型", "tcn-dual-tower", "P1-03", "", "sequence|negative", "RED", "completed"),

# ===== P2 校准 =====
("P2", "P2-01", "canonical", "RealMLP PSEUDO 定标 + 尺度门禁", "realmlp-pseudo-calib", "S-07", "P2", "calibration", "GREEN", "completed"),

# ===== C 系列 Clean Baseline =====
("C1", "C-01", "canonical", "C1 Clean RealMLP-v2a", "clean-realmlp", "P2-01", "", "protocol|realmlp", "GREEN", "completed"),
("C2", "C-02", "canonical", "C2 RealMLP Ablation (30-epoch)", "realmlp-ablation", "C-01", "", "protocol|realmlp", "GREEN", "completed"),
("C3", "C-03", "canonical", "C3 Clean Table v2", "clean-table", "C-01", "", "protocol|tabular", "GREEN", "completed"),
("C4", "C-04", "canonical", "C4 Clean Baseline v2 冻结", "clean-baseline-freeze", "C-02", "", "protocol|baseline", "GREEN", "completed"),

# ===== P3 下一代方法 =====
("P3-01", "P3-01", "canonical", "SAE 自编码器表示", "sae", "C-04", "", "unsupervised|negative", "RED", "completed"),
("P3-02", "P3-02", "canonical", "状态条件化 (E01 延伸)", "state-conditioned", "C-04", "", "conditional|negative", "RED", "completed"),
("P3-03", "P3-03", "canonical", "TinyLOBERT 掩码预训练", "tiny-lobert-mask", "C-04", "", "unsupervised|negative", "RED", "completed"),
("P3-04", "P3-04", "canonical", "2.5D 网格表示", "grid-2d5", "C-04", "", "representation|negative", "RED", "completed"),
("P3-05", "P3-05", "canonical", "NHP 强度诊断", "nhp-intensity", "C-04", "", "point-process|negative", "RED", "completed"),

# ===== P4 隐藏信息 (P4-02 拆分 + P4-08B 补录) =====
("P4-01a", "P4-01", "canonical", "600s market 长上下文取证", "market-600s-forensics", "C-04", "P4-01a", "market|hidden-info", "GREEN", "completed"),
("P4-02a", "P4-02", "canonical", "LB142 v10 factors 逆向审计", "factor-audit", "P4-01", "P4-02A", "hidden-info", "YELLOW", "completed"),
("P4-02b", "P4-03", "canonical", "LB142 market-form primitives 重建", "market-forms", "P4-02", "P4-02B", "hidden-info|market", "YELLOW", "completed"),
("P4-02c", "P4-04", "canonical", "OFI protocol 冻结验证", "ofi-protocol", "P4-03", "P4-02C", "order-flow|hidden-info", "YELLOW", "completed"),
("P4-03", "P4-05", "canonical", "Target 逆向取证", "target-forensics", "P4-01", "", "hidden-info|target", "YELLOW", "completed"),
("P4-04", "P4-06", "canonical", "0.5 价簇取证 (ask 全空假象)", "price-cluster", "P4-01", "", "hidden-info|market", "GREEN", "completed"),
("P4-05", "P4-07", "canonical", "月度漂移可预测性", "monthly-drift", "P4-01", "", "drift|negative", "RED", "completed"),
("P4-06A", "P4-08", "canonical", "Long residual (短/长/双窗)", "long-residual", "P4-01", "P4-06A", "residual|negative", "RED", "completed"),
("P4-07", "P4-09", "canonical", "Halfgroup 分组检验", "halfgroup", "P4-01", "", "validation|market", "YELLOW", "completed"),
("P4-08A", "P4-10", "canonical", "Loss ablation (MSE vs cosine)", "loss-ablation", "P2-01", "P4-08A", "cosine|objective", "RED", "superseded"),
("P4-08B", "P4-11", "canonical", "cosine 生产化 (lambda_cos 审计)", "cosine-prod", "P4-10", "P4-08B", "cosine|production", "YELLOW", "completed"),
("P4-08C", "P4-12", "canonical", "cosine blend → submission 候选", "cosine-blend", "P4-11", "P4-08C", "cosine|submission", "YELLOW", "completed"),
("P4-08D", "P4-13", "canonical", "simple-MLP cosine 全量生产", "cosine-simple-prod", "P4-11", "P4-08D", "cosine|production", "YELLOW", "completed"),
("P4-08E", "P4-14", "canonical", "cosine 案件收口 (v7_like 复刻)", "cosine-v7like", "P4-10", "P4-08E", "cosine|objective", "YELLOW", "completed"),
("P4-H1H2", "P4-15", "canonical", "资产/时间身份取证 (H1/H2)", "identity-forensics", "P4-01", "", "hidden-info|negative", "RED", "completed"),
("P4-LB142", "P4-16", "canonical", "LB142 分歧取证", "lb142-divergence", "P4-01", "", "hidden-info|lb142", "YELLOW", "completed"),
("P4-MH", "P4-17", "canonical", "Market history 特征", "market-history", "P4-01", "", "market|features", "YELLOW", "completed"),

# ===== M 系列 (M02-T 补录) =====
("M01-A", "M-01", "canonical", "Event Flow 残差表示", "event-flow", "C-04", "M01-A", "residual|negative", "RED", "completed"),
("M02", "M-02", "canonical", "LOB 几何特征", "lob-geometry", "M-01", "", "geometry|negative", "RED", "completed"),
("M02-T", "M-03", "canonical", "LOB 几何 temporal 变体", "lob-geometry-temporal", "M-02", "M02-T", "geometry|negative", "RED", "completed"),
("M03", "M-04", "canonical", "Path Signature (depth-2)", "path-signature", "M-01", "", "signature|negative", "RED", "completed"),
("M04", "M-05", "canonical", "Optiver Interaction 特征族", "optiver-interactions", "M-01", "", "interaction|negative", "RED", "completed"),
("M05", "M-06", "canonical", "Market-State KNN", "market-state-knn", "M-01", "", "retrieval|negative", "RED", "completed"),
("M06", "M-07", "canonical", "Cross-sectional 动态审计", "cross-sectional", "M-01", "", "cross-section|negative", "RED", "completed"),

# ===== E 系列 =====
("E01", "E-01", "canonical", "ReVol-lite 状态条件化", "revol-lite", "C-04", "", "conditional", "YELLOW", "completed"),
("E02", "E-02", "canonical", "Reconditionor-lite", "reconditionor-lite", "E-01", "", "conditional|negative", "RED", "completed"),
("E03", "E-03", "canonical", "稳定性审计", "stability-audit", "E-01", "", "validation", "GREEN", "completed"),

# ===== P5 市场 =====
("P5-01", "P5-01", "canonical", "Market-only 序列审判", "market-sequence", "P4-01", "", "market|sequence", "YELLOW", "completed"),
("P5-02I", "P5-02", "canonical", "信息审计 (五条实锤)", "info-audit", "P5-01", "P5-02I|P5-02i", "market|diagnostics", "GREEN", "completed"),
("P5-A", "P5-03", "canonical", "MAG-Gate 幅度门控", "mag-gate", "P5-02", "P5-A|MAG-Gate", "amplitude|negative", "RED", "completed"),
("P5-B", "P5-04", "canonical", "SCFI 条件创新 (LGB)", "scfi", "P5-02", "P5-B|SCFI", "conditional|order-flow", "GREEN", "completed"),
("P5-C", "P5-05", "canonical", "RICS 跨通道几何", "rics", "P5-02", "P5-C|RICS", "geometry|negative", "RED", "completed"),
("P5-D", "P5-06", "canonical", "SCFI 生产验证 (LGB 3seed+Cat)", "scfi-prod-verify", "P5-04", "P5-D", "conditional|production", "GREEN", "completed"),
("P5-E", "P5-07", "canonical", "RealMLP 生产 learner spot-check", "realmlp-scfi-spotcheck", "P5-06", "P5-E", "conditional|realmlp", "GREEN", "completed"),

# ===== P6/P6R 生产 =====
("P6", "P6-01", "canonical", "RealMLP-C 生产推理 + 提交候选", "production-inference", "P5-07", "", "production", "YELLOW", "completed"),
("P6R-00", "P6R-00", "canonical", "Retrieval 残差检索", "retrieval-residual", "P4-08", "", "retrieval|negative", "RED", "completed"),
("P6R-01", "P6R-01", "canonical", "Local vs Global Ridge 终裁", "local-vs-global-ridge", "P6R-00", "", "retrieval", "NA", "planned"),

# ===== P7 幅度 =====
("P7-AMP", "P7-01", "canonical", "GPT1 预注册幅度门控快速版", "amplitude-gate", "P5-03", "P7-AMP", "amplitude|negative", "RED", "completed"),
]
