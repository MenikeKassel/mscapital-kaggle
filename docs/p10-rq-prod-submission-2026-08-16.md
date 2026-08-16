# P10 RQ 生产提交记录 (2026-08-16) — 如实记录

## 一句话结论

**PSEUDO 本地 0.143758（历史最高），LB 0.116（远低于 v8b 锚点 0.142）——生产外推失败。**
PSEUDO 高分不可信：定标模型与提交模型不是同一个，γ 跨模型套用，协议错位导致乐观偏差。

---

## 1. 实验目的

把 P9 验证过的两条正信号（SCFI Z 特征 + neutralization）升级到生产管线：
1. RQ 全量训练（152+73Z，train 0-70 全部 1,257,637 样本）→ test 预测
2. neutralization 生产应用（PSEUDO 定标 γ → test 应用）→ submission

## 2. 执行步骤

| # | 步骤 | 脚本/命令 | 结果 |
|---|---|---|---|
| 1 | RQ 全量训练 30 epochs (152+73Z, 0-70) | `scripts/p10_rq_prod_scfi.py` | ✅ train cos 0.2445, ~70min, test pred 647,896 落盘 |
| 2 | neutralization 生产应用 | `scripts/p10_neutralize_prod.py` | ✅ γ=1.0, submission 生成 |
| 3 | 提交 #1 (ref 55543480) | Python kaggle API | ❌ FAILED: 列名 `target`（应为 `prediction`） |
| 4 | 提交 #2 (ref 55543658) | 修正列名重提 | ✅ COMPLETE, **LB 0.116** |

### 过程中修复的 bug
- `p10_neutralize_prod.py` 模型路径指向 `output/p10_rq_prod_scfi/arm_C/best.pt`（不存在）→ 改为 `output/p9_scfi_rq/arm_C/best.pt`
- cat_dims 误用 eval 集计算（类别数与 train 不同 → embedding 维度不匹配 3376 vs 3392）→ 改为 train 集计算
- 提交列名 `target` → `prediction`（对照历史成功提交 `submission_v9_cos_a25.csv`）

## 3. PSEUDO 结果（定标用, 非提交模型）

- 定标模型：**p9_scfi_rq arm_C**（152+73Z, **m0-32 训练** / m33-70 eval, best-epoch 选择）
- frozen 51-70 基线：0.143239
- γ 扫描：0.25→+0.0002, 0.5→+0.0003, 0.75→+0.0004, **1.0→+0.0005**（单调，γ=1.0 最优）
- **选定 γ=1.0, frozen cos = 0.143758**

⚠️ **重要**：此 PSEUDO 是 arm_C 模型（m0-32 训练）的 eval 表现，**不是**提交模型（0-70 全量训练）的表现。

## 4. 提交结果

| 提交 | ref | 状态 | LB |
|---|---|---|---|
| RQ full + neut γ=1.0 (列名错误) | 55543480 | ERROR | — |
| RQ full + neut γ=1.0 (修正) | 55543658 | COMPLETE | **0.116** |

**历史锚点**（LB, cosine 越大越好）：
- v8b (v7+lb142 0.5/0.5): **0.142**
- v7 (表格0.8+RealMLP0.2): 0.135
- RealMLP 单模型复刻: 0.134
- 本次 RQ full + neut: **0.116** ← 远低于所有锚点

## 5. 失败分析（如实）

### 已知事实
1. PSEUDO 0.143758 与 LB 0.116 差距 0.028，远超正常 PSEUDO-LB 差（v8b 0.1426 vs 0.142 ≈ 0.0006）
2. 提交模型 = RQ 全量 0-70 训练；PSEUDO 模型 = arm_C m0-32 训练。**两个模型，不可互相外推**
3. γ=1.0 是在 arm_C 模型上定标的，套用到全量模型 test 预测上——**γ 跨模型套用无依据**
4. 全量 RQ 模型本身在 test 上的表现从未被 PSEUDO 验证过（train 0-70 后没有留出窗口可验证）

### 可能原因（假设, 未逐一验证）
- **H1（最可能）**：RQ 全量模型 test 预测本身质量差（无验证窗口），neutralization 又在错误的 Z 关系上减去 γ·ŷ_Z，双重劣化
- H2：neutralization 的 Z 定标关系在 test 月份不成立（test 是未来月份，波动率-预测关系可能漂移）
- H3：全量训练协议（0-70, 无 best-epoch 选择）与 arm_C 协议（best-epoch on eval）差异大

### 教训
1. **PSEUDO 高分 ≠ 生产可提交**：生产模型必须与定标模型同源（同一协议、同一 checkpoint 选择），否则 PSEUDO 无外推意义
2. neutralization 的 γ 定标必须用**待提交模型自己**的 PSEUDO 预测（全量训练时留出 tail 窗口验证）
3. 提交格式列名 `sample_id,prediction`（详见 docs/kaggle-submission-format.md）

## 6. 后续行动（建议, 未执行）

- [ ] 用 P6 已验证的 RealMLP-C（152+Z, PSEUDO 0.139248 / blendΔ +0.0014）替代本次 RQ 全量路线——P6 已有完整的 PSEUDO 门禁记录
- [ ] 若坚持 RQ 路线：全量训练需留出验证窗口（如 0-60 训练/61-70 验证）定标 γ，再 refit 全量提交
- [ ] neutralization 与 v8b 融合叠加验证（P9 遗留, 未做）

## 7. 产物

- `output/p10_rq_prod_scfi/rq_scfi_test_pred.npz` — RQ 全量 test 预测 (647,896)
- `output/p10_rq_prod_scfi/submission_rq_scfi_neut.csv` — 提交文件 (LB 0.116)
- `output/p10_rq_prod_scfi/run.log` — 训练日志 (30 epochs)
- `output/p10_rq_prod_scfi/neutralize.log` — neutralization 日志
- `docs/kaggle-submission-format.md` — 提交格式规范
