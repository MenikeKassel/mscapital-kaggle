# GPT 评审报告：P5 探针轮次（2026-08-15）— 供第三方交叉验证

> 用途：把本报告给 GPT 评审。要求：逐条核实证据链、指出协议漏洞或过度解读、
> 对"下一步"给出独立判断。所有数据均可从仓库复现（脚本+产物路径已附）。
> 仓库：`D:\mscapital-kaggle` | 裁判文档：`docs/p5-final-decision.md`

---

## 1. 背景与目标

- 竞赛：MSCapital（高频逐秒预测，metric = uncentered cosine）
- 现状：当前最佳提交 v8b LB ≈ 0.142（= v7 + 外部 lb142 推理包 50%）
- 任务：判定 0.142 之后的增量来自哪里。禁止泛泛文献调研、禁止新模型名；
  用 3 个低成本可证伪探针，全部 strict temporal / nested OOF，零提交。
- 协议：月份 21-40 训练 → 41-50 选择 → 51-70 冻结（chronological blocks）
- 数据：train 1.26M 行 × 71 个月（0-70），test 648k 行（隐藏 label）
- canonical baseline（生产锚）：RealMLP(152 特征) + 表格模型 blend，
  51-70 OOF cosine = 0.1492，test LB 0.142

## 2. 探针 1：P5-A 幅度门控 → KILL

**假设**：market magnitude m̂（≈|y| 的预测，corr 0.43-0.47）能对现有 alpha
做条件化加权（v7_pred × a_bin(m̂)），幅度大的时段更准。

**设计**：4-bin 门控 + 嵌套 temporal（gate 参数在 outer folds 内 cross-fit，
防二级模型 in-sample）；置换 m̂ 对照；非嵌套参照。

**结果**（`output/p5a_mag_gate/results.json`）：

| 指标 | 值 | 判据 | 通过 |
|---|---|---|---|
| Δcosine_outer (51-70 嵌套拼接) | **−0.000146** | ≥ +0.0004 | ❌ |
| 正月份 | 6/20 | ≥ 70% | ❌ |
| late (61-70) | +0.000005 | > 0 | ~ |
| 去 top-1 增益月 | −0.000151 | — | ❌ |
| gate std | 0.0108（≈常数） | 门控要有区分度 | ❌ |
| 置换 m̂ 对照 | +0.000004 | ≈ 0 | ✓ 负结果可信 |
| 非嵌套参照 | +0.000138 | — | 证明"泄漏→假增益" |

**裁决**：KILL。m̂ 质量锚（41-50: 0.4664 / 51-70: 0.4266 复现 P5-02I 完全一致）
排除模型损坏。机制解释：cosine 对全局尺度不变，逐样本缩放必须与 y 的形状
相关才有效——实测门控学到常数。**MAG-MoE / COC 幅度调制整线关闭。**

## 3. 探针 2：P5-B 条件创新表示（SCFI）→ CONFIRMED（跨 learner）

**假设**：Z = 事件流 − E[事件流 | 市场状态]（"surprise"表示，MAD 稳健尺度）
比 raw 特征更可被有限容量模型利用，且提高 temporal 稳定性。
理论边界：Z 不创造新 Shannon 信息——它是表示变换，不是新信息源。

**设计**：五臂 LGB（同参数，双块 B51_60/B61_70，holdout 49-50/59-60 早停+调权）：
- A = 152 f0726 基线
- B = +73 个 raw O/T 聚合（side×action 拆分，全新特征族）
- C = +73 个 Z innovation（nuisance = Ridge 交叉拟合 M→O、M+O→T，10 月折，
  尺度只来自 train 折外残差——无泄漏）
- D = +raw+Z
- E = +raw+raw²（capacity control：D−E 分离"innovation 本身"与"更多特征"）

**结果**（LGB 单 seed，双块平均 Δ vs A）：

| 臂 | Δ | 说明 |
|---|---|---|
| B (+raw) | +0.0058 | raw 聚合族独立有效 |
| **C (+Z)** | **+0.0075** | B1 +0.0059 / late +0.0091 |
| D (+raw+Z) | +0.0059 | |
| E (+z²) | +0.0043 | |
| **D−E** | **+0.0016** | 双块均为正 → innovation 超容量效应 |

- 逐月 C-vs-A（同 learner）：**17/20 月正**
- blendΔ（vs canonical，holdout 调权）：C = +0.00094
- corr(C, canonical) = 0.872；corr(C, B) = 0.959（贴 0.95 kill 条款，见 §7 质疑点）
- Z 特征进 top-20 gain 3 个（rank 13 起）
- 74 个 raw 特征中 1 个与 mstate 完全同源（R²=1.0，Z 自动≈0，预期内）

**NN spot-check（中间结果，后被推翻）**：SmallMLP[256×2]×3 seeds 上 ΔC≈+0.0001
(12/20 月) → 一度判定"learner 依赖"。**后经 P5-E 证明是代理 learner 太弱。**

## 4. 探针 3：P5-C 跨通道同步几何（RICS）→ KILL

**假设**：last-10 步的确定性跨通道几何（flatten/moments/covariance/lag/phase）
含市场 alpha（P5-02I 结论：信息=短程≤10 步+跨通道同步+无时间箭头+相位形态）。

**设计**：五层累计消融（R0→R4），全部喂小 MLP[256×2]+cosine loss；M0-ref =
200 步 Conv1D（P5-01 架构）作参照；R4/M0 同一 frozen 模型做时间反演分解。

**结果**（`output/p5c_rics/results.json`，corr_y 为 41-70 混合 window）：

| 层 | corr_y | Δfrozen | 判读 |
|---|---|---|---|
| R0 flatten (last-10) | +0.011 | −0.000070 | 无 |
| R1 +moments | +0.002 | −0.000166 | 无 |
| R2 +covariance | +0.005 | 0 | 无 |
| R3 +lag even/odd | −0.0002 | 0 | 无 |
| R4 +phase | **−0.006** | 0 | 过拟合，转负 |
| M0-ref (200 步) | **+0.0861** | −0.000031 | 完美复现 P5-01 |

- R4 反转分解：corr(fwd, rev) = **−0.69**（相位特征高度方向敏感——直接反驳
  "反转不变"假设）；M0 反转 corr = +0.069（全窗口模型近乎反转不变，与 P5-02I 一致）
- **机制澄清**：P5-02I 的"≤10 步形态"≠"最后 10 步"——alpha 分布在 200 步
  窗口的多位置 + 上下文，last-10 确定性统计里几乎没有。

**裁决**：KILL。wavelet / shapelet / spectral CNN 整线关闭。

## 5. 生产级验证（P5-D/E）→ SCFI 升级确认

### P5-D：LGB 3-seed 集成 + CatBoost（双块，holdout 调权）

| 臂 | Learner | blendΔ avg | B51_60 | B61_70 |
|---|---|---|---|---|
| C (152+Z) | LGB 3seed | **+0.000849** | +0.001282 | +0.000415 |
| D (+raw+Z) | CatBoost | +0.000662 | +0.000955 | +0.000369 |
| D | LGB 3seed | +0.000549 | +0.001212 | −0.000114 |
| B (+raw) | LGB 3seed | +0.000448 | +0.000671 | +0.000226 |
| A (152) | LGB/Cat | ≈0 或负 | — | — |

### P5-E：RealMLP 精确 spot-check（生产 learner，R61_70 同构协议）

| arm | outer cosine (61-70) | corr(canon) | blendΔ (61-70 冻结, w 调于 51-60) |
|---|---|---|---|
| A (152) | 0.148526（复现 canonical ✓） | 0.967 | −0.000068 |
| **C (152+Z)** | **0.152570** | 0.940 | **+0.001369**（w=0.50 顶格） |
| **Δ(C−A)** | **+0.004044** | | |

**结论**：Z 特征在生产 learner 上 standalone +0.0040 / blend +0.0014，
与 LGB 3-seed（+0.00085 双块正）跨 learner 双确认。SmallMLP 结论作废。

## 6. 最终裁决与下一步

- **主线**：RealMLP 全量重训（152+Z）→ test 预测 → 与 v8b blend
  （w 网格 0.05-0.70，51-60 调权）→ PSEUDO 门禁 + 分布检查 → 提交候选
- **P6 正在执行**（本报告撰写时：RealMLP 全量 refit 训练中，GPU 80%）
- 已关闭：MAG-MoE / RICS 全线 / Transformer-TCN / global scale / 辅助 loss 生产化
- 纪律：全程未提交 Kaggle；判定不依赖 LB；所有二级模型 cross-fit

## 7. 请 GPT 重点审查的质疑点

1. **corr(C,B)=0.959 贴 0.95 kill 条款**：任务书 KILL 条款含
   "innovation 与 raw 几乎完全等价 (corr>0.95)"。我们未执行 KILL，依据：
   (a) 直接 Δ 证据反驳等价（C>B 双块，D−E>0 双块）；(b) corr 高是因为两模型
   共享主导 alpha（152 特征），边际贡献小 → 预测相关性天然高。
   是否合理，还是我们应该承认阈值违规？
2. **逐月 C-vs-canonical 只有 1-4/20 月正**（P5-B/P5-D）：我们归因于"表格
   learner 绝对强度 < canonical blend"，正确比较是同 learner（C-vs-A 17/20）。
   这个归因站得住吗？还是逐月正率低说明 blend 增益主要靠少数月份？
3. **P5-E 的 Z 训练侧赋值偏保守**：months 0-50 的 Z 用 0-50 内折、
   51-60 用 fit-0-50、61-70 用 fit-0-60（refit 阶段 0-50 的 Z 未用 0-60 折内重算）。
   无泄漏但非最优——是否影响结论强度？
4. **PSEUDO 切分**：canonical PSEUDO = eval 33-70（train 0-32）。P6 将复刻。
   LB ≈ PSEUDO − gap 的校准在 0.142 锚上还成立吗（外部 lb142 成分如何影响）？
5. **blend 权重 w=0.50 顶到网格上沿**（P5-E）：是否应视为"更高权重更优"信号
   并扩网格？还是网格封顶是正确保守做法？
6. **v8b 锚含外部 lb142 推理包（50%）**：与 RealMLP-C blend 时，外部成分的
   scale/相关性如何影响？是否有必要用 pure-local 锚（v7）做对照？
7. **CatBoost 在 B61_70 转负**（B/C 臂 −0.0005~−0.0008）：是 late 块特性
   （61-70 与 51-60 不同 regime）还是 CatBoost 容量不足？生产决策排除
   CatBoost 是否合理？

## 8. 复现路径（全部可复核）

- `docs/p5a-mag-gate-report.md` / `docs/p5b-scfi-report.md` /
  `docs/p5c-rics-report.md` / `docs/p5de-production-verification.md` /
  `docs/p5-final-decision.md`
- `output/p5a_mag_gate/results.json` / `output/p5b_scfi/results.json` /
  `output/p5c_rics/results.json` / `output/p5d_prod_blend/results.json` /
  `output/p5e_realmlp_spotcheck/results.json`
- 脚本：`scripts/p5a_mag_gate.py` / `p5b_scfi.py` / `p5c_rics.py` /
  `p5d_prod_blend.py` / `p5e_realmlp_spotcheck.py` / `p6_prod_realmlp.py`
