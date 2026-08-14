# MSCapital 科研化方案 v1.2.0

> 来源: GPT1 + GPT2 对 project_report_v2 的评审拍板 (2026-08-11)
> 状态: **已批准, P0.5 执行中**
> 前版: v1.1.1

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0.0 | 2026-08-10 | 初版 |
| v1.1.0 | 2026-08-11 | 双GPT评审: 验证体系重构 |
| v1.1.1 | 2026-08-11 | GPT拍板: GO + P0 执行 |
| v1.2.0 | 2026-08-11 | v2报告双GPT评审: P0通过; 插入P0.5校准层; 收紧4结论措辞; P0.5与P1-1并行 |

## 双 GPT 评审要点 (全部吸收)

### 结论措辞修正 (关联≠因果)

| v2 表述 | 修正为 |
|---|---|
| 预测主力=漂移主力 | 高预测重要性特征与高 domain-discrimination 特征**高度重合** (关联) |
| 0.017 来自价差/深度/波动漂移 | covariate shift 是**重要候选机制**, concept drift 未排除 |
| MLP 假冠军 | MLP 的 CV1 优势高度依赖近期 regime; **需公平训练预算复核** (固定 optimizer steps) |
| 0.122 真实平台期 | 90D 表示+同质模型**局部平台**, 突破需新信息表示 |
| 月度cos上升→更可预测 | cosine 对尺度不敏感; 需检查 target 形状 (kurtosis/分位数/尾部); 措辞改为"observed cos 较高, 原因未区分" |

### 关键方法论修正

1. **Temporal Matrix 混入 training-size confound**: T1(30个月train) vs T4(50个月train) 分数差同时含 regime+数据量; MLP 固定 30ep 时大 fold 天然更多 optimizer updates → **P0.5-A Fixed-window Matrix** (固定 30m→10m)
2. **MLP fairness check** (P0.5-B): T1/T2/T3 上 30ep vs 固定steps vs 80ep+早停; 若 0.1106→0.119 则重跑 MLP 行; 若仅 0.112 则锁定"MLP 弱"结论
3. **P0.5-C Drift intervention**: R0(90特征) vs R1(删top10 drift) vs R2(归一化替换); folds T3/T4/H2/PSEUDO; 看 Δnear vs Δfar; 只有 domain AUC↓ 且 forecasting robustness↑ 才能升级"漂移=泛化瓶颈"
4. **Pseudo-LB38 更名 Long-Horizon Stress Fold** (33m train 未控制数据量, 是压力测试非严格 pseudo-LB)
5. **P1-1 特征规范**: 6 类 primitive (归一化OFI L1/L2 / add-cancel-execute imbalance / arrival+bustiness / large-order相对不平衡 / impact-resilience / fast-vs-slow flow); 大单用**相对阈值** (局部top5%, volume/local median); 只有 L1/L2 → 称 "2-level OFI" 不称 MLOFI; 无法可靠识别的 cancel/queue 特征**宁缺毋滥**
6. **P1-1 筛选轴**: 不是 Pearson 低相关, 是 **Alpha axis (Δcos_temporal) × Drift axis (ΔAUC_domain)** 双轴; 理想: temporal +0.0025 且 domain AUC +0.003 (不制造第二个 m_sp_mean)
7. **P1-2 设计**: 1s 秒级聚合先行 (60×12-20ch); 双塔 (FAST order/tx + SLOW market); **双通道归一化** (invariant channels → encoder + regime state channels → MLP), 不二选一; 第一版小 TCN (dilation 1/2/4/8, 32→64ch); sequence-only 先测 corr
8. **晋级规则 v3**: Hard gates = worst future fold 不降 + Long-Horizon Stress 不降 + **blend gain 多 fold 同号正**; decay slope 等 P0.5 后恢复; corr<0.90 仅前置诊断
9. **提交冻结**: 剩余 2 次配额保留, 等新 representation 在 Pseudo/远期 fold 稳定提升再提交

## 执行顺序 (P0.5 与 P1-1 并行)

```
P0.5-A Fixed-window Matrix (30m→10m × 4 folds)     ← 本轮
P0.5-B MLP fairness (T1/T2/T3, 公平预算)            ← 本轮
P0.5-C Drift intervention (R0/R1/R2 × 4 folds)     ← 本轮
  + monthly_target_diagnostics.csv                  ← 本轮
P1-1 Stationary Microstructure Representation (6类 primitive + 双轴筛选)
P1-2 1s dual-tower TCN (Kaggle GPU)
P2 sequence + tabular
```
