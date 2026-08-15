# GPT 评审请求：RealMLP 研究进展 + O→T lag response 探针方案 (2026-08-15)

> 评审对象：GPT（外部评审循环）
> 评审请求方：MSCapital 研究（Hermes 整理）
> 前置约定：请区分【论文事实】/【机制解释】/【推断】三档证据；所有数字可复现，复现路径见 §7。

---

## 0. 请 GPT 评审的三件事

1. **核心方案**：O→T lag response 探针（§4）——设计是否合理？门禁是否够严？有没有更便宜的验证路径？
2. **自曝质疑点**（§6）——我们明确知道自己不确定的地方，请逐一评审。
3. **风险盲区**——我们没看到的坑（包括但不限于：指标-方法交互、漂移、序列模型的前科 v6 TCN 灾难）。

---

## 1. 本轮研究摘要（30 秒版）

- 深读了 RealMLP 论文（arXiv 2407.04491 v3，Better by Default）+ 官方代码 pytabkit + 轻量实现 realmlp-td-s_standalone，并核验了此前一份 GPT 评审（23 条断言，20 条属实、4 处小错、采纳 5 条增量）。
- 逆向分析了公开 LB 0.142 推理包（yangq369/lb142）：**5×序列模型 (0.6) + 1×RealMLP (0.4)**，确认不可完整复刻（grids 制造代码缺失 + factors 特征黑箱）。
- 核对公开 RFMF RealMLP notebook（yunsuxiaozi，LB 0.134）源码：**我们项目的 152 特征（f0726 复刻）就是它的特征体系**；它做了"跨表统计交叉"（x_ 特征，同期比值/差值），**没有做"跨表时序联动"**（滞后/相位/事件对齐）。
- 全项目方法查重（method-map + registry + failed-experiments 三件套）：聚合类特征家族**几乎全部做过**（多为负结果）；**唯一未测 = O→T lag response + ETCI 事件时**。

## 2. 事实基线（数字已校准，口径：uncentered cosine，越大越好）

| 提交/模型 | 特征体系 | LB | 备注 |
|---|---|---|---|
| v1-v3 表格融合 | 90 特征 | 0.122 | 树+MLP，90 特征甜点位（社区+自证） |
| v5 表格融合 | 112 特征（90+22 微观） | 0.125 | P1-03 最大单步 +0.0036 (PSEUDO) |
| 公开 RealMLP 单模型 | 152 特征（0726 复刻） | 0.134 | = rfmf-realmlp notebook |
| v7 表格0.8+RealMLP0.2 | 152 特征 | 0.135 | |
| v8b +lb142 0.5 | 152 特征 | 0.142 | 当前最佳，排名 #30 |

本地校准（docs/calibration.md）：Regime A（表格）LB ≈ PSEUDO − 0.008~0.010；Regime B（RealMLP 族）gap ≈ 0.0047（仅 1 点，不当公式用）；Regime C（外部预测）无共同验证，只当 probe。

## 3. 关键研究发现

### 3.1 RealMLP 论文（论文事实）
- RealMLP = 系统性调优的 MLP recipe（3×256 隐藏层），非新架构。核心组件：robust scaling + smooth clipping、PBLD 数值嵌入、learnable feature scaling、parametric Mish、NTP、β2=0.95、coslog4、scheduled dropout/wd。
- 消融（Table B.1/B.2，TD-S 上测，误差增加%）：β2=0.999 +22.8%（回归）、无嵌入 +20.6%、无 smooth clip +9.5%、constant LR +13.5%、constant dropout +3.6%、constant wd +3.1%、parametric act +4.8%、scaling layer +1.0%、NTP +1.3%、数据驱动 init +1.2%。
- 逐数据集（附录 D 全 12 表解析）：**大样本回归是 RealMLP 主战场**（meta-test REG 对 CatBoost/LGBM/XGB 胜率 67.5%/61.9%/76.3%）；小样本（Grinsztajn ≤10K）输 CatBoost（35.7-36%）；高维噪声特征（madelon +145%）、高频原始时序（electricity +45%）是弱区。
- 集成实证（Table B.4）：refit 5 模型 + joint stopping 回归误差降 ~8.7% > 任何单组件；refit > bagging。
- HPO 空间（Table C.14）：lr 是唯一大范围搜索的连续参数；β2/coslog4/NTP 未入搜索空间（默认即最优）。

### 3.2 lb142 逆向（源码级）
- 结构：`pred = 0.6 × mean(unit(5×v9 MultiStream)) + 0.4 × unit(v10 RealMLP)`。
- v9 = 三流序列模型：market 200 步×12 通道 + tx 60 步×8 + order 60 步×10，CNN+Attention+**CrossStreamMixer（跨流混合层）**，pred_demean=True，lr=1e-3, wd=1e-4, cosine 调度。
- v10 = **论文完整组件 RealMLP**（RobustScaleSmoothClip + PBLD + ScalingLayer + NTP，n_ens=8，widths 256-256-64）+ **λ_cos=0.05 的 centered-cosine 混合损失**（MSE + 0.05×(1−cos)，pred/target 均去均值），EMA 0.998。
- **不可复刻**：grids（数十 GB 序列棋盘）制造代码不在包内（通道语义只能逆向猜测）；factors（best_factors parquet，v10 特征）完全黑箱。仅推理可复现（确定性 blend，与 submission_ref 逐行一致 <1e-8）。
- 对我们最重要的结论：**lb142 的独立信息（corr≈0.82，~18%）大概率来自序列流的跨表时间结构**——表格特征（无论谁造的）都缺这个维度。

### 3.3 rfmf-realmlp / 0726 特征（源码级核对）
- 152 特征 = 0726 notebook 输出（单表聚合 5 块 + x_ 交叉 ~50 个）。
- 跨表统计交叉有：x_trans_order_vol_ratio、x_trans_order_buy_diff、x_tx_order_rate_ratio 等（同期比值/差值）。
- **跨表时序联动完全没有**：无滞后响应、无相位、无事件对齐、无互相关——所有聚合都是单表内完成后再 join。
- 结论：**"152 特征无覆盖 O→T lag" 得到代码级证实**。

### 3.4 方法查重（三件套）
- 已做过（多为负结果）：OFI（P4-04 弱正）、到达率/burstiness（P1-01，P1-04 相对化 RED）、事件流（M-01 RED）、波动率类（E01/P4-17/P7-01，failed-experiments 明令"不重复"）、LOB 几何/交互/签名（M-02~M-05 全 RED）、网格/自监督（P3-01/03/04 RED）、market 表内跨通道同步（P5-02I GREEN，但**表内**）、lag-cov/相位（P5-05 RED，**market 表内**）、SCFI 条件创新（P5-04 GREEN 有效）。
- **唯一未测**：O→T lag response（order 表 × transaction 表的跨表时序滞后）；ETCI 事件时。

## 4. 核心方案：O→T lag response 探针

### 4.1 定义
把 order 表与 transaction 表的事件流按时间轴对齐，度量"order 事件 → 后续 k 秒窗口内 tx 响应"的滞后结构（转化率、响应时延、响应强度），并检验其对未来收益（y）的信息含量。

### 4.2 依据
- 正向：lb142 v9 的 CrossStreamMixer 吃三流棋盘（外部证据，LB 0.142 的 60% 权重）；P5-02I 证明"表内跨通道同步是核心信息"（机制暗示，但为表内）；152 特征代码级确认无此维度。
- 反向风险：P5-02I 结论是"短程（≤30s）+ 跨通道同步 + 时间方向无关"——O→T 若成立，大概率同样短程；P5-05 证明 market 表内 lag-cov/相位无信息（跨表是否不同未知）。

### 4.3 探针设计（三阶段，成本递增，每阶段有独立门禁）

**Stage 1：诊断探针（最便宜，先做）**
- 在原始 order/transaction 事件上直接计算跨表滞后统计量：lag-k 窗口内 tx 对 order 的响应（如"order 流入后 k 秒内 tx 成交量/方向"），k ∈ {1,3,5,10,30}s。
- 检验：corr(滞后统计量, y)、月度稳定性（逐月 corr 一致性）、与 152 特征的 corr（正交性门禁 <0.7）。
- 门禁：corr(y) 显著 + 20 月正比例 ≥ 70% + 与 152 正交。

**Stage 2：特征化（若 Stage 1 过门禁）**
- 把通过门禁的滞后特征做成表格特征（窗口聚合版），喂现有 RealMLP 复刻 + LGBM，按 PSEUDO 门槛评估。
- 注意：这本质上就是"跨表时序特征的聚合形态"，与 lb142 factors 的可能构成一致。

**Stage 3：序列化（若 Stage 2 有效）**
- 三流棋盘（market 200×18 / tx 60×8 / order 60×10，market-centered 价格表示 + z-score/clip 归一化）+ CrossStreamMixer 复刻（lb142 v9 结构，借鉴而非逐字节复刻）。
- 预注册门禁：test 侧 corr 结构验证（v6 TCN 灾难教训：PSEUDO 对序列模型不可靠，test corr(tab,seq) < 0.03 是致命预警）。

### 4.4 与并行线的优先级
- O→T 探针（Stage 1 半天内出诊断）优先于 RealMLP 组件迁移的**部分条目**（组件迁移见下）。
- RealMLP 组件迁移仍建议：robust+clip → β2=0.95 → cosine LR decay → parametric Mish → PL/PBLD → scaling layer → scheduled reg → coslog4 → NTP/init（论文消融排序 + 实现成本排序），但**所有组件实验必须在 cosine 指标下重新验证**（论文收益是 nRMSE 口径，cosine 下可能重排——lb142 v10 的 λ_cos=0.05 是唯一实战参照）。

## 5. 并行待办（不动，仅告知）
- lb142 的 factors/grids 获取：唯一途径是联系作者（yangq369/Yallon），本地无。
- ETCI 事件时：与 O→T 同源，排在 O→T 之后。
- 提交配额纪律不变：本地双窗口 + 分布门禁才提交。

## 6. 自曝质疑点（请 GPT 重点评审）

1. **"唯一推荐"是否过度自信**：O→T lag 的外部证据（lb142）和机制暗示（P5-02I 表内同步）都是间接的；跨表时序在项目内零覆盖不等于有信号。Stage 1 门禁能否足够快地证伪？
2. **lb142 的 18% 独立信息归因是否成立**：也可能是 market 序列 + 不同特征/不同 seed 造成的，不一定是 order→tx 时序。我们有没有更便宜的归因验证（例如对 lb142 submission 与 v7 的分歧样本做特征级诊断）？
3. **cosine 指标下 RealMLP 组件收益重排假设**：论文所有组件消融在 nRMSE 下；我们假定"与指标正交的组件（预处理/β2/schedule）收益保持、与幅度相关的组件（output clipping、PBLD 的幅度表示）收益缩水"。这个假设有没有反例？
4. **Stage 3 序列模型的前科**：v6 TCN PSEUDO +0.004 → LB 0.082 的灾难说明序列模型在本数据的验证协议不可靠。CrossStreamMixer 复刻是否值得做？还是停留在 Stage 2（特征化）就够了？
5. **P5-02I 的"时间方向无关"结论**（reverse 几乎不掉）是否暗示 O→T 的因果方向性本身不重要，从而 lag response 的"方向"版本（order→tx）和"反向"版本（tx→order）应该同时测？

## 7. 复现路径（所有证据可查）

| 证据 | 位置 |
|---|---|
| 论文全文（v3） | research/paper-reading-2026-08/realmlp.txt（+ html2txt.py 转换脚本） |
| 附录 D 逐数据集解析脚本 | research/paper-reading-2026-08/parse_appendix_d.py（12 表可复算） |
| pytabkit 官方代码 | research/paper-reading-2026-08/pytabkit_code/ |
| realmlp-td-s_standalone | research/paper-reading-2026-08/tds_standalone_code/ |
| lb142 推理包（33 文件，md5 已与 Kaggle 原始上传核对一致） | D:\mscapital-forecasting\reference\lb142\ |
| rfmf-realmlp + rfmf-0726data 源码（Kaggle pull API 获取） | research/paper-reading-2026-08/rfmf_realmlp_source.ipynb、rfmf_0726data_source.ipynb |
| 方法台账三件套 | docs/method-map.md、experiments/registry.csv、docs/failed-experiments.md |
| 校准协议 | docs/calibration.md |
| P5-02I 信息审计 | docs/p5-02i-info-audit-report.md |
| P5-05 RICS | docs/p5c-rics-report.md |
