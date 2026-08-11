# 主题池（R1 调研产出 + 既有候选，M0 管理）

> 每个候选标注来源与状态。状态：`待选` / `进行中` / `已落地` / `已证伪` / `搁置`。
> 选题优先级见 COLLABORATION.md §9（直击 12,12 缺口 > 普适降 shots/提速 > 精度精修 > 其余）。
> 基线数据：12,12 baseline err 2.28e-4（dim~1e5, 落后 SHCI ~20×）；10o baseline 9.76e-7（近收敛）。

## 直击 12,12 缺口（采样覆盖 / 大空间对角化效率）

| 候选 | 来源 | 思路（一行） | 状态 |
|---|---|---|---|
| 半随机 semistochastic PT2 | REVIEW「方向③-B」| 近场确定性 + 远场抽样估 E_PT2，降大体系枚举成本 | 待选 |
| 更大采样量 + 改进采样分布 | REVIEW L2 结论 | 12,12 覆盖顽固，需 Ansatz/采样端突破 | 待选 |
| Krylov 型采样电路（SKQD 式化学适配）| SURVEY §7.2 / §8.9 | 多尺度参数化酉族张子空间，超出单电路覆盖 | 待选（SKQD 作者自评化学近期不可用，需评估）|
| NO 自洽换基迭代扩展（开壳层）| REVIEW「后续可选」| 换基提升稀疏度 → 减对角化维度（b62d636 已否决 adaptive 收缩路径，NO 自洽不同）| 待选 |
| linkstr GPU 子空间修正 | REVIEW GPU 节 | 全空间 mid 内存爆炸（t1=norb²·na_full·nb），探索压缩方案 → solve_sci(gpu) 大维度提速 | 待选 |

## 普适降 shots / 提速

| 候选 | 来源 | 思路（一行） | 状态 |
|---|---|---|---|
| 本征矢重要性采样 → NQS/ARNN 泛化先验 | REVIEW「后续可选」/ SURVEY §8.9 | Transformer ARNN 生成位串替代电路采样（风险 4/5，重依赖）| 搁置 |
| qDRIFT 随机化 | SURVEY §8.9 | 降 Krylov/演化电路深度（中低优先）| 待选 |
| `recommend_sqd_params` 接入真实校准 | REVIEW「后续可选」| calibrate 拟合的 KS/KT1 回填推荐器 | 待选 |
| `noise_impact` 支持 T2/读出 + 多参数安全区 | REVIEW「后续可选」| 噪声评估扩展 | 待选 |

## 精度精修（弱/中关联）

| 候选 | 来源 | 思路（一行） | 状态 |
|---|---|---|---|
| 完整 J + 多参数 orbital rotation（ffsim 级 UCJ）| REVIEW「后续可选」/ SURVEY §6-5 | 替换简化 SVD UCJ | 待选 |
| 自旋分辨哈密顿量（h_alpha ≠ h_beta）| REVIEW「后续可选」| spin-orbital SQD 后端，解锁 UHF | 待选 |
| OBDF v^ext 10× 归一化开放问题 | REVIEW OBMP2 节 | 解析 10× 常数来源 → 移除 scale 经验参数 | 待选 |
| 配置恢复 tie-breaking 随机性统计测试 | REVIEW「后续可选」| 稳定性工程 | 待选 |

## 文献新方向（R1 每轮可补充，避免与已落地重复）

已落地方向（**不要再提**）：CSQD 聚类恢复（f75d817）、自旋 λ 惩罚（542ed52）、
AS-SQD 主动采样（solve_sqd_active）、OBDF/OBMP2 下折叠（obmp2 模块）、GPU matrix-free
（matrixfree + selected_ci_gpu）、PT2/evpt2 修正、distill 自蒸馏、CIPSI/HCI/SHCI 库内实现、
自然轨道换基（basis 模块）。

## 已证伪 / 明确排除（防重复调研）

| 候选 | 结论 | 数据/出处 |
|---|---|---|
| A1 无限采样外推（E vs 1/√S）| 证伪：SQD 能量是变分下界非统计量 | REVIEW「A1/A3」err 3.6e-2 vs 3.6e-5 |
| σ² 线性外推 | 证伪：过冲到 FCI 之下 | REVIEW 方向 D（N₂/C₂ 均过冲）|
| include(S+D) 用于 >2 occ/spin | 排除：N₂ 7e/spin 停在 2.25e-2 平台 | REVIEW「第 6 轮收尾」|
| adaptive NO 换基（大体系默认参数）| 证伪：子空间缩 1/3，差 7-45× | REVIEW L2-a |
| UCJ 采样（CCSD 不收敛时）| 排除：R=3.0 RHF-CCSD 不收敛 → t2 不可靠 | REVIEW L2-b |
| distill（远未收敛体系）| 证伪：学错分布、有害 | REVIEW「L1」12,12 r2-4 恶化 |
| pyhci / pyscf-forge / naive-hci 外部包 | 排除：网络/编译不可装 | REVIEW「真正的 HCI 实现」|
| cupy-cuda13x | 排除：cu13 wheel 全 0.0.1 占位，缺 libcurand | HANDOFF §0 |
| AB-SND ARNN（近期）| 搁置：只测自旋模型未测分子化学 + 重依赖 | SURVEY §8.9 |
