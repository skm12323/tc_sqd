# tc_sqd 理论综述：功能体系、方法探索与对比

> 本文从**理论角度**严谨综述 tc_sqd 仓库：① 库的功能体系如何在数学上成立；
> ② 各方法探索（基础 SQD → SQD+VQE → include 单双激发 → 方向 A UCJ 辅助 →
> 方向 B CIPSI）的动机、机制与边界；③ 与经典方法（CCSD/CCSD(T)/CISD）及
> 确定性选态方法（HCI）的对比。所有数值均来自本仓库在 `WSL + conda tc`
> 环境、STO-3G 基组下的实测。
>
> 配套文档：`README.md`（使用）、`REVIEW.md`（审查与验证历史）、`DISCUSSION.md`（双 agent 协作）。

---

## 1. 引言：基态问题的指数复杂度与量子-经典混合路线

分子基态能量问题可表述为：给定 $N$ 个电子的费米子哈密顿量 $H$，求其基态
$E_0 = \langle\Psi_0|H|\Psi_0\rangle$。在完备单粒子基下，$H$ 作用于
$\binom{M}{N_\alpha}\binom{M}{N_\beta}$ 维的 Fock 空间（$M$ 为空间轨道数），
维度随体系指数增长——精确对角化（FCI）只适用于极小体系。

**变分原理** 是所有波函数方法的共同基石：

$$E[\Psi] = \frac{\langle\Psi|H|\Psi\rangle}{\langle\Psi|\Psi\rangle} \ge E_0 .$$

求解策略按"波函数 Ansatz 的刻画方式"分为三类：

| 路线 | Ansatz 形式 | 代表 | 强关联适用性 |
|---|---|---|---|
| 经典单参考 | 有限阶耦合簇/微扰 | CCSD、CCSD(T)、MP2 | 弱-中关联 |
| 确定性选态 | 迭代选择重要行列式 | HCI、CIPSI、SHCI | 强关联（空间大） |
| 量子采样 | 量子电路采样张成子空间 | **SQD** | 取决于采样覆盖 |

**Sample-based Quantum Diagonalization (SQD)** 属于第三条路线：它把"波函数
Ansatz"替换为"**采样得到的行列式集合**"，在采样张成的子空间内做对角化。
其核心洞察是：**变分下界不受 Ansatz 形式限制，只受子空间对基态的覆盖限制**——
采样覆盖越好，对角化结果越接近 $E_0$。tc_sqd 的全部方法探索正是围绕
"如何让采样张成的子空间更高效地覆盖基态"展开。

---

## 2. 理论基础

### 2.1 费米子哈密顿量与二次量子化

在空间轨道基 $\{\phi_p\}$ 下，电子哈密顿量为

$$H = \sum_{pq} h_{pq}\, a^\dagger_p a_q + \tfrac{1}{2}\sum_{pqrs} \langle pq|rs\rangle\, a^\dagger_p a^\dagger_q a_s a_r,$$

其中 $h_{pq}$ 为单电子积分，$\langle pq|rs\rangle$ 为双电子积分（本库用 chemist
记号 $(pq|rs)$）。选择正交归一 MO 基后，积分变换为 $h = C^\dagger h_{AO} C$ 与
四指标收缩。本库 `from_pyscf` 自动完成 AO → MO 变换、核排斥能提取、frozen-core
修正（core 能量 + core 对活性的平均场势），确保哈密顿量在活性空间内自洽。

### 2.2 Slater 行列式与 CI 展开

任意 $N$ 电子态可按 Slater 行列式展开：

$$|\Psi\rangle = \sum_{I} c_I\,|\Phi_I\rangle,\qquad |\Phi_I\rangle = \prod_{p\in I} a^\dagger_p |0\rangle .$$

行列式间的哈密顿矩阵元由 **Slater–Condon 规则**给出：仅相差 $\le 2$ 个
自旋轨道的行列式对才有非零矩阵元。这是本库 `build_ci_matrix`（构造显式 CI
矩阵）与 `solve_sci`（子空间对角化）的数学基础——由于矩阵的稀疏连接结构，
即使子空间维度上万，矩阵-向量乘积也高效。

### 2.3 变分原理与子空间对角化

设子空间 $V = \mathrm{span}\{|\Phi_I\rangle\}_{I\in\mathcal{S}}$，投影算子
$P_V$。子空间基态满足

$$P_V H P_V\,|\Psi_V\rangle = E_V\,|\Psi_V\rangle,\qquad E_V \ge E_0 .$$

$E_V \to E_0$ 当且仅当子空间对基态 $|\Psi_0\rangle$ 的覆盖趋于完备：
$\|P_V|\Psi_0\rangle\| \to 1$。这一视角决定了 SQD 的**误差来源是子空间覆盖
而非采样噪声本身**（尽管覆盖也受采样噪声影响）。

### 2.4 SQD 的一般框架

SQD 的典型流程（本库 `diagonalize_fermionic_hamiltonian` 的迭代循环）：

1. **采样**：从参数化量子电路（HF、LUCJ、UCJ 或任意电路）采样 $S$ 个比特串。
   每个比特串按 `[β_{n-1}…β_0 | α_{n-1}…α_0]` 约定编码一个 Slater 行列式。
2. **配置恢复**：修正采样引入的粒子数违例（噪声翻转比特）。
3. **子空间构造**：比特串去重 → 字符串集合 → 采样张成的子空间
   $\mathcal{S}$（可并入外部强制配置 `include_configurations`）。
4. **对角化**：求 $P_V H P_V$ 最低本征值 $E_V$。
5. **迭代**：以当前本征矢更新平均占据数，作为下一轮配置恢复的参考（occupancy
   refinement），直至收敛。

**关键性质（口径澄清，见 §5.2）**：本库的子空间表示是**字符串乘积**
（α 集合 × β 集合，维度 $= n_a \times n_b$），而非采样得到的 det 对集合。
对角化维度与采样成本（shots）是两回事。

---

## 3. 仓库功能体系（按理论模块）

### 3.1 采样与比特串处理（`counts` / `sampler`）

- **比特串约定**：`[β_{n-1}…β_0 | α_{n-1}…α_0]`，与 qiskit-addon-sqd 一致。
  这一约定贯穿采样、恢复、对角化全链路，是避免静默错算的关键。
- **统一采样后端** `sample(circuit, n_samples, backend=...)`：模拟器（tc）/
  真机（qcloud）一行切换，下游 SQD 流水线不变。

### 3.2 配置恢复（`configuration_recovery`）

采样比特串因噪声偏离正确粒子数。本库两条恢复路线：

- **平均占据恢复** `recover_configurations`：以参考占据向量为锚，通过随机填充
  修正粒子数违例（qiskit-addon-sqd 的启发式）。
- **T1 感知恢复** `estimate_true_occupancies`：从观测位串**反卷积**振幅阻尼
  （T1）造成的占据低估，得到真实平均占据再喂回恢复器。per-qubit γ 不均匀时
  RMSE 降 ~33%。

### 3.3 子空间对角化与 SQD 求解器（`fermion`）

三种求解路径，统一入口 `compute_ground_state_energy(method=...)`：

| method | 数学 | 适用 |
|---|---|---|
| `fci` | 全空间 Davidson（PySCF `direct_spin1`）| 精确基准 |
| `direct` | 显式 CI 矩阵 + `numpy.linalg.eigvalsh` | 小体系/教学 |
| `sqd` | 迭代 SQD（采样 → 恢复 → 对角化 → 更新占据）| 量子采样 |

**稳健对角化**（`solve_sci` 基态分支）：小空间（dim ≤ 1000）显式建 H + numpy
eigh；大空间用 scipy `eigsh`（ARPACK SA）。这是准简并体系的必要保障——
Davidson 迭代可能收敛到局部根（见 §4.5 的 C₂ 案例）。

### 3.4 量子态制备：LUCJ → UCJ（`lucj`）

采样电路的质量决定子空间覆盖。本库从弱到强提供三级：

- **HF**：无关联，单 det。
- **简化 LUCJ** `build_lucj_circuit`：CCSD t2 振幅驱动的占据-空 Givens 门。
  **必须由 t2 而非 t1 驱动**——H₂ 的 t1≈0（Brillouin 定理），相关能几乎全来自 t2。
- **UCJ** `build_ucj_circuit` + `ucj_decomposition`：t2 → occ-vir 块 SVD →
  多层 $(\hat{U},\hat{J})$，其中 $\hat{U}=e^{\kappa}$ 为 anti-Hermitian 轨道
  旋转，$\hat{J}$ 为对角 Coulomb。UCJ 态取精确形式

$$|\Psi_{\mathrm{UCJ}}\rangle = \prod_{\ell}\left(\hat{U}_\ell\, e^{i\hat{J}_\ell}\, \hat{U}_\ell^\dagger\right)|\mathrm{HF}\rangle .$$

**理论要点（本库实测）**：UCJ **单态期望** $\langle\Psi|H|\Psi\rangle$ 对单层
对角 J **无法低于 HF**——$\hat{U}$ 旋转把 HF 展开引入对称禁戒的单激发，污染
拉高能量。因此 UCJ 的价值**不在单态、而在子空间对角化**（`ucj_subspace_energy`
确定性验证 H₂ = FCI）。`include_jastrow=False` 默认成立：SQD 采样只依赖
$|\Psi|^2$，对角 $e^{i\hat{J}}$ 相位不改采样概率 → 省略 RZZ 省深度无损。

### 3.5 噪声模拟与误差预测（`noise` / `predict`）

- **噪声模拟**：密度矩阵 Kraus 通道（退相干/振幅阻尼/去极化）。关键机制
  （实测）：**退相干不改 diag → SQD 免疫**；振幅阻尼改 diag（T1 主导误差）；
  去极化保迹。
- **误差预测**：解析模型 $\varepsilon = K_S/\sqrt{S} + K_{T_1}\,\gamma_{T_1}$
  （基态；激发态 ×3），其中 $K_S$ 反映采样统计、$K_{T_1}$ 反映振幅阻尼率
  $\gamma_{T_1} = 1-\exp(-\text{depth}\cdot t_g/T_1)$。`calibrate` 在用户自己的
  电路/体系上拟合 $K_S, K_{T_1}$（注意 $K_S$ 依赖电路覆盖质量，勿锚定固定值）。

### 3.6 真机集成（`hardware`）

校准加载 → 最优 qubit 子图选择（min T₂ 最大化 + 连通性 + BFS 序）→ 真机采样
（REM + 字节序自校准）→ SQD 后处理。为白名单受限环境设计，模拟器 mock 测试覆盖
REM 回退与字节序分支。

---

## 4. 方法探索历程

本节按"误差来源逐步消除"的线索梳理方法演进。核心问题始终是：**子空间覆盖
不足是 SQD 误差的根源**（而非 Ansatz 或优化本身）。

### 4.1 基础 SQD：采样统计极限

从固定电路采样 → 子空间对角化。LiH 实测：纯采样误差 ~3.9e-3（5-seed std
1.6e-3），采样统计主导。提高 shots 可降 $K_S/\sqrt{S}$ 项，但覆盖的**系统性**
缺陷（电路张不成关键 det）无法靠 shots 弥补。

### 4.2 SQD+VQE：过拟合问题与 n_seeds

把采样后的 SQD 总能量作为损失，用 Nelder-Mead 优化 LUCJ 角度。**关键教训**：
单 seed 优化在训练集（固定随机种子）上误差 -6e-4，但换 seed 验证反弹到
+5.5e-3——**固定 seed 严重过拟合**。多 seed 平均目标（`n_seeds=3`）缓解到
+1.1e-3。这揭示了 SQD+VQE 的统计极限：误差 ~1e-3 受采样统计约束，无法靠
优化突破。

### 4.3 include 单双激发：= FCI 的机制与前提

`excited_configurations` 经典生成 HF + 单/双激发配置，强制并入子空间
（`include_configurations`），采样只提供权重。LiH 上误差 +4e-3 → **1e-16
（= FCI）**，1000 shots 即达。

**机制（防误读）**：LiH/STO-3G **每自旋仅 2 电子**，单+双激发已穷尽该自旋的
全部 15 个行列式 → α/β 笛卡尔积 15×15 = 225 = **全 FCI 空间**。这不是
"CISD 精确"（真 CISD 误差 +1.34e-5），而是"单双激发 = 全空间"的巧合。
对 >2 占据/自旋的体系（N₂ 7e/spin），单双激发覆盖不足，include(S+D) 误差
~2.25e-2——这是**经典单双激发的真机可行性边界**，也是量子采样价值重新变大的
强关联区。

### 4.4 方向 A：UCJ 辅助配置补充（强关联突破）

**动机**：经典单双激发对强关联（N₂ 三键）覆盖不足。UCJ 电路采样产生**超出
单双激发的高激发 det**，与 S+D 合并后 SQD 覆盖强关联。

**稳定性工程**（对比中发现并解决）：
1. **SVD 符号规范化**：CCSD 浮点噪声 → SVD 的 U/V 列符号歧义 → kappa 方向
   翻转 → 电路变化。修复：U 列最大元为正 + V 同步翻转。
2. **多 scale 合并**（3,5,10,20）：强关联体系近简并轨道使 t2（进而 kappa）对
   轨道基敏感，单 scale 跨进程误差在 1e-3~2e-2 波动。多 scale 保证高激发 det
   总被覆盖。
3. **独立随机旋转源**（`n_random=2`）：ry-cnot-ry 数保持 gadget 的随机轨道
   旋转，对近简并轨道方向鲁棒。

**结果**：N₂ 拉伸 UCJ-SQD 3.1e-3（化学精度），经典单参考全崩
（CCSD/CCSD(T)/CISD 均 0.14-0.20 Ha）。**方向 A 是"少量采样 shots 达化学
精度"路线**——量子资源省，但需要采样电路覆盖准简并基态。

### 4.5 方向 B：CIPSI 迭代精化（高精度 refine 层）

**动机**：化学精度之上如需 FCI 级，用 PT2-CIPSI 从 UCJ 种子出发自动补全子空间。

**算法**（`solve_cipsi`）：每轮 ① 子空间对角化 → ② 取 |c|>ε 主导 det 枚举
单/双激发连接 → ③ 扩展空间上 `contract_2e` 一次得 $\langle a|H|\Psi\rangle$
→ ④ 按 $PT2_a = |\langle a|H|\Psi\rangle|^2 / (E_{gs}-E_a)$ 加入。

**关键观察**：UCJ 种子字符串已覆盖全空间大部分（N₂ 89-120/120，C₂ 133/210），
CIPSI 单双激发闭包**一轮补全到全空间 = FCI**。代价是 det 规模 = 全空间
（与 HCI 近全空间相当）。**方向 B 是"高精度 refine 层"，不是"少量 det"路线**。

### 4.6 迭代法的准简并陷阱（C₂ 案例，库 bug 修复）

C₂/STO-3G 双 π 准简并（基态 -74.690041 vs 第二根 -74.639599，二重简并，差
0.0504 Ha）暴露两类迭代法缺陷：

1. **FCI 基准假收敛**：`direct_spin1.kernel` 默认 `conv_tol=1e-10` 的 Davidson
   在 Ritz 值稳定但残差仍 ~6e-6 时判定收敛，假收敛到第二根（基准虚高 0.0504 Ha）。
   修复：默认 `conv_tol=1e-12, max_cycle=1000`。
2. **solve_sci 准简并陷阱**：即便基准正确、基态主导 det 全在子空间，
   Davidson 仍从部分初始向量收敛到第二根。修复：改用 scipy `eigsh`（ARPACK SA）。

**教训**：对近简并体系，迭代对角化的"收敛"标志不可轻信；`eigsh`/显式 eigh
提供更稳健的基态。这也是 §2.3 变分下界在数值上的保障：合法的子空间对角化
**不可能**给出低于 FCI 的能量，一旦出现必是求解器或基准 bug。

---

## 5. 对比汇总

### 5.1 经典方法对比（STO-3G，误差 vs FCI，Ha）

| 分子 | 关联强度 | CCSD | CCSD(T) | CISD | **UCJ-SQD** | **solve_cipsi** |
|---|---|---|---|---|---|---|
| LiH | 弱 | 1.1e-5 | 2.1e-6 | 1.3e-5 | **3.1e-12** | = FCI |
| H₂O | 弱-中 | 1.2e-4 | 5.0e-5 | 7.2e-4 | **2.8e-14** | — |
| BeH₂ | 中 | 4.1e-4 | 1.9e-4 | 8.0e-4 | **2.3e-7** | — |
| N₂ 平衡 | 中 | 3.9e-3 | 2.2e-3 | 1.2e-2 | **2.1e-5** | — |
| N₂ 拉伸 | 强 | 1.4e-1 | 1.4e-1 | 2.0e-1 | **3.1e-3** | **< 1e-4**（S+D 种子）|
| C₂ 平衡 | 强（准简并）| 3.5e-2 | 4.8e-2 | — | **3.5e-5~1.3e-3** | **= FCI** |

**结论**：弱关联区经典单参考足够；强关联区经典单参考崩溃（0.1-0.2 Ha），
UCJ-SQD 以化学精度保持（比 CCSD(T) 好 ~40-45×）；CIPSI 从种子出发补全到 FCI。

### 5.2 HCI 对比与口径澄清（重要）

| 方法 | "det 数" | 含义 | 误差 |
|---|---|---|---|
| HCI（单双闭包）| 2,116 | 具体 det 集合（= 对角化空间）| 2.25e-2（平台）|
| HCI（近全空间）| 9,604 | 具体 det 集合 | 5.5e-8（= FCI）|
| UCJ-SQD | 765-1,339 | **采样 det 数**（bsm 行，shots 成本）| 1.16e-3 |
| UCJ-SQD（对角化维度）| ~7,900-14,400 | **字符串乘积**（N₂ 89-120²）| 1.16e-3 |

**⚠️ 口径不可直接比大小**：HCI 的"det 数"是对角化空间（具体 det 集合）；
UCJ-SQD 的 765-1,339 是**采样成本**（shots），其实际对角化维度（字符串乘积）
反而更大。**UCJ-SQD 的真实优势是采样效率**（少量 shots 达化学精度、量子资源
省），而非对角化空间小。方向 B（CIPSI）的 det 规模 = 全空间，与 HCI 近全空间
相当，走的是"高精度代价换空间"路线。

### 5.3 三种路线总结

| 路线 | 子空间来源 | 精度 | det 规模 | 量子资源 |
|---|---|---|---|---|
| UCJ-SQD（方向 A）| 采样 + S+D | 化学精度 | 中（采样成本低）| **低** |
| include(S+D) | 经典生成 | = FCI（仅 ≤2 occ/spin）| 小 | 极低 |
| CIPSI（方向 B）| 种子 + PT2 扩展 | = FCI | **全空间** | 低（种子采样）+ 经典迭代 |
| HCI | 确定性选态 | → FCI | 大 | 无 |

---

## 6. 理论局限与开放问题

1. **强关联 + 准简并的覆盖问题**：C₂ 修复后 UCJ-SQD 8/8 化学精度，但依赖
   采样电路覆盖准简并基态。提高 shots（3000→8000）**不**改变覆盖率——问题在
   电路设计而非采样量。开放：如何设计对近简并轨道鲁棒的采样电路。
2. **CIPSI 的空间代价**：方向 B 补全到全空间，失去"少量 det"优势。开放：
   能否以受限生成空间（二阶微扰严格筛选）在 ~2000-3000 dets 达 ~1e-5。
3. **字符串乘积表示 vs 具体 det 集合**：SQD 库的对角化维度是字符串乘积，
   与经典选态方法的 det 集合口径不同，对比需澄清。开放：是否支持具体 det 对
   集合表示以精确控制维度。
4. **frozen-core 近似**：冻结 core-valence 关联 ~2e-4 Ha（LiH）。开放：全活性
   空间或嵌入方法（如与 Vayesta 共存）。
5. **UCJ 的 ffsim 级精确化**：当前 J 对角为启发式（非 ffsim `UCJOpSpinBalanced`
   精确）。开放：完整 J + 多参数 orbital rotation。
6. **自旋分辨哈密顿量**：UHF 式 h_alpha ≠ h_beta 需 spin-orbital SQD 后端，
   当前显式拒绝。
7. **迭代法的准简并稳健性**：C₂ 教训表明 Davidson 的收敛标志在近简并下不可
   轻信。开放：自动化准简并检测与稳健对角化策略的推广。

---

*本文随仓库演进更新。数值见 `REVIEW.md`，使用见 `README.md`。*
