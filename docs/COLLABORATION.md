# tc_sqd 多 Agent 协作协议（SQUAD 协议 v1.0）

> 本文件是 tc_sqd 仓库多 agent 长程协作的**唯一权威流程文档**。任何 agent 被
> spawn 时，其 prompt 只需引用本文件 + `docs/rounds/STATE.md` 即可独立上岗。
> 本文档与 STATE.md **自包含**（含环境、命令、格式模板），不依赖会话上下文。
>
> 配套文件：
> - `docs/rounds/STATE.md` —— 当前轮次状态（**compact 恢复的唯一入口**）
> - `docs/rounds/TOPIC_POOL.md` —— 主题池（R1 产出，M0 管理）
> - `D:\tc-sqd-test-spec\TEST_SPEC.md` —— 测试规格（A-G 类，权威验收标准）
> - `REVIEW.md` —— 跨轮验证历史（只追加，不改写）
> - `docs/HANDOFF_gpu_obdf.md` —— 交接文档范例（含环境血泪史）

---

## 0. 北极星目标（从 fig 各 commit 结论提炼）

### 0.1 现状基线（2026-08-11，REVIEW/TEST_SPEC 实测）

| 体系 | 全空间 | SQD 现状 | SHCI 现状 | 差距 |
|---|---|---|---|---|
| C₂/STO-3G | 44,100 | improved SQD 高维反超（交叉 ~dim 15000）| 低维优 | SQD 胜 |
| N₂/STO-3G | 14,400 | 趋同，SHCI 略优 | 略优 | 平手 |
| N₂/cc-pVDZ (10o) R=3.0 | 63,504 | 中高维反超 ~90×；baseline err 9.76e-7 @dim 47k；373s@100shots | err ~2e-4 全空间 205s | SQD 胜（精度） |
| C₂/cc-pVDZ (10o) | 63,504 | 双交叉（2-3×10⁴ 区间优）| 两端优 | 平手 |
| **N₂/cc-pVDZ (12,12) R=3.0** | **853,776** | **err 2.28e-4 @dim 1e5（远未收敛），1512s@100shots** | **err ~1e-10 级（eps=1e-3），812s** | **SHCI 完胜（主攻目标）** |

shots 现状：active ~100 shots 达化学精度（低采样端已是优势，不再苛求）。

### 0.2 北极星指标（量化验收）

> 对**每个新方法/改进**，最终以 R5 在 TEST_SPEC 体系矩阵上的实测对照验收：

1. **精度**：误差 vs 活性全空间 FCI 真基态（**禁用 CASCI 参考**，跳根陷阱）。
   - 底线：化学精度 1.6 mHa；
   - 比肩：与 SHCI **同子空间维度**下误差同量级（≤3×）。
2. **速度**：总 wall-time ≤ 3× SHCI 同任务（当前 best SQD ≈1.8-1.9× SHCI，保持即可）。
3. **shots**：≤ 500（active 100 达化学精度是现状标杆，新方法不得显著倒退）。
4. **主攻缺口**：12,12 强关联大空间——把 err 2.28e-4 压到 ≤3× SHCI 同维度误差，
   同时 wall ≤ 3× SHCI、shots ≤ 500。**其余体系不得回归**（回归即失败）。
5. 每轮改进**必须带可复现数据**（固定 seed、断点缓存、新旧对照），禁止"感觉有提升"。

### 0.3 目标来源（fig 相关 commit 语义）

- `630c143/c961fab`：12,12 85 万维能力边界 + 参考口径差异发现（三方 0.1-0.5 mHa）
- `11b399b`：C₂/cc-pVDZ 双交叉——方法优劣体系依赖，须多体系验证
- `7031c34/150a709/74f3a1d/ffd0b9d`：improved SQD (active+PT2) 为当前最强 SQD 变体
- `365030f`：减误差（维度轴）/ 提速（shots 轴）双轴口径，报告须分轴陈述
- `bce3716/9abe11a`：库内 HCI/SHCI 实现 = 对照基线，任何改进都与之比

---

## 1. 角色与职责

| 角色 | 代号 | 任务 | 主要工具 | 产物（写入） |
|---|---|---|---|---|
| 协调者（主 agent） | **M0** | 调度 5 角色、维护 STATE/TOPIC_POOL、复盘、决策主题、汇总结论进 REVIEW.md | Agent spawn、TodoWrite | `rounds/round_N/summary.md`、`STATE.md`、`TOPIC_POOL.md` |
| 前沿调研 | **R1** | 网络搜索 SQD 前沿 → 汇总 → 指导价值评分 → 候选主题 | WebSearch/WebFetch（arXiv HTML 优先）| `round_N/research.md` |
| 理论适配 | **R2** | 理解 R1 前沿 → 推导适配 tc_sqd 的理论 + 可证伪预测 + 实现指导 | Read（SURVEY/REVIEW/源码）| `round_N/theory.md` |
| 实现 | **R3** | 按 R2 指导写代码（含测试），风格对齐库 | Write/Edit/Bash | `round_N/implementation.md` |
| 审查+测试 | **R4** | 静态 review + 跑通全库 tests；修 bug；反馈 R3 | Bash（pytest）、Read | `round_N/review.md` |
| 跑分 | **R5** | TEST_SPEC B/C/D/E/F/G 类（A 类归 R4）+ 特定点优劣测试 | Bash（后台长任务）、plot 脚本 | `round_N/benchmark.md` + `TEST_REPORT.md` |

**角色可以复用**：M0 可兼任；一个 agent 实例负责一个角色一次 spawn，**不要**让
一个 agent 同时扮演多个角色（视角污染、长程不稳定）。

---

## 2. 每轮工作流（门控流水线）

```
        ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  M0 ──▶│  R1  │──▶│  R2  │──▶│  R3  │──▶│  R4  │──▶│  R5  │──▶ M0 复盘 ──▶ 下一轮
 选题   │调研  │   │理论  │   │实现  │   │审查  │   │跑分  │   │  总结/决策  │
        └──────┘    └──────┘    └──────┘    └──────┘    └──────┘
          ↑                                 │              │
          └───────── 证伪/失败回退 ─────────┘              ▼
        （R4 拒收或 R5 证伪 → 记录后回主题池，不恋战）   结论供全员阅读
```

### 2.1 阶段定义与门控

| 阶段 | 输入 | 输出 | 门控（不满足不得进入下一阶段） |
|---|---|---|---|
| **选题**（M0） | TOPIC_POOL + STATE | 选定主题、目标体系 | 每轮**恰好 1 个主题**；优先攻击 12,12 缺口 |
| **R1** | 主题方向 | `research.md` | 每条候选含指导价值评分（高/中/低）+ 理由；top3 排序 |
| **R2** | `research.md` | `theory.md` | 含 ≥1 条**可证伪定量预测** + 改动模块清单 + P0/P1/P2 优先级 |
| **R3** | `theory.md` | 代码 + `implementation.md` | 代码可运行；新功能带测试；自测通过；无僵尸代码 |
| **R4** | R3 交付 | `review.md` | 全库 pytest **0 失败**（已知抖动项单跑验证）；blocker 清零 |
| **R5** | R4 通过的代码 | `benchmark.md` + `TEST_REPORT.md` | 与 R2 预测逐条对照（证实/证伪/部分）；新方法曲线与旧方法同图对照 |
| **复盘**（M0） | 全部产物 | `summary.md`、STATE 更新、REVIEW 追加 | 结论状态三选一：**已验证 / 已证伪 / 需重试**；决策下轮主题 |

### 2.2 失败处理（快速失败原则）

- **R3 实现失败**（>时间盒 2× 仍未通）：记录阻碍点 → 回退 R2 修订理论 → 再试一次；
  仍失败则判"证伪/不可行"，主题回池，**不留半成品代码**（删除或 `git revert`）。
- **R4 拒收**（blocker）：bug 清单回 R3 修复 → R4 复验，最多 2 轮；仍不过则该方向搁置。
- **R5 证伪**（预测不达标）：**不是失败，是有价值结论**。如实记录数据与差距分析，
  该方向标记"证伪（数据支持）"，禁止悄悄改写 R2 预测使其"看起来通过"。
- 任何阶段超时：记录进度 → M0 决定续跑或暂停。**长任务（12,12 整图 ~5.7h）必须
  后台运行，绝不前台阻塞。**

### 2.3 时间盒建议（每轮总量半天到一天工作量）

R1 ≤ 2h ｜ R2 ≤ 2h ｜ R3 ≤ 4h ｜ R4 ≤ 2h ｜ R5 ≤ 4h（跑分可后台并行）。

---

## 3. 接口契约（各角色产物格式模板）

> 产物文件都写到 `docs/rounds/round_<NNN>/`，文件名固定。**格式即契约**，
> 下游 agent 只按模板字段消费。

### 3.1 `research.md`（R1 → M0/R2）

```markdown
# Round N 调研：<主题>
> 日期 / 搜索范围（arXiv 年份、关键词）/ 已落地对照（读 SURVEY §7/§8.9 避免重复）

## 候选清单
| # | 名称 | arXiv id | 核心机制（≤5 句） | 与已落地方向的差异 | 落地依赖 | 预期收益维度(精度/速度/shots) | 价值评分 | 风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... | ... | 高/中/低 | ... |

## 推荐排序（top 3）
1. **<名称>** —— 为什么有实际指导价值（对 12,12 缺口 or 其余体系）……
2. ...
3. ...

## 明确排除
- <名称>：理由（已落地/不适用分子/依赖太重）—— 防重复调研
```

价值评分标准：**高** = 分子可落地 + 直击 12,12 缺口或明显提速/降 shots + 依赖轻；
**中** = 有理论价值但落地不确定性高；**低** = 仅背景参考。

### 3.2 `theory.md`（R2 → R3）

```markdown
# Round N 理论：<方法名>
## 1. 理论推导
   （数学公式、与本库 SQD 框架的接合点；引用 SURVEY/REVIEW 已有结论）
## 2. 适配 tc_sqd 设计
   - 改动模块：<文件清单>
   - 新 API 建议：签名 + 默认值 + docstring 草稿
   - 参数选择依据（引 REVIEW 实测数值）
## 3. 可证伪预测（P0 必须 ≥1 条）
   | # | 体系 | 条件 | 预测 | 验收阈值 |
   |---|---|---|---|---|
   | P1 | N₂/cc-pVDZ (12,12) | shots=500, max_strings=... | err ≤ 3× SHCI 同维度 | 实测 err < ... |
## 4. 实现优先级：P0（必须）/ P1（建议）/ P2（可选）
## 5. 风险与回退方案
```

**可证伪预测是 R2 的核心产出**：数值必须具体（体系、shots、维度、误差阈值），
否则 R5 无法验收。预测基准一律引用 TEST_SPEC §6.5 的 baseline 表。

### 3.3 `implementation.md`（R3 → R4）

```markdown
# Round N 实现
## 改动文件清单
| 文件 | 改动摘要 | 行数变化 |
|---|---|---|
## 新 API 签名（粘贴 docstring 头部）
## 新增测试
| 测试文件 | 测试函数 | 断言 |
## 自测结果（命令 + 输出摘要）
## 已知限制 / 未完成项（诚实列出）
```

### 3.4 `review.md`（R4 → M0/R3，bug 格式）

```markdown
# Round N 审查
## 静态 review 发现
| # | 文件:行 | 严重度(blocker/major/minor) | 问题 | 修复建议 |
## 动态测试
- 全库 pytest：<通过数>/<总数>，失败 <n>
- 新功能测试：<通过数>/<总数>
- 已知抖动项：单跑结果
## Bug 反馈（blocker/major → R3；minor 可自行修复并注明）
| # | 复现命令 | 根因 | 修复 |
## 结论：放行 R5 / 需修复后复验
```

### 3.5 `benchmark.md`（R5 → 全员）

```markdown
# Round N 跑分
## 与 R2 预测对照
| 预测 # | 实测值 | 判定(证实/证伪/部分) | 差距分析 |
## B 类误差-维度表（新方法 vs SHCI vs 旧 improved SQD）
| 体系 | dim | SQD err(±std) | SHCI err | 比值 |
## C 类速度表
| 体系 | 方法 | wall | 峰值内存 | 相对 SHCI |
## F 类适应度表（方法 × 体系）
## 结论（供全员）
- 新方法是否达标北极星？哪一维达标哪一维不达标？
- 对下一轮的建议主题
```

同时按 TEST_SPEC §8 追加到 `D:\tc-sqd-test-spec\TEST_REPORT.md`（追加节，不覆盖历史）。

### 3.6 `summary.md`（M0 复盘）

```markdown
# Round N 复盘
## 本轮成效（一句话 + 关键数据）
## 结论状态：已验证 / 已证伪 / 需重试
## 对 REVIEW.md 的追加（M0 执行，只追加不改写）
## 下轮主题决策 + 理由
## 遗留事项（谁、什么、何时）
```

---

## 4. 工作留痕规范

### 4.1 目录与文件

```
docs/
├── COLLABORATION.md        # 本文件（协议，不可轻易改动；改需 M0 批准）
├── HANDOFF_gpu_obdf.md     # 交接文档（范例）
├── solve_sqd_api.md
└── rounds/
    ├── STATE.md            # 当前状态（compact 恢复唯一入口，M0 每轮更新）
    ├── TOPIC_POOL.md       # 主题池（候选 + 状态：待选/进行中/已落地/已证伪/搁置）
    └── round_001/
        ├── research.md  ├── theory.md  ├── implementation.md
        ├── review.md    ├── benchmark.md  └── summary.md
    └── round_002/ ...
```

### 4.2 git 提交规范（防止中途丢失，核心留痕手段）

- **每个阶段产物完成后立即 commit**（粒度：一次产出 = 一个 commit）。
- message 格式：`round<N>-<role>: <简述>`，如 `round7-research: 调研 ARNN 采样器（排除）`。
- 提交命令（Mimosa 钩子强制拦截，用下面命令绕过，警告均为 third_party 既有问题）：
  ```bash
  git -c core.hooksPath=/dev/null add -A
  git -c core.hooksPath=/dev/null commit -m "round<N>-<role>: ..."
  ```
- **禁止 push**（除非用户明确要求）。
- 中途 compact 丢失上下文时，git log 就是恢复路径。

### 4.3 数据与文件纪律（防污染）

1. **缓存一律保留**：`_plot_data_*.npy` / `_*_ints.npz`（--plot 秒出与积分复用依赖），
   删除需 M0 明确批准；重建代价分钟级到数小时。
2. **方法演进另写脚本**：新方法对照**新建** plot 脚本 + 新缓存名（`_plot_data_<new>.npy`），
   **禁止覆盖**旧脚本/旧图/旧缓存（TEST_SPEC §3.5 硬性规定）。
3. **固定 seed**：默认 0；多 seed 用 3（低采样端涨落），高采样端单 seed。
4. **参考口径**：一律用库内全空间对角化真基态；**禁用 CASCI**（近简并跳根，C₂/cc-pVDZ
   差 9.3 mHa 前科）；85 万维参考口径差异 0.1-0.5 mHa 属预期，如实记录不算失败。
5. **结论三态标记**：已验证 / 已证伪 / 需重试——文档中每个结论必须落一个状态。
6. REVIEW.md **只追加不改写**；旧结论修正需"追加修正节 + 引用新数据"，由 M0 执行。

---

## 5. 进度记忆与 compact 恢复协议

### 5.1 恢复入口（compact 后第一步）

```
1. 读 docs/rounds/STATE.md          —— 权威状态（必读）
2. 读 docs/rounds/round_<当前N>/ 的产物 —— 按需
3. 读 docs/rounds/TOPIC_POOL.md     —— 下轮候选
4. 数值基线：TEST_SPEC.md（验收）+ REVIEW.md（历史）
5. 用户记忆（C:\Users\lenovo\.zcode\cli\memories\projects\tc_sqd-*/memory/）
   —— 跨会话关键决策指针
```

### 5.2 STATE.md 模板（M0 每轮复盘后更新，**其余角色不得改**）

```markdown
# 协作状态（M0 维护）
> 更新：<日期> ｜ 当前轮：round_<NNN> ｜ 阶段：<选题/R1/.../复盘/空闲>

## 当前轮
- 主题：<名称> ｜ 目标体系：<体系> ｜ 决策依据：<一行>
- 各阶段状态：[x] R1 调研 ｜ [x] R2 理论 ｜ [ ] R3 实现 ｜ [ ] R4 审查 ｜ [ ] R5 跑分
- 进行中产物：<最新文件的路径与关键内容摘要>

## 已完成轮次（近 5 轮）
| 轮 | 主题 | 结论状态 | 关键数据（一行） |
|---|---|---|---|

## 下一步行动（按优先级）
- [ ] <动作>（责任人，何时）

## 关键决策记录
- <日期>：<决策>（原因）

## 待 R3 修复 / 待 R5 验证的遗留项
- <谁、什么、状态>
```

### 5.3 记忆同步（M0 负责）

- 每轮复盘后，M0 把**跨会话才需要记住的决策**写入用户持久记忆目录
  （`.../memory/`，文件格式见记忆规范：name/description/type + 正文；更新 MEMORY.md 索引）。
- 记忆内容限于：协作协议要点、关键决策、环境陷阱——**不存**代码结构/测试结果
  （仓库文档已记录，避免重复）。
- 现有记忆参考：`wsl-conda-tc-env.md`（环境）。

---

## 6. 测试与跑分规范（对齐 TEST_SPEC v1.0）

### 6.1 分工

| 类 | 内容 | 执行者 |
|---|---|---|
| A | 单元测试正确性（pytest 全库）| **R4** |
| B | 误差基准（SQD vs SHCI 5 体系图）| **R5** |
| C | 速度/资源基准（benchmark_sqd.py）| **R5** |
| D | Dice 交叉验证 | **R5**（可选，需 PYSCF_EXT_PATH）|
| E | 诊断（跳根/口径/外推证伪/噪声免疫/预算）| **R5** |
| F | 改进方法跨体系适应度 | **R5** |
| G | GPU 加速 | **R4**（正确性）/ R5（性能，有 GPU 时）|

### 6.2 关键命令（WSL 内）

```bash
# 全库回归（R4）
cd /mnt/d/tc_sqd
/home/lenovo/miniconda/envs/tc/bin/python -m pytest tests/ -q -rf   # 验收 ≥122 通过 0 失败

# 单测试模块
PYTHONPATH=src python -m tests.test_h2_sqd

# B 类图（R5）
python examples/plot_improved_sqd_vs_shci_c2_ccpvdz.py           # 收集+出图（~1.5-2h，后台）
python examples/plot_improved_sqd_vs_shci_c2_ccpvdz.py --plot    # 缓存秒出图

# C 类基准
python benchmarks/benchmark_sqd.py --quick --out bench_quick.csv

# D 类前置
export PYSCF_EXT_PATH=/home/lenovo/shciscf    # 测完清理 FCIDUMP/input.dat/output.dat

# 提交（绕过 Mimosa 钩子）
git -c core.hooksPath=/dev/null commit -m "..."
```

### 6.3 已知抖动项（全库失败时先单跑验证，不算回归）

- `tests/test_basis.py::test_ccsd_no_increases_sparsity`（CCSD 收敛波动）
- `tests/test_noise.py::test_solve_sqd_robust_combines_zne_budget`（统计抖动）

### 6.4 新功能测试硬性要求（R3 遵守，R4 验收）

- 新 API 必须配测试：**正确性锚点**（对 FCI/解析值断言）+ **反例/边界**（非法输入、
  已知失败场景锁认知，参照 `test_include_excitations_not_fci_strong_correlation` 风格）。
- 修改核心模块（fermion/cipsi/configuration_recovery）必须过全库回归。
- GPU 相关：无 cupy 环境 skip/回退 numpy，不允许因缺 GPU 失败。

---

## 7. 环境速查（WSL）

| 项 | 值 |
|---|---|
| 运行时 | WSL Ubuntu-22.04 + conda env `tc`（`~/miniconda`，**不是** miniconda3）|
| Python | `/home/lenovo/miniconda/envs/tc/bin/python`（3.10 / pyscf 2.14 / numpy 2.2.6 / scipy 1.15.3 / tensorcircuit 0.12.0）|
| 仓库（WSL）| `/mnt/d/tc_sqd`（Windows `D:\tc_sqd`）|
| 测试规格 | `/mnt/d/tc-sqd-test-spec/TEST_SPEC.md`；报告写入 `TEST_REPORT.md` |
| GPU | RTX 5080 Laptop 16GB + `cupy-cuda12x 14.1.1`（**勿用 cupy-cuda13x**，占位 wheel 缺 libcurand）；`tc_sqd.noise.has_gpu()` 检测 |
| Dice/SHCI | `/home/lenovo/Dice/Dice-master/bin/Dice` + `PYSCF_EXT_PATH=/home/lenovo/shciscf` |
| 代理 | WSL NAT：Windows Clash 开 Allow LAN，走 gateway（如 `http://172.29.128.1:7897`）|
| Windows 侧执行 | `wsl -d Ubuntu-22.04 -e <python> <script>`；PowerShell 需 `$env:MSYS2_ARG_CONV_EXCL='*'` |

更多坑（cupy 安装血泪史、Vayesta 共存、numpy 兼容补丁）见 `docs/HANDOFF_gpu_obdf.md` §0 与 `README.md` 安装节。

---

## 8. 安全边界与禁忌（全员遵守）

1. **不覆盖**：旧 plot 脚本 / 旧图 / 旧缓存 / 旧测试文件 / REVIEW.md 历史节。
2. **不删**：`_plot_data_*.npy`、`_*_ints.npz`（--plot 与积分复用依赖）。
3. **不改**：`docs/COLLABORATION.md` 协议本身（修订需 M0 批准 + 版本号递增）。
4. **不 push**；commit 用 §4.2 命令。
5. **不静默改结论**：数值与文档不符时，先复现定位（µHa 偏移前科：先查参考口径，
   再怀疑算法）；确为 bug 则修 + 测试锁定 + 记录。
6. **不留僵尸代码**：失败的实现要么修通要么删除/revert。
7. **不引入重依赖**：新方法优先 numpy/scipy/pyscf/tensorcircuit 生态内实现；
   需新包必须 M0 批准（参考 `pyhci`/`pyscf-forge` 安装失败前科）。
8. **文档语言**：仓库文档一律中文；代码注释/docstring 跟随库内风格。

---

## 9. 轮次节奏与主题池管理

- **每轮 1 个主题**，主题粒度 = "一个可验证的方法改进"（如：半随机 PT2 降枚举成本、
  Krylov 采样电路、基设计扩展、semistochastic 采样等），不是"探索性研究"。
- 主题来源：TOPIC_POOL.md（R1 调研产出 + REVIEW「后续可选改进」清单 +
  SURVEY §7.3/§8.9 未落地方向）。
- 选题优先级（M0 决策）：
  1. 直击 12,12 缺口（采样覆盖 / 大空间对角化效率）；
  2. 普适降 shots 或提速且不损精度；
  3. 弱/中关联体系的精度精修；
  4. 其余（含探索性）。
- **轮间衔接**：R5 结论是下一轮 R1 的输入（回填 TOPIC_POOL：已验证→归档；
  证伪→标记防重复；"部分"→精化主题再入池）。
- 建议每 5 轮做一次**全局复盘**：把关键新结论追加进 REVIEW.md，把过时 memory
  更新/删除，压缩 TOPIC_POOL。

---

## 10. 附录：agent spawn 提示词模板（M0 使用）

```
你是 tc_sqd 协作的 <角色名>（<代号>）。先读 D:\tc_sqd\docs\COLLABORATION.md 的
<角色章节号> 和 D:\tc_sqd\docs\rounds\STATE.md（当前轮：<round_N>，主题：<主题>），
再读对应轮次产物（round_N/ 下已有文件）。严格按协议模板输出到
D:\tc_sqd\docs\rounds\round_<N>\<产物文件名>，完成后 git commit
（git -c core.hooksPath=/dev/null commit -m "round<N>-<role>: <简述>"）。
环境：WSL（python 在 /home/lenovo/miniconda/envs/tc/bin/python，仓库 /mnt/d/tc_sqd）。
你的任务：<角色任务一句话>。禁止：<该角色的禁忌>。
```

---

*协议版本 v1.0（2026-08-11）。修订记录见 git log（`docs/COLLABORATION.md` 路径）。*
