# 协作状态（M0 维护）

> 更新：2026-08-11 ｜ 当前轮：round_000（筹备，尚未开题）｜ 阶段：空闲
> 本文件由 M0 在每轮复盘后更新，**其余角色不得修改**。compact 后先读本文件。

## 当前轮

- 主题：（无，等待 M0 从 TOPIC_POOL 选题）
- 目标体系：（无）
- 各阶段状态：— （协议刚建立，第一轮尚未启动）
- 进行中产物：无

## 已完成轮次（近 5 轮）

| 轮 | 主题 | 结论状态 | 关键数据（一行） |
|---|---|---|---|
| round_000 | 协作协议搭建（COLLABORATION.md v1.0 + STATE + TOPIC_POOL）| 已完成 | 5 角色门控流水线 + 留痕/恢复机制就位 |

## 下一步行动（按优先级）

- [ ] M0：从 TOPIC_POOL.md 挑选 round_001 主题（优先 12,12 缺口相关），spawn R1
- [ ] M0：确认协作协议被各角色正确消费（首轮跑通后复盘微调）

## 关键决策记录

- 2026-08-11：确立 SQUAD 协议 v1.0 —— 5 角色（R1 调研/R2 理论/R3 实现/R4 审查/R5 跑分）
  + M0 协调，门控流水线，每轮 1 主题，产物按固定模板落盘 rounds/round_N/。
  北极星：12,12 强关联大空间 err ≤3× SHCI 同维度 + wall ≤3× SHCI + shots ≤500，
  其余体系不回归。目标基线数据见 COLLABORATION.md §0。

## 待 R3 修复 / 待 R5 验证的遗留项

- 无（首轮未启动）

## 环境备忘（摘自 HANDOFF，供各角色快速参考）

- WSL：Ubuntu-22.04；python = `/home/lenovo/miniconda/envs/tc/bin/python`（3.10/pyscf 2.14/numpy 2.2.6/tc 0.12）
- 仓库：`/mnt/d/tc_sqd`；测试规格：`/mnt/d/tc-sqd-test-spec/TEST_SPEC.md`
- 测试：`cd /mnt/d/tc_sqd && PYTHONPATH=src python -m pytest tests/ -q -rf`（≥122 通过）
- GPU：RTX 5080 + cupy-cuda12x（勿用 cuda13x）；`tc_sqd.noise.has_gpu()` 检测
- 提交：`git -c core.hooksPath=/dev/null commit -m "round<N>-<role>: ..."`（绕过 Mimosa 钩子）
- 长任务后台跑（12,12 整图 ~5.7h），勿前台阻塞
