# tc_sqd

**Sample-based Quantum Diagonalization for TensorCircuit**

一个适配 TensorCircuit + numpy 1.x/2.x + PySCF 的轻量级 SQD 包，参考 `qiskit-addon-sqd` 设计，但不依赖 numpy>=2 / jax。

## 目录结构

```
tc_sqd/
├── README.md                 # 本文件
├── requirements.txt          # 依赖列表
├── src/                      # 包源码
│   └── tc_sqd/
│       ├── __init__.py       # 统一导出
│       ├── counts.py         # 比特串↔整数互转、TC采样适配
│       ├── configuration_recovery.py  # 平均占据数配置恢复
│       ├── subsampling.py     # 批量子采样、汉明权重后选择
│       ├── fermion.py        # CI矩阵、SQD对角化、轨道优化、基态能量
│       └── qubit.py          # Pauli哈密顿量子空间投影
├── tests/                    # 测试
│   ├── __init__.py
│   └── test_h2_sqd.py       # H2全流程 + TFIM + 基态能量测试
├── examples/                 # 示例
│   └── h2_sqd_demo.py       # H2 SQD 完整演示
└── docs/                     # 文档
    ├── README.md            # 详细 API 说明
    ├── API.md               # API 速查表
    └── usage.md             # 简要用法介绍
```

## 安装

```bash
conda create -n tc python=3.10
conda activate tc
pip install -r requirements.txt
```

## 快速开始

```bash
# 运行测试
PYTHONPATH=src python -m tests.test_h2_sqd

# 运行示例
PYTHONPATH=src python examples/h2_sqd_demo.py
```

## 核心用法

```python
import tc_sqd

# 从哈密顿量积分计算基态能量（三种方法）
e = tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore, method="fci")
e = tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore, method="direct")
e = tc_sqd.compute_ground_state_energy(
    h1e, eri, norb, nelec, ecore=ecore, method="sqd",
    bitstring_matrix=bsm, probabilities=probs,
)
```

详见 `docs/README.md`。
