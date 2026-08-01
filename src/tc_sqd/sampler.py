"""tc_sqd.sampler —— 统一采样后端 (模拟器先行, 真机后验)。

把 TensorCircuit 无噪声模拟与腾讯真机采样统一成单一入口 :func:`sample`,
下游 SQD 流水线 (recover -> diagonalise -> energy) 无需感知后端差异。
呼应"真机不能频繁用"的约束: 开发/调试用 ``backend="tc"``, 交付用
``backend="qcloud"``。

.. code-block:: python

    bsm, probs = tc_sqd.sample(circ, 3000, backend="tc")           # 模拟器
    bsm, probs = tc_sqd.sample(circ, 8192, backend="qcloud",       # 真机
                               backend_kwargs={"device_name": "59Q",
                                               "physical_qubits": pq})
    e = tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec,
                                           ecore=ecore, method="sqd",
                                           bitstring_matrix=bsm,
                                           probabilities=probs)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["sample", "BACKENDS"]

BACKENDS = ("tc", "qcloud")


def sample(circuit, n_samples: int, *, backend: str = "tc",
           backend_kwargs: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """统一采样入口。

    Parameters
    ----------
    circuit : tensorcircuit.Circuit
        待采样电路 (HF / LUCJ / 任意 TC 电路)。
    n_samples : int
        采样数 (shots)。
    backend : {"tc", "qcloud"}
        ``"tc"``      —— TC 无噪声模拟采样 (默认; 开发/调试/教学)。
        ``"qcloud"``  —— 腾讯真机采样 (需白名单机)。``backend_kwargs`` 必含
        ``device_name``; 可选 ``physical_qubits`` / ``enable_rem`` /
        ``h1e`` / ``eri`` / ``norb`` / ``nelec`` / ``ecore`` / ``e_hf_ref``
        (字节序自校准) 等, 全部透传给 :func:`tc_sqd.hardware.sample_on_hw`。
    backend_kwargs : dict | None
        后端专属参数。

    Returns
    -------
    (bitstring_matrix, probabilities) : (ndarray (S, 2n), ndarray (S,))
        统一格式, 直接喂 ``recover_configurations`` /
        ``compute_ground_state_energy``。

    Raises
    ------
    ValueError
        ``n_samples`` 非正、``backend`` 未知、``qcloud`` 缺 ``device_name``。
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    kw = dict(backend_kwargs or {})

    if backend == "tc":
        from .counts import sample_from_circuit
        return sample_from_circuit(circuit, n_samples=n_samples)

    if backend == "qcloud":
        device_name = kw.pop("device_name", None)
        if device_name is None:
            raise ValueError(
                "backend='qcloud' 需要 backend_kwargs['device_name']."
            )
        from .hardware import sample_on_hw
        res = sample_on_hw(device_name, circuit, shots=n_samples, **kw)
        return res["bsm"], res["probs"]

    raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}.")
