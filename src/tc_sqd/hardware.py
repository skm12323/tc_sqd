"""tc_sqd.hardware —— 腾讯 qcloud 真机 SQD 一站式工具。

整合 D:\\qubit_toolkit (select_qubits) + D:\\exp (真机 SQD 流水线) 到 tc_sqd,
使其成为腾讯真机 SQD 的全栈工具 (qiskit 生态没有针对腾讯硬件的整合):

  - ``load_calibration``    从 tc qcloud 设备读校准快照 (T1/T2/读出/CZ/拓扑)
  - ``select_qubits``       多起点贪心选最优 nq 物理qubit子图 (min(T2)+连通+读出/CZ)
  - ``bitstring_matrix_to_energy``  采样 bsm -> recover -> 子空间对角化 -> 基态能量 (SQD 后处理)
  - ``sample_on_hw``        tc qcloud 真机采样 (编译+submit_task+REM+字节序自校准)

典型真机流水线:
  cal = load_calibration(device)            # 读校准
  pq = select_qubits(cal, nq=2*norb)       # 选最优比特
  bsm = sample_on_hw(device, circuit, pq)  # 真机采样 (HF/LUCJ 电路)
  e = bitstring_matrix_to_energy(bsm, h1e, eri, norb, nelec, ecore)  # SQD 对角化

依赖: tensorcircuit (tc.cloud, 真机采样); select_qubits/load_calibration 纯 numpy + tc.cloud。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ===================== 校准加载 (从 tc qcloud 设备读) =====================
def load_calibration(device_name: str) -> dict:
    """从 tc qcloud 设备读校准快照, 返回 dict (拓扑 + 每 qubit 质量/相干 + 每 link CZ)。

    结构: {device, n_qubits, n_links, qubits: {id: {T1_us,T2_us,readout_f0,readout_f1,
    sq_err,sq_gate_ns,freq_mhz}}, edges: {"a-b": {cz_err,cz_gate_ns}}, topology: [[a,b],...]}
    """
    import tensorcircuit as tc
    d = tc.cloud.apis.get_device(device_name)
    props = d.list_properties()
    bits = props.get("bits", {})
    links = props.get("links", {})
    qubits = {}
    for q in bits:
        b = bits[q]
        qubits[int(q)] = {
            "T1_us": b.get("T1"), "T2_us": b.get("T2"),
            "readout_f0": b.get("ReadoutF0Err"), "readout_f1": b.get("ReadoutF1Err"),
            "sq_err": b.get("SingleQubitErrRate"),
            "sq_gate_ns": b.get("SingleQubitGateLenInNs"),
            "freq_mhz": b.get("Freqency"),
        }
    edges = {}
    for k, v in links.items():
        a = int(v.get("A", k[0] if isinstance(k, tuple) else 0))
        b = int(v.get("B", k[1] if isinstance(k, tuple) else 0))
        edges[f"{a}-{b}"] = {"cz_err": v.get("CZErrRate"), "cz_gate_ns": v.get("GateLenInNs")}
    topology = [list(map(int, e)) for e in d.topology()]
    return {"device": device_name, "n_qubits": len(qubits), "n_links": len(edges),
            "qubits": qubits, "edges": edges, "topology": topology}


# ===================== select_qubits (多起点贪心, 从 qubit_toolkit 整合) =====================
def _parse_cal(cal: dict):
    """校准 dict -> (T2, readout, cz_err, adj) numpy 结构。"""
    qubits = cal["qubits"]
    T2 = {int(q): float(v.get("T2_us") or 0) for q, v in qubits.items()}
    RO = {int(q): ((float(v.get("readout_f0") or 0) + float(v.get("readout_f1") or 0)) / 2)
          for q, v in qubits.items()}
    CZ = {}
    adj = {q: set() for q in T2}
    for k, v in cal["edges"].items():
        a, b = map(int, k.split("-"))
        cz = float(v.get("cz_err") or 0)
        CZ[(a, b)] = cz; CZ[(b, a)] = cz
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    return T2, RO, CZ, adj


def _greedy_from(start, nq, adj, T2, RO):
    S = {start}
    while len(S) < nq:
        cands = set()
        for q in S:
            cands |= adj.get(q, set())
        cands -= S
        if not cands:
            return None
        best = max(cands, key=lambda q: (T2.get(q, 0), -RO.get(q, 1)))
        S.add(best)
    return S


def _evaluate(S, T2, RO, CZ, adj, w_t2=1.0, w_ro=50.0, w_cz=200.0, w_edges=0.5):
    minT2 = min(T2[q] for q in S)
    meanRO = sum(RO[q] for q in S) / len(S)
    edges = sum(1 for q in S for q2 in adj.get(q, set()) if q2 in S) // 2
    meanCZ = (sum(CZ.get((q, q2), 0) for q in S for q2 in adj.get(q, set()) if q2 in S) / 2) / max(edges, 1)
    return w_t2 * minT2 - w_ro * meanRO - w_cz * meanCZ + w_edges * edges


def _bfs_order(S, adj, T2):
    from collections import deque
    start = max(S, key=lambda q: T2.get(q, 0))
    seen = {start}; P = [start]; dq = deque([start])
    while dq:
        x = dq.popleft()
        for y in sorted(adj.get(x, set()), key=lambda z: -T2.get(z, 0)):
            if y in S and y not in seen:
                seen.add(y); P.append(y); dq.append(y)
    for q in sorted(S - set(P), key=lambda z: -T2.get(z, 0)):
        P.append(q)
    return P


def select_qubits(calibration: dict, nq: int, *, n_starts: int = 15,
                  w_t2: float = 1.0, w_ro: float = 50.0, w_cz: float = 200.0,
                  w_edges: float = 0.5) -> List[int]:
    """从校准快照选最优 nq 物理qubit子图 (BFS 序映射, 相邻逻辑->相邻物理)。

    返回 PHYSICAL_QUBITS 列表 (长度 nq), 逻辑 q0..q_{nq-1} -> 物理 P[i]。
    评分: max(min(T2)) 为主 (木桶), 扣读出/CZ, 加相邻边数。
    """
    T2, RO, CZ, adj = _parse_cal(calibration)
    best = None
    for start in sorted(T2, key=lambda q: -T2[q])[:n_starts]:
        S = _greedy_from(start, nq, adj, T2, RO)
        if S is None:
            continue
        sc = _evaluate(S, T2, RO, CZ, adj, w_t2, w_ro, w_cz, w_edges)
        if best is None or sc > best[0]:
            best = (sc, S)
    if best is None:
        raise ValueError(f"无法找到 {nq}-连通子图 (设备太小/拓扑太碎)。")
    return _bfs_order(best[1], adj, T2)


# ===================== SQD 后处理 (bsm -> recover -> diag -> 能量) =====================
def bitstring_matrix_to_energy(bsm, h1e, eri, norb, nelec, ecore: float = 0.0,
                                probs=None, max_iterations: int = 5) -> float:
    """采样 bitstring matrix -> 配置恢复 + 子空间对角化 -> 基态能量 (含 ecore)。

    bsm 布局为 tc_sqd 约定的降序 [β_{n-1}..β0 | α_{n-1}..α0] (counts_dict_to_bitstring_matrix
    产生的格式)。直接复用 ``compute_ground_state_energy`` (它正确处理 bsm 布局)。
    """
    from .fermion import compute_ground_state_energy
    kwargs = dict(ecore=ecore, method="sqd",
                  bitstring_matrix=np.asarray(bsm, dtype=bool),
                  max_iterations=max_iterations)
    if probs is not None:
        kwargs["probabilities"] = probs
    return float(compute_ground_state_energy(h1e, eri, norb, nelec, **kwargs))


# ===================== 真机采样 (tc qcloud, 含 REM + 字节序自校准) =====================
def sample_on_hw(device_name, circuit, physical_qubits=None, shots: int = 8192,
                  h1e=None, eri=None, norb=None, nelec=None, ecore: float = 0.0,
                  enable_rem: bool = True, e_hf_ref: Optional[float] = None) -> dict:
    """tc qcloud 真机采样 -> (可选 REM + 字节序自校准) -> bitstring matrix + SQD 能量。

    circuit: tc.Circuit (HF/LUCJ)。physical_qubits: select_qubits 输出 (可选)。
    若给 h1e/eri/norb/nelec/ecore, 同时算 SQD 能量; e_hf_ref 给则自校准字节序。

    返回 dict: {counts, bsm, e_sqd (若给积分), reverse_key (字节序)}。
    需在白名单机跑 (tc qcloud 真机)。
    """
    import tensorcircuit as tc
    d = tc.cloud.apis.get_device(device_name)
    opts = {"coupling_map": d.topology()}
    if physical_qubits:
        opts["initial_layout"] = list(physical_qubits)
    c1, info = tc.compiler.default_compile(circuit, compiled_options=opts)
    t = tc.cloud.apis.submit_task(device=d, circuit=c1, shots=shots)
    counts = t.results()

    # REM (读出缓解)
    nq = 2 * norb if norb else circuit.num_qubits
    if enable_rem:
        try:
            mit = tc.results.rem.ReadoutMit(d.name + "?o=0")
            mit.cals_from_system(nq)
            counts = mit.apply_correction(counts, qubits=nq, **info)
        except Exception as ex:
            pass  # REM 失败用原始 counts

    # counts -> bsm (tc qcloud 字节序: 试两种, 若给 e_hf_ref 自校准)
    from .counts import counts_dict_to_bitstring_matrix
    bsm_a, probs_a = counts_dict_to_bitstring_matrix(counts, nq)
    if h1e is not None and e_hf_ref is not None and norb is not None:
        bsm_b, probs_b = counts_dict_to_bitstring_matrix(
            {k[::-1]: v for k, v in counts.items()}, nq)
        e_a = bitstring_matrix_to_energy(bsm_a, h1e, eri, norb, nelec, ecore, probs_a)
        e_b = bitstring_matrix_to_energy(bsm_b, h1e, eri, norb, nelec, ecore, probs_b)
        if abs(e_b - e_hf_ref) < abs(e_a - e_hf_ref):
            bsm_a, probs_a, e_a = bsm_b, probs_b, e_b
            reverse = True
        else:
            reverse = False
        return {"counts": counts, "bsm": bsm_a, "e_sqd": e_a, "reverse_key": reverse}

    e_sqd = None
    if h1e is not None and norb is not None:
        e_sqd = bitstring_matrix_to_energy(bsm_a, h1e, eri, norb, nelec, ecore, probs_a)
    return {"counts": counts, "bsm": bsm_a, "e_sqd": e_sqd, "reverse_key": False}
