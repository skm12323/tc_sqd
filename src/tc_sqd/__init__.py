"""tc_sqd --- Sample-based Quantum Diagonalization for TensorCircuit.

A lightweight SQD package adapted for TensorCircuit + numpy 1.x, inspired by
``qiskit-addon-sqd`` but free of the numpy>=2 / jax hard dependencies.

Modules
-------
- ``tc_sqd.counts``                 -- bitstring matrix <-> int helpers, TC sampling adapter
- ``tc_sqd.configuration_recovery`` -- average-occupancy configuration recovery
- ``tc_sqd.subsampling``            -- batch subsampling & Hamming-weight postselection
- ``tc_sqd.fermion``                -- CI matrix, SQD diagonalisation, orbital optimisation
- ``tc_sqd.qubit``                  -- qubit-subspace projection & diagonalisation
- ``tc_sqd.integrated``              -- one-call ``solve_sqd`` entry point (端到端: 含采样/迭代, 返回 SCIResult)
- ``tc_sqd.fermion`` ``compute_ground_state_energy`` -- 积分→能量快速单入口 (采样外部提供, 返回 float; 与 solve_sqd 分工见各 docstring / docs/solve_sqd_api.md §9)
"""

# numpy 2.x ↔ tensorcircuit 0.12 兼容补丁: 导入 tc_sqd 即自动 apply。
# 若脚本先 import tensorcircuit 再 import tc_sqd, 需调整顺序, 或运行
# `python -m tc_sqd._compat install` 写入 sitecustomize 一劳永逸。
from . import _compat  # noqa: F401

from .counts import (
    bitarray_to_int,
    int_to_bitarray,
    counts_dict_to_bitstring_matrix,
    sample_from_circuit,
)
from .configuration_recovery import (
    recover_configurations,
    recover_configurations_clustered,
    postselect_by_hamming_weight,
    estimate_true_occupancies,
)
from .subsampling import (
    subsample,
    postselect_by_hamming_right_and_left,
    limit_subspace,
)
from .fermion import (
    SCIState,
    SCIResult,
    bitstring_matrix_to_ci_strs,
    enlarge_batch_from_transitions,
    build_ci_matrix,
    solve_sci,
    solve_sci_batch,
    solve_sci_csf,
    solve_fermion,
    diagonalize_fermionic_hamiltonian,
    optimize_orbitals,
    rotate_integrals,
    compute_ground_state_energy,
    excited_configurations,
    truncate_excited_configurations,
)
from .qubit import (
    sort_and_remove_duplicates,
    matrix_elements_from_pauli,
    project_operator_to_subspace,
    solve_qubit,
)
from .integrated import solve_sqd, solve_sqd_auto, solve_sqd_best, solve_sqd_improved
from .lucj import (
    get_ccsd_amplitudes,
    build_lucj_circuit,
    optimize_ansatz_parameters,
    circuit_stats,
    lucj_report,
    ucj_decomposition,
    ucj_matrix_energy,
    ucj_subspace_energy,
    build_ucj_circuit,
    ucj_assisted_configurations,
    solve_ucj_assisted,
)
from .cipsi import (
    solve_cipsi,
    solve_sqd_active,
    solve_sqd_adaptive,
    solve_hci,
    solve_sqd_ev,
    solve_sqd_distill,
    eigenvector_importance_sample,
)
from .basis import (
    natural_orbitals_from_rdm,
    rotate_to_natural_orbitals,
    ccsd_natural_orbitals,
    rdm1_from_sci_result,
    natural_orbital_occupancies,
    natural_orbital_basis_from_fci,
    NaturalOrbitalResult,
    solve_sqd_natural_orbitals,
)
from . import noise
from .noise import (
    has_gpu,
    statevector_to_density,
    apply_dephasing,
    apply_amp_damping,
    apply_depolarizing,
    density_to_bitstring_matrix,
    apply_t1_bitstrings,
    zero_noise_extrapolate_t1,
    solve_sqd_robust,
    noise_impact,
)
from .predict import (
    gamma_T1,
    predict_sqd_error,
    depth_budget,
    DepthBudget,
    max_depth_for_accuracy,
    plan_sampling,
    SamplingPlan,
    recommend_sqd_params,
    SqdParams,
    calibrate,
)
from .hardware import (
    select_qubits,
    bitstring_matrix_to_energy,
    load_calibration,
    sample_on_hw,
)
from .molecule import (
    MolecularData,
    from_pyscf,
)
from .sampler import (
    sample,
    BACKENDS,
)
from .diagnostics import (
    shannon_entropy,
    subspace_dimension,
    energy_convergence,
    sampling_report,
    extrapolate_infinite_samples,
    extrapolate_energy_variance,
    extrapolate_ev_pt2,
)

__version__ = "0.1.0"

__all__ = [
    # counts
    "bitarray_to_int",
    "int_to_bitarray",
    "counts_dict_to_bitstring_matrix",
    "sample_from_circuit",
    # configuration_recovery
    "recover_configurations",
    "recover_configurations_clustered",
    "postselect_by_hamming_weight",
    "estimate_true_occupancies",
    # subsampling
    "subsample",
    "postselect_by_hamming_right_and_left",
    "limit_subspace",
    # fermion
    "SCIState",
    "SCIResult",
    "bitstring_matrix_to_ci_strs",
    "enlarge_batch_from_transitions",
    "build_ci_matrix",
    "solve_sci",
    "solve_sci_batch",
    "solve_sci_csf",
    "solve_fermion",
    "diagonalize_fermionic_hamiltonian",
    "optimize_orbitals",
    "rotate_integrals",
    "compute_ground_state_energy",
    "excited_configurations",
    "truncate_excited_configurations",
    # qubit
    "sort_and_remove_duplicates",
    "matrix_elements_from_pauli",
    "project_operator_to_subspace",
    "solve_qubit",
    # integrated
    "solve_sqd",
    "solve_sqd_auto",
    "solve_sqd_best",
    "solve_sqd_improved",
    # lucj
    "get_ccsd_amplitudes",
    "build_lucj_circuit",
    "optimize_ansatz_parameters",
    "circuit_stats",
    "lucj_report",
    "ucj_decomposition",
    "ucj_matrix_energy",
    "ucj_subspace_energy",
    "build_ucj_circuit",
    "ucj_assisted_configurations",
    "solve_ucj_assisted",
    # cipsi (PT2 筛选生成集扩展: UCJ 种子 -> 近 FCI 精化 + 主动/自适应采样闭环 + HCI
    #        + 能量-方差外推 EV + 本征矢重要性采样)
    "solve_cipsi",
    "solve_sqd_active",
    "solve_sqd_adaptive",
    "solve_hci",
    "solve_sqd_ev",
    "solve_sqd_distill",
    "eigenvector_importance_sample",
    # basis (基设计: 自然轨道换基 + 自洽迭代, 提升子空间构建效率)
    "natural_orbitals_from_rdm",
    "rotate_to_natural_orbitals",
    "ccsd_natural_orbitals",
    "rdm1_from_sci_result",
    "natural_orbital_occupancies",
    "natural_orbital_basis_from_fci",
    "NaturalOrbitalResult",
    "solve_sqd_natural_orbitals",
    # noise (密度矩阵 Kraus 噪声模拟, cupy GPU 可选 — qiskit-Aer 风格 + tc GPU)
    "has_gpu",
    "statevector_to_density",
    "apply_dephasing",
    "apply_amp_damping",
    "apply_depolarizing",
    "density_to_bitstring_matrix",
    "apply_t1_bitstrings",
    "zero_noise_extrapolate_t1",
    "solve_sqd_robust",
    "noise_impact",
    # predict (噪声容限预测器, 独有)
    "gamma_T1",
    "predict_sqd_error",
    "depth_budget",
    "DepthBudget",
    "max_depth_for_accuracy",
    "plan_sampling",
    "SamplingPlan",
    "recommend_sqd_params",
    "SqdParams",
    "calibrate",
    # hardware (腾讯真机一站式: 校准/选比特/真机采样/SQD 后处理)
    "select_qubits",
    "bitstring_matrix_to_energy",
    "load_calibration",
    "sample_on_hw",
    # molecule (from_pyscf 一键分子接口)
    "MolecularData",
    "from_pyscf",
    # sampler (统一采样后端: tc 模拟 / qcloud 真机)
    "sample",
    "BACKENDS",
    # diagnostics (采样质量诊断报告 + 无限采样外推 A1 + 能量-方差外推 D)
    "shannon_entropy",
    "subspace_dimension",
    "energy_convergence",
    "sampling_report",
    "extrapolate_infinite_samples",
    "extrapolate_energy_variance",
    "extrapolate_ev_pt2",
]
