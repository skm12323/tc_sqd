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
- ``tc_sqd.integrated``              -- one-call ``solve_sqd`` entry point
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
    postselect_by_hamming_weight,
)
from .subsampling import (
    subsample,
    postselect_by_hamming_right_and_left,
)
from .fermion import (
    SCIState,
    SCIResult,
    bitstring_matrix_to_ci_strs,
    enlarge_batch_from_transitions,
    build_ci_matrix,
    solve_sci,
    solve_sci_batch,
    solve_fermion,
    diagonalize_fermionic_hamiltonian,
    optimize_orbitals,
    rotate_integrals,
    compute_ground_state_energy,
)
from .qubit import (
    sort_and_remove_duplicates,
    matrix_elements_from_pauli,
    project_operator_to_subspace,
    solve_qubit,
)
from .integrated import solve_sqd
from .lucj import (
    get_ccsd_amplitudes,
    build_lucj_circuit,
)
from . import noise
from .noise import (
    has_gpu,
    statevector_to_density,
    apply_dephasing,
    apply_amp_damping,
    apply_depolarizing,
    density_to_bitstring_matrix,
)
from .predict import (
    gamma_T1,
    predict_sqd_error,
    max_depth_for_accuracy,
)
from .hardware import (
    select_qubits,
    bitstring_matrix_to_energy,
    load_calibration,
    sample_on_hw,
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
    "postselect_by_hamming_weight",
    # subsampling
    "subsample",
    "postselect_by_hamming_right_and_left",
    # fermion
    "SCIState",
    "SCIResult",
    "bitstring_matrix_to_ci_strs",
    "enlarge_batch_from_transitions",
    "build_ci_matrix",
    "solve_sci",
    "solve_sci_batch",
    "solve_fermion",
    "diagonalize_fermionic_hamiltonian",
    "optimize_orbitals",
    "rotate_integrals",
    "compute_ground_state_energy",
    # qubit
    "sort_and_remove_duplicates",
    "matrix_elements_from_pauli",
    "project_operator_to_subspace",
    "solve_qubit",
    # integrated
    "solve_sqd",
    # lucj
    "get_ccsd_amplitudes",
    "build_lucj_circuit",
    # noise (密度矩阵 Kraus 噪声模拟, cupy GPU 可选 — qiskit-Aer 风格 + tc GPU)
    "has_gpu",
    "statevector_to_density",
    "apply_dephasing",
    "apply_amp_damping",
    "apply_depolarizing",
    "density_to_bitstring_matrix",
    # predict (噪声容限预测器, 独有)
    "gamma_T1",
    "predict_sqd_error",
    "max_depth_for_accuracy",
    # hardware (腾讯真机一站式: 校准/选比特/真机采样/SQD 后处理)
    "select_qubits",
    "bitstring_matrix_to_energy",
    "load_calibration",
    "sample_on_hw",
]
