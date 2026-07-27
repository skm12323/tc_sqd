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
]
