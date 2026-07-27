"""Tests for the integrated SQD entry point ``tc_sqd.solve_sqd``.

Covers both operation modes of ``solve_sqd``:

* ``mode="single"``     -- one-shot configuration recovery + diagonalisation
* ``mode="iterative"``  -- full iterative SQD loop

Run with::

    PYTHONPATH=src python tests.py

(or simply ``python tests.py`` -- ``src`` is added to ``sys.path`` here).
No pip / environment changes are performed by this script.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from pyscf import gto, scf

import tensorcircuit as tc

import tc_sqd
from tc_sqd import solve_sqd, compute_ground_state_energy

tc.set_backend("numpy")


# --------------------------------------------------------------------------- #
# Shared H2 reference setup
# --------------------------------------------------------------------------- #
def _build_h2():
    """Build MO-basis integrals for H2 / STO-3G and return everything needed."""
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol).run(verbose=0)
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri_ao = mol.intor("int2e_sph")
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, mo, mo, mo, mo)
    ecore = mf.energy_nuc()
    norb = mo.shape[1]
    nelec = (mol.nelectron // 2, mol.nelectron // 2)
    return h1e, eri, norb, nelec, ecore


def _build_circuit(norb, nelec):
    """HF state + entangling gates (mirrors examples/h2_sqd_demo.py)."""
    nq = 2 * norb
    c = tc.Circuit(nq)
    c.x(0)        # alpha orbital 0
    c.x(norb)     # beta  orbital 0
    theta = 0.8
    c.ry(0, theta=theta)
    c.cnot(0, 1)
    c.ry(0, theta=-theta)
    c.ry(norb, theta=theta)
    c.cnot(norb, norb + 1)
    c.ry(norb, theta=-theta)
    return c


def _fci_reference(h1e, eri, norb, nelec, ecore):
    return compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="fci"
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_single_mode():
    print("=" * 60)
    print("Test: solve_sqd(mode='single')")
    print("=" * 60)
    h1e, eri, norb, nelec, ecore = _build_h2()
    e_fci = _fci_reference(h1e, eri, norb, nelec, ecore)

    circ = _build_circuit(norb, nelec)
    bsm, probs = tc_sqd.sample_from_circuit(circ, n_samples=3000)

    result = solve_sqd(
        h1e, eri, norb, nelec,
        ecore=ecore,
        bitstring_matrix=bsm,
        probabilities=probs,
        mode="single",
    )
    e_sqd = result.energy + ecore
    print(f"  E(SQD, single) = {e_sqd:.8f}")
    print(f"  E(FCI)         = {e_fci:.8f}")
    print(f"  diff           = {e_sqd - e_fci:.2e}")
    assert abs(e_sqd - e_fci) < 2e-2, f"single-mode energy too far: {e_sqd - e_fci}"
    print("  PASS\n")


def test_iterative_mode():
    print("=" * 60)
    print("Test: solve_sqd(mode='iterative')")
    print("=" * 60)
    h1e, eri, norb, nelec, ecore = _build_h2()
    e_fci = _fci_reference(h1e, eri, norb, nelec, ecore)

    circ = _build_circuit(norb, nelec)
    bsm, probs = tc_sqd.sample_from_circuit(circ, n_samples=3000)

    result = solve_sqd(
        h1e, eri, norb, nelec,
        ecore=ecore,
        bitstring_matrix=bsm,
        probabilities=probs,
        mode="iterative",
        max_iterations=5,
        seed=42,
        verbose=True,
    )
    e_sqd = result.energy + ecore
    print(f"  E(SQD, iter)   = {e_sqd:.8f}")
    print(f"  E(FCI)         = {e_fci:.8f}")
    print(f"  diff           = {e_sqd - e_fci:.2e}")
    assert abs(e_sqd - e_fci) < 2e-2, f"iterative-mode energy too far: {e_sqd - e_fci}"
    print("  PASS\n")


def test_circuit_mode():
    """``solve_sqd`` can sample directly from a circuit (one-call pipeline)."""
    print("=" * 60)
    print("Test: solve_sqd(circuit=...) one-call sampling")
    print("=" * 60)
    h1e, eri, norb, nelec, ecore = _build_h2()
    e_fci = _fci_reference(h1e, eri, norb, nelec, ecore)

    circ = _build_circuit(norb, nelec)
    result = solve_sqd(
        h1e, eri, norb, nelec,
        ecore=ecore,
        circuit=circ,
        n_samples=3000,
        mode="single",
        seed=7,
    )
    e_sqd = result.energy + ecore
    print(f"  E(SQD, single) = {e_sqd:.8f}")
    print(f"  E(FCI)         = {e_fci:.8f}")
    print(f"  diff           = {e_sqd - e_fci:.2e}")
    assert abs(e_sqd - e_fci) < 2e-2
    print("  PASS\n")


def test_invalid_mode():
    print("=" * 60)
    print("Test: invalid mode raises ValueError")
    print("=" * 60)
    h1e, eri, norb, nelec, ecore = _build_h2()
    try:
        solve_sqd(h1e, eri, norb, nelec, mode="bogus")
        raise AssertionError("expected ValueError for invalid mode")
    except ValueError:
        pass
    print("  PASS\n")


def test_default_uniform_probs():
    """Omitting ``probabilities`` falls back to uniform weights."""
    print("=" * 60)
    print("Test: uniform probabilities default")
    print("=" * 60)
    h1e, eri, norb, nelec, ecore = _build_h2()
    e_fci = _fci_reference(h1e, eri, norb, nelec, ecore)

    circ = _build_circuit(norb, nelec)
    bsm, _ = tc_sqd.sample_from_circuit(circ, n_samples=2000)
    result = solve_sqd(
        h1e, eri, norb, nelec,
        ecore=ecore,
        bitstring_matrix=bsm,           # no probabilities -> uniform
        mode="single",
    )
    e_sqd = result.energy + ecore
    print(f"  E(SQD, single) = {e_sqd:.8f}")
    print(f"  E(FCI)         = {e_fci:.8f}")
    assert abs(e_sqd - e_fci) < 2e-2
    print("  PASS\n")


if __name__ == "__main__":
    print("Running integrated SQD tests (tc environment)...\n")
    test_single_mode()
    test_iterative_mode()
    test_circuit_mode()
    test_invalid_mode()
    test_default_uniform_probs()
    print("All integrated SQD tests passed.")
