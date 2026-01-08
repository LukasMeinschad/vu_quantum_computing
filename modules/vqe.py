"""VQE utilities.

This module contains small, reusable building blocks for running VQE based on
Qiskit Algorithms + Qiskit Nature.

Included
--------
- `interpret_expectation_value`: converts an estimator expectation value into an
  interpreted total energy (Hartree) via `ElectronicStructureProblem.interpret`.
- `energy_cost_function`: computes the energy for a parameter vector.
- `run_vqe_single_point`: runs an optimizer-driven VQE loop for a single fixed
  Hamiltonian and stores the optimization history (no plotting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from types import SimpleNamespace

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms import MinimumEigensolverResult


def _build_measurement_circuit_for_pauli(base_circuit: Any, pauli_label: str) -> Any:
    """Return a circuit that measures the given Pauli string in the Z basis.

    This appends the standard basis-change gates then measures all qubits.
    """
    qc = base_circuit.copy()

    # Qiskit Pauli labels are ordered with qubit-0 on the right.
    n = len(pauli_label)
    if getattr(qc, "num_qubits", n) != n:
        # Allow some flexibility, but keep failure mode obvious.
        raise ValueError(
            "Pauli label length does not match circuit qubit count: "
            f"len(label)={n}, circuit.num_qubits={getattr(qc, 'num_qubits', None)}"
        )

    for qubit in range(n):
        op = pauli_label[n - 1 - qubit]
        if op == "I":
            continue
        if op == "X":
            qc.h(qubit)
        elif op == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif op == "Z":
            pass
        else:
            raise ValueError(f"Unsupported Pauli op '{op}' in label '{pauli_label}'")

    # Ensure measurements are present for Sampler.
    qc.measure_all()
    return qc


def _counts_to_pauli_expectation(counts: dict[str, int], pauli_label: str) -> float:
    """Compute expectation value of a Pauli string from Z-basis counts.

    Assumes the circuit included the necessary basis rotations for X/Y terms.
    """
    n = len(pauli_label)
    active_qubits = [q for q in range(n) if pauli_label[n - 1 - q] != "I"]
    if not active_qubits:
        return 1.0

    shots = int(sum(counts.values()))
    if shots <= 0:
        raise ValueError("No shots in counts; cannot estimate expectation value")

    exp = 0.0
    for bitstring, c in counts.items():
        # bitstring is big-endian with qubit-0 on the right
        parity = 0
        for q in active_qubits:
            bit = bitstring[n - 1 - q]
            parity ^= bit == "1"
        eigenvalue = -1.0 if parity else 1.0
        exp += eigenvalue * float(c)

    return exp / float(shots)


def _estimate_sparsepauliop_expectation_with_sampler(
    *,
    sampler: Any,
    base_circuit: Any,
    params: np.ndarray,
    operator: Any,
    shots: int,
    measurement_circuits: Optional[list[Any]] = None,
    pauli_labels: Optional[list[str]] = None,
) -> float:
    """Estimate <operator> using a shot-based Sampler.

    `operator` is expected to be a `SparsePauliOp` (or compatible) with `.paulis`
    and `.coeffs`.
    """
    if shots is None or int(shots) <= 0:
        raise ValueError("shots must be a positive integer")

    labels = (
        list(pauli_labels)
        if pauli_labels is not None
        else list(operator.paulis.to_labels())
    )
    coeffs = np.asarray(operator.coeffs)
    if len(labels) != len(coeffs):
        raise ValueError("Observable labels/coeffs length mismatch")

    # Pre-build measurement circuits if not provided.
    if measurement_circuits is None:
        measurement_circuits = [
            _build_measurement_circuit_for_pauli(base_circuit, lab) for lab in labels
        ]

    # Batch all non-identity terms into one sampler call.
    energy_ev = 0.0
    pubs: list[Any] = []
    pub_indices: list[int] = []
    for i, (lab, coeff) in enumerate(zip(labels, coeffs)):
        coeff_r = float(np.real(coeff))
        if np.isclose(coeff_r, 0.0):
            continue
        if set(lab) == {"I"}:
            energy_ev += coeff_r
            continue
        pubs.append((measurement_circuits[i], params))
        pub_indices.append(i)

    if pubs:
        job = sampler.run(pubs, shots=int(shots))
        res = job.result()
        for j, i in enumerate(pub_indices):
            counts = res[j].data.meas.get_counts()
            term_exp = _counts_to_pauli_expectation(counts, labels[i])
            energy_ev += float(np.real(coeffs[i])) * term_exp

    return float(energy_ev)


@dataclass(frozen=True)
class SinglePointVQEResult:
    """Container for a single-point VQE run."""

    result: Any
    energies: list[float]
    isa_ansatz: Any
    isa_observables: Any


def interpret_expectation_value(exp_val: Any, problem: Any) -> float:
    """Interpret an expectation value in the context of an electronic problem.

    Parameters
    ----------
    exp_val:
            Expectation value (scalar or 1-element array) returned by Qiskit Estimator.
    problem:
            Qiskit Nature ElectronicStructureProblem (or compatible) with `.interpret`.

    Returns
    -------
    float
            Interpreted total energy in Hartree.
    """
    ev = np.real(exp_val)
    if isinstance(ev, np.ndarray):
        ev = ev.item() if ev.size == 1 else float(ev.ravel()[0])

    sol = MinimumEigensolverResult()
    sol.eigenvalue = ev
    return float(problem.interpret(sol).total_energies[0])


def energy_cost_function(
    params: np.ndarray,
    *,
    estimator: Any = None,
    isa_ansatz: Any,
    isa_observables: Any,
    problem: Any,
    sampler: Any = None,
    shots: Optional[int] = None,
    measurement_circuits: Optional[list[Any]] = None,
    pauli_labels: Optional[list[str]] = None,
) -> float:
    """VQE energy objective for a given parameter vector."""
    if sampler is not None:
        ev = _estimate_sparsepauliop_expectation_with_sampler(
            sampler=sampler,
            base_circuit=isa_ansatz,
            params=np.asarray(params, dtype=float),
            operator=isa_observables,
            shots=int(shots) if shots is not None else 1024,
            measurement_circuits=measurement_circuits,
            pauli_labels=pauli_labels,
        )
        return interpret_expectation_value(ev, problem)

    if estimator is None:
        raise ValueError("Either estimator or sampler must be provided")
    job = estimator.run([(isa_ansatz, isa_observables, params)])
    exp_val = job.result()[0].data.evs
    return interpret_expectation_value(exp_val, problem)


def run_vqe_single_point(
    *,
    problem: Any,
    qubit_hamiltonian: Any,
    ansatz: Any,
    optimizer: Optional[Any] = None,
    initial_params: Optional[np.ndarray] = None,
    backend: Optional[Any] = None,
    estimator: Optional[Any] = None,
    sampler: Optional[Any] = None,
    shots: Optional[int] = None,
    optimization_level: int = 0,
    seed: Optional[int] = None,
    store_energies: bool = True,
    verbose: bool = False,
) -> SinglePointVQEResult:
    """Run VQE for a single fixed point (no plotting).

    Notes
    -----
    - Energy tracking is done by wrapping the cost function (no SciPy callback).
      This avoids the "callback passed twice" error and avoids double-evaluations.
    - Stored energies are per cost-function evaluation (not necessarily optimizer
      iterations for all optimizers).
    """
    if backend is None:
        backend = AerSimulator()

    if optimizer is None:
        optimizer = COBYLA(maxiter=300)

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level
    )
    isa_ansatz = pm.run(ansatz)
    isa_observables = qubit_hamiltonian.apply_layout(isa_ansatz.layout)

    # Default primitives: use shot-based sampler if provided, else estimator.
    if sampler is None and estimator is None:
        estimator = EstimatorV2()
    if sampler is None and shots is not None:
        raise ValueError("shots was set but sampler=None; provide sampler=SamplerV2()")
    if sampler is None and estimator is None:
        raise ValueError("Either estimator or sampler must be provided")
    if sampler is None:
        # keep type checkers happy
        pass
    else:
        # Precompute measurement circuits once per VQE run for speed.
        pauli_labels = list(isa_observables.paulis.to_labels())
        measurement_circuits = [
            _build_measurement_circuit_for_pauli(isa_ansatz, lab)
            for lab in pauli_labels
        ]

    if initial_params is None:
        rng = np.random.default_rng(seed)
        initial_params = rng.random(isa_ansatz.num_parameters)

    # Edge case: circuits like HartreeFock have no free parameters.
    # SciPy-based optimizers can error on zero-dimensional optimization. In this
    # case we simply evaluate the energy once and return.
    if getattr(isa_ansatz, "num_parameters", 0) == 0:
        energy = float(
            energy_cost_function(
                initial_params,
                estimator=estimator,
                isa_ansatz=isa_ansatz,
                isa_observables=isa_observables,
                problem=problem,
                sampler=sampler,
                shots=shots,
                measurement_circuits=(
                    measurement_circuits if sampler is not None else None
                ),
                pauli_labels=pauli_labels if sampler is not None else None,
            )
        )
        energies: list[float] = [energy] if store_energies else []
        result = SimpleNamespace(
            fun=energy,
            x=np.array(initial_params, copy=True),
            nit=0,
            nfev=1,
            success=True,
            message="No variational parameters; evaluated once.",
        )
        if verbose:
            print(f"Eval 001 - Energy = {energy:.10f} Ha")
        return SinglePointVQEResult(
            result=result,
            energies=energies,
            isa_ansatz=isa_ansatz,
            isa_observables=isa_observables,
        )

    energies: list[float] = []

    def cost_logged(p: np.ndarray) -> float:
        e = float(
            energy_cost_function(
                p,
                estimator=estimator,
                isa_ansatz=isa_ansatz,
                isa_observables=isa_observables,
                problem=problem,
                sampler=sampler,
                shots=shots,
                measurement_circuits=(
                    measurement_circuits if sampler is not None else None
                ),
                pauli_labels=pauli_labels if sampler is not None else None,
            )
        )
        if store_energies:
            energies.append(e)
        if verbose:
            print(f"Eval {len(energies):03d} - Energy = {e:.10f} Ha")
        return e

    result = optimizer.minimize(fun=cost_logged, x0=initial_params)
    return SinglePointVQEResult(
        result=result,
        energies=energies,
        isa_ansatz=isa_ansatz,
        isa_observables=isa_observables,
    )
