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

import numpy as np

from typing import Any, Optional
from dataclasses import dataclass
from types import SimpleNamespace
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms import MinimumEigensolverResult

from modules.results_io import results_print


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
    estimator: Any,
    isa_ansatz: Any,
    isa_observables: Any,
    problem: Any,
) -> float:
    """VQE energy objective for a given parameter vector."""

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
    optimization_level: int = 0,
    seed: Optional[int] = None,
    store_energies: bool = True,
    verbose: bool = False,
) -> SinglePointVQEResult:
    """Run VQE for a single fixed point.

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

    if estimator is None:
        estimator = EstimatorV2()

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
            results_print(f"Eval 001 - Energy = {energy:.10f} Ha")

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
            )
        )
        if store_energies:
            energies.append(e)
        if verbose:
            results_print(f"Eval {len(energies):03d} - Energy = {e:.10f} Ha")

        return e

    result = optimizer.minimize(fun=cost_logged, x0=initial_params)
    return SinglePointVQEResult(
        result=result,
        energies=energies,
        isa_ansatz=isa_ansatz,
        isa_observables=isa_observables,
    )
