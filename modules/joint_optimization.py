"""Joint geometry + ansatz parameter optimization.

This module implements a *joint* optimization over
- a diatomic bond distance `d`
- a variational ansatz parameter vector `theta`

The objective is the electronic ground-state energy estimation:

    minimize_{theta, d}  E(theta, d) + penalty(d)

Notes
-----
- We keep the ansatz circuit fixed (built at a reference distance) and only
  update the Hamiltonian observable as the geometry changes.
- This requires that the number of qubits stays constant across the distance
  interval. For typical diatomics with fixed electron count and basis, that
  holds.
- COBYLA does not support bounds directly, so we apply a soft quadratic penalty
  outside a chosen interval [d_min, d_max].
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)

from modules.ansatz import build_ansatz
from modules.molecule import (
    MoleculeSpec,
    build_fermionic_hamiltonian,
    build_molecule_problem,
)
from modules.vqe import interpret_expectation_value


@dataclass(frozen=True)
class JointOptimizationResult:
    """Results of joint optimization over (theta, distance)."""

    result: Any
    optimal_distance: float
    optimal_energy: float
    optimal_theta: np.ndarray
    history_distances: list[float]
    history_raw_energies: list[float]
    history_cost_energies: list[float]


def make_qubit_mapper(*, mapper: str, problem: Any):
    """Create a qubit mapper based on the specified mapping scheme.

    Parameters
    ----------
    mapper : str
        The mapping scheme to use. Must be one of:
        - 'JordanWigner': Standard Jordan-Wigner mapping
        - 'BravyiKitaev': Bravyi-Kitaev mapping (more efficient gate depth)
        - 'Parity': Parity mapping (reduces qubit count)
    problem : Any
        The electronic structure problem. Required to extract num_particles
        for the Parity mapper.

    Returns
    -------
    Mapper
        An instance of the requested qubit mapper.

    Raises
    ------
    ValueError
        If an unsupported mapper name is provided.
    """
    if mapper == "JordanWigner":
        return JordanWignerMapper()
    if mapper == "BravyiKitaev":
        return BravyiKitaevMapper()
    if mapper == "Parity":
        return ParityMapper(num_particles=problem.num_particles)

    raise ValueError(f"Unsupported mapper: {mapper}")


def joint_optimize_diatomic_bond_length(
    *,
    atom1: str,
    atom2: str,
    initial_distance: float,
    distance_window: tuple[float, float] | None,
    penalty_strength: float,
    basis: str,
    charge: int,
    spin: int,
    freeze_core: bool,
    active_space: tuple[int, int] | None,  # (num_electrons, num_spatial_orbitals)
    mapper: str,
    ansatz_type: str,
    entanglement: str,
    reps: int,
    optimizer: Optional[Any],
    maxiter: int,
    backend: Optional[Any],
    optimization_level: int,
    seed: Optional[int],
    initial_theta: Optional[np.ndarray],
    verbose: bool,
) -> JointOptimizationResult:
    """Jointly optimize diatomic bond distance and variational ansatz parameters.

    Performs simultaneous optimization of the molecular geometry (bond distance)
    and the variational ansatz parameters to minimize the electronic ground-state
    energy. The ansatz circuit structure is fixed at the reference distance, and
    only the observable (Hamiltonian) changes with geometry.

    The optimization uses COBYLA (or a user-provided optimizer) to minimize:
        E(theta, d) + penalty(d)
    where penalty enforces soft constraints on the bond distance.

    Parameters
    ----------
    atom1 : str
        Element symbol for the first atom (e.g., 'H', 'Li').
    atom2 : str
        Element symbol for the second atom.
    initial_distance : float
        Initial bond distance in Angstrom.
    distance_window : tuple[float, float] | None
        Optional (d_min, d_max) interval. A quadratic penalty is applied
        outside this window to enforce soft constraints.
    penalty_strength : float
        Coefficient for the soft constraint penalty term.
    basis : str
        Quantum chemistry basis set (e.g., 'sto3g', '6-31g').
    charge : int
        Net charge of the molecule.
    spin : int
        Spin multiplicity (2*S, where S is total spin).
    freeze_core : bool
        If True, freeze core electrons in the active space.
    active_space : tuple[int, int] | None
        (num_electrons, num_spatial_orbitals) for the active space.
        If None, all electrons and orbitals are included.
    mapper : str
        Qubit mapper to use ('JordanWigner', 'BravyiKitaev', or 'Parity').
    ansatz_type : str
        Type of ansatz circuit ('EfficientSU2', 'UCCSD', 'HF', etc.).
    entanglement : str
        Entanglement pattern for the ansatz ('linear', 'full', 'circular', etc.).
    reps : int
        Number of repetitions/blocks in the ansatz circuit.
    optimizer : Optional[Any]
        Classical optimizer instance. If None, COBYLA is used.
    maxiter : int
        Maximum number of optimization iterations.
    backend : Optional[Any]
        Quantum backend for simulation. If None, AerSimulator is used.
    optimization_level : int
        Transpilation optimization level (0-3).
    seed : Optional[int]
        Random seed for reproducibility.
    initial_theta : Optional[np.ndarray]
        Initial ansatz parameters. If None, random initialization is used.
    verbose : bool
        If True, print evaluation details during optimization.

    Returns
    -------
    JointOptimizationResult
        Contains:
        - result: Raw optimizer result object
        - optimal_distance: Best bond distance found (in Angstrom)
        - optimal_energy: Electronic energy at optimum (in Hartree)
        - optimal_theta: Best ansatz parameters
        - history_distances: List of all tested distances
        - history_raw_energies: List of electronic energies
        - history_cost_energies: List of cost values (energy + penalty)

    Notes
    -----
    - The number of qubits must remain constant across the distance interval.
    - For diatomics with fixed electron count and basis, this is typically satisfied.
    - The ansatz circuit is built once at the reference distance and reused.

    Raises
    ------
    ValueError
        If distance_window is invalid or if qubit count changes during optimization.
    """

    if backend is None:
        backend = AerSimulator()

    if optimizer is None:
        optimizer = COBYLA(maxiter=maxiter)

    # --- Reference problem (fixes number of qubits and ansatz parameter count) ---

    ref_molecule = f"{atom1} 0.0 0.0 0.0; {atom2} 0.0 0.0 {float(initial_distance)}"

    ref_spec = MoleculeSpec(
        atom=ref_molecule,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    ref_problem = build_molecule_problem(
        ref_spec,
        freeze_core=freeze_core,
        active_space=active_space,
        sanitize_active_space_flag=True,
    )

    qubit_mapper = make_qubit_mapper(mapper=mapper, problem=ref_problem)
    ref_qubit_op = qubit_mapper.map(ref_problem.second_q_ops()[0])

    # Build ansatz at reference point (fixed circuit structure).
    if ansatz_type in ("EfficientSU2", "TwoLocal", "RealAmplitudes"):
        ansatz = build_ansatz(
            ansatz_type,
            num_qubits=ref_qubit_op.num_qubits,
            entanglement=entanglement,
            reps=reps,
        )
    else:
        ansatz = build_ansatz(
            ansatz_type,
            problem=ref_problem,
            qubit_mapper=qubit_mapper,
            reps=reps,
        )

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level
    )
    isa_ansatz = pm.run(ansatz)

    estimator = EstimatorV2()

    num_theta = int(getattr(isa_ansatz, "num_parameters", 0))

    if initial_theta is None:
        rng = np.random.default_rng(seed)
        initial_theta = rng.random(num_theta)
    else:
        initial_theta = np.asarray(initial_theta, dtype=float)
        if initial_theta.shape != (num_theta,):
            raise ValueError(
                f"initial_theta must have shape {(num_theta,)}, got {initial_theta.shape}"
            )

    x0 = np.concatenate([initial_theta, [float(initial_distance)]])

    # --- History tracking ---
    history_distances: list[float] = []
    history_raw: list[float] = []
    history_cost: list[float] = []

    if distance_window is not None:
        d_min, d_max = map(float, distance_window)
        if d_min >= d_max:
            raise ValueError("distance_window must satisfy d_min < d_max")
    else:
        d_min = d_max = 0.0

    def _evaluate_energy(theta: np.ndarray, distance: float) -> float:
        atom_str = f"{atom1} 0.0 0.0 0.0; {atom2} 0.0 0.0 {float(distance)}"
        spec = MoleculeSpec(
            atom=atom_str,
            basis=basis,
            charge=charge,
            spin=spin,
        )
        problem = build_molecule_problem(
            spec,
            freeze_core=freeze_core,
            active_space=active_space,
            sanitize_active_space_flag=True,
        )

        qubit_op = qubit_mapper.map(build_fermionic_hamiltonian(problem))
        if qubit_op.num_qubits != ref_qubit_op.num_qubits:
            raise ValueError(
                "Number of qubits changed across distances; "
                "joint optimization requires a fixed qubit count. "
                f"ref={ref_qubit_op.num_qubits}, got={qubit_op.num_qubits}"
            )

        isa_observables = qubit_op.apply_layout(isa_ansatz.layout)
        job = estimator.run([(isa_ansatz, isa_observables, theta)])
        exp_val = job.result()[0].data.evs
        return float(interpret_expectation_value(exp_val, problem))

    def _penalty(distance: float) -> float:
        if distance_window is None:
            return 0.0
        if distance < d_min:
            return float(penalty_strength) * (d_min - distance) ** 2
        if distance > d_max:
            return float(penalty_strength) * (distance - d_max) ** 2
        return 0.0

    def joint_cost(x: np.ndarray) -> float:
        theta = np.asarray(x[:-1], dtype=float)
        distance = float(x[-1])

        raw_e = _evaluate_energy(theta, distance)
        cost_e = raw_e + _penalty(distance)

        history_distances.append(distance)
        history_raw.append(raw_e)
        history_cost.append(cost_e)

        if verbose:
            k = len(history_cost)
            msg = f"Eval {k:03d}: d={distance:.6f} {DistanceUnit.ANGSTROM.name}, E={raw_e:.10f} Ha"
            if distance_window is not None:
                msg += f", cost={cost_e:.10f}"
            print(msg)

        return cost_e

    # Edge case: HF (or other fixed-state circuits) -> theta is empty.
    if num_theta == 0:
        e0 = float(_evaluate_energy(np.array([], dtype=float), float(initial_distance)))
        c0 = float(e0 + _penalty(float(initial_distance)))
        history_distances.append(float(initial_distance))
        history_raw.append(e0)
        history_cost.append(c0)
        result = SimpleNamespace(
            fun=c0,
            x=np.array([float(initial_distance)], dtype=float),
            nit=0,
            nfev=1,
            success=True,
            message="No variational parameters; evaluated once.",
        )
        return JointOptimizationResult(
            result=result,
            optimal_distance=float(initial_distance),
            optimal_energy=e0,
            optimal_theta=np.array([], dtype=float),
            history_distances=history_distances,
            history_raw_energies=history_raw,
            history_cost_energies=history_cost,
        )

    opt_result = optimizer.minimize(fun=joint_cost, x0=x0)

    x_opt = np.asarray(opt_result.x, dtype=float)
    theta_opt = x_opt[:-1]
    d_opt = float(x_opt[-1])

    # Report *raw* energy at optimum (without penalty) for convenience.
    e_opt_raw = float(_evaluate_energy(theta_opt, d_opt))

    return JointOptimizationResult(
        result=opt_result,
        optimal_distance=d_opt,
        optimal_energy=e_opt_raw,
        optimal_theta=np.array(theta_opt, copy=True),
        history_distances=history_distances,
        history_raw_energies=history_raw,
        history_cost_energies=history_cost,
    )