"""Joint geometry + ansatz parameter optimization.

This module implements a *joint* optimization over
- a diatomic bond distance `d`
- a variational ansatz parameter vector `theta`

The objective is the electronic ground-state energy evaluated by an Estimator:

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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional

import numpy as np

from qiskit_nature.units import DistanceUnit

from modules.ansatz import AnsatzKind, build_ansatz
from modules.molecule import MoleculeSpec, build_fermionic_hamiltonian, build_molecule_problem
from modules.vqe import interpret_expectation_value


Axis = Literal["x", "y", "z"]


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


@dataclass(frozen=True)
class WaterJointOptimizationResult:
    """Results of joint optimization over (theta, r1, r2, angle_deg) for H2O."""

    result: Any
    optimal_r1: float
    optimal_r2: float
    optimal_angle_deg: float
    optimal_energy: float
    optimal_theta: np.ndarray
    history_r1: list[float]
    history_r2: list[float]
    history_angle_deg: list[float]
    history_raw_energies: list[float]
    history_cost_energies: list[float]


def _diatomic_atom_string(
    atom1: str,
    atom2: str,
    distance: float,
    *,
    axis: Axis = "x",
    symmetric: bool = True,
) -> str:
    axis = axis.lower()  # type: ignore[assignment]
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of: 'x', 'y', 'z'")

    if symmetric:
        a = -0.5 * float(distance)
        b = +0.5 * float(distance)
        p1 = {"x": (a, 0.0, 0.0), "y": (0.0, a, 0.0), "z": (0.0, 0.0, a)}[axis]
        p2 = {"x": (b, 0.0, 0.0), "y": (0.0, b, 0.0), "z": (0.0, 0.0, b)}[axis]
    else:
        a = 0.0
        b = float(distance)
        p1 = {"x": (a, 0.0, 0.0), "y": (0.0, a, 0.0), "z": (0.0, 0.0, a)}[axis]
        p2 = {"x": (b, 0.0, 0.0), "y": (0.0, b, 0.0), "z": (0.0, 0.0, b)}[axis]

    return f"{atom1} {p1[0]} {p1[1]} {p1[2]}; {atom2} {p2[0]} {p2[1]} {p2[2]}"


def _make_qubit_mapper(*, mapper: str, problem: Any):
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )

    if mapper == "JordanWigner":
        return JordanWignerMapper()
    if mapper == "BravyiKitaev":
        return BravyiKitaevMapper()
    if mapper == "Parity":
        return ParityMapper(num_particles=problem.num_particles)
    raise ValueError(f"Unsupported mapper: {mapper}")


def _water_atom_string(
    *,
    r1: float,
    r2: float,
    angle_deg: float,
) -> str:
    """Create a PySCF atom string for H2O with O at origin.

    Geometry convention
    -------------------
    - O at (0, 0, 0)
    - H1 at (0, 0, r1)
    - H2 in the xz-plane with angle(H1-O-H2)=angle_deg:
        H2 = (r2*sin(angle), 0, r2*cos(angle))

    This guarantees the H–O–H angle equals `angle_deg`.
    """

    r1 = float(r1)
    r2 = float(r2)
    angle_deg = float(angle_deg)

    angle_rad = float(np.deg2rad(angle_deg))
    x2 = r2 * float(np.sin(angle_rad))
    z2 = r2 * float(np.cos(angle_rad))

    return f"O 0.0 0.0 0.0; H 0.0 0.0 {r1}; H {x2} 0.0 {z2}"


def joint_optimize_diatomic_bond_length(
    *,
    atom1: str,
    atom2: str,
    initial_distance: float,
    distance_window: tuple[float, float] | None = None,
    penalty_strength: float = 100.0,
    axis: Axis = "x",
    symmetric_geometry: bool = True,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    unit: DistanceUnit = DistanceUnit.ANGSTROM,
    freeze_core: bool = False,
    active_space: tuple[int, int] | None = None,  # (num_electrons, num_spatial_orbitals)
    mapper: str = "JordanWigner",
    ansatz_kind: AnsatzKind = "EfficientSU2",
    entanglement: str = "linear",
    reps: int = 1,
    optimizer: Optional[Any] = None,
    maxiter: int = 200,
    backend: Optional[Any] = None,
    optimization_level: int = 0,
    seed: Optional[int] = 42,
    initial_theta: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> JointOptimizationResult:
    """Jointly optimize diatomic bond distance and ansatz parameters.

    Parameters
    ----------
    distance_window:
        Optional (d_min, d_max). If provided, a soft quadratic penalty is added
        outside this interval.

    Returns
    -------
    JointOptimizationResult
        Includes optimizer result and full evaluation history.
    """

    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if backend is None:
        backend = AerSimulator()

    if optimizer is None:
        optimizer = COBYLA(maxiter=maxiter)

    # --- Reference problem (fixes number of qubits and ansatz parameter count) ---
    ref_atom = _diatomic_atom_string(
        atom1,
        atom2,
        float(initial_distance),
        axis=axis,
        symmetric=symmetric_geometry,
    )
    ref_spec = MoleculeSpec(
        atom=ref_atom,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
    )
    ref_problem = build_molecule_problem(
        ref_spec,
        freeze_core=freeze_core,
        active_space=active_space,
        sanitize_active_space=True,
    )

    qubit_mapper = _make_qubit_mapper(mapper=mapper, problem=ref_problem)

    ref_qubit_op = qubit_mapper.map(build_fermionic_hamiltonian(ref_problem))

    # Build ansatz at reference point (fixed circuit structure).
    if ansatz_kind in ("EfficientSU2", "TwoLocal", "RealAmplitudes"):
        ansatz = build_ansatz(
            ansatz_kind,
            num_qubits=ref_qubit_op.num_qubits,
            entanglement=entanglement,
            reps=reps,
        )
    else:
        ansatz = build_ansatz(
            ansatz_kind,
            problem=ref_problem,
            qubit_mapper=qubit_mapper,
            reps=reps,
        )

    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_ansatz = pm.run(ansatz)

    estimator = EstimatorV2()

    num_theta = int(getattr(isa_ansatz, "num_parameters", 0))

    if initial_theta is None:
        rng = np.random.default_rng(seed)
        initial_theta = rng.random(num_theta)
    else:
        initial_theta = np.asarray(initial_theta, dtype=float)
        if initial_theta.shape != (num_theta,):
            raise ValueError(f"initial_theta must have shape {(num_theta,)}, got {initial_theta.shape}")

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
        atom_str = _diatomic_atom_string(
            atom1,
            atom2,
            float(distance),
            axis=axis,
            symmetric=symmetric_geometry,
        )
        spec = MoleculeSpec(
            atom=atom_str,
            basis=basis,
            charge=charge,
            spin=spin,
            unit=unit,
        )
        problem = build_molecule_problem(
            spec,
            freeze_core=freeze_core,
            active_space=active_space,
            sanitize_active_space=True,
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
            msg = f"Eval {k:03d}: d={distance:.6f} {unit.name}, E={raw_e:.10f} Ha"
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


def joint_optimize_water_geometry(
    *,
    initial_r1: float,
    initial_r2: float,
    initial_angle_deg: float,
    r1_window: tuple[float, float] | None = None,
    r2_window: tuple[float, float] | None = None,
    angle_window_deg: tuple[float, float] | None = None,
    penalty_strength: float = 100.0,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    unit: DistanceUnit = DistanceUnit.ANGSTROM,
    freeze_core: bool = False,
    active_space: tuple[int, int] | None = None,  # (num_electrons, num_spatial_orbitals)
    mapper: str = "JordanWigner",
    ansatz_kind: AnsatzKind = "EfficientSU2",
    entanglement: str = "linear",
    reps: int = 1,
    optimizer: Optional[Any] = None,
    maxiter: int = 200,
    backend: Optional[Any] = None,
    optimization_level: int = 0,
    seed: Optional[int] = 42,
    initial_theta: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> WaterJointOptimizationResult:
    """Jointly optimize H2O geometry (r1, r2, angle) and ansatz parameters.

    Parameters
    ----------
    initial_r1, initial_r2:
        Initial O–H bond lengths in `unit` (typically Angstrom).
    initial_angle_deg:
        Initial H–O–H bond angle in degrees.
    r1_window, r2_window, angle_window_deg:
        Optional soft constraint windows. Outside the window a quadratic penalty
        is added to the objective.

    Returns
    -------
    WaterJointOptimizationResult
        Includes optimizer result and full evaluation history.
    """

    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if backend is None:
        backend = AerSimulator()

    if optimizer is None:
        optimizer = COBYLA(maxiter=maxiter)

    # --- Reference problem (fixes qubit count and ansatz parameter count) ---
    ref_atom = _water_atom_string(r1=initial_r1, r2=initial_r2, angle_deg=initial_angle_deg)
    ref_spec = MoleculeSpec(
        atom=ref_atom,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
    )
    ref_problem = build_molecule_problem(
        ref_spec,
        freeze_core=freeze_core,
        active_space=active_space,
        sanitize_active_space=True,
    )

    qubit_mapper = _make_qubit_mapper(mapper=mapper, problem=ref_problem)
    ref_qubit_op = qubit_mapper.map(build_fermionic_hamiltonian(ref_problem))

    if ansatz_kind in ("EfficientSU2", "TwoLocal", "RealAmplitudes"):
        ansatz = build_ansatz(
            ansatz_kind,
            num_qubits=ref_qubit_op.num_qubits,
            entanglement=entanglement,
            reps=reps,
        )
    else:
        ansatz = build_ansatz(
            ansatz_kind,
            problem=ref_problem,
            qubit_mapper=qubit_mapper,
            reps=reps,
        )

    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_ansatz = pm.run(ansatz)
    estimator = EstimatorV2()

    num_theta = int(getattr(isa_ansatz, "num_parameters", 0))
    if initial_theta is None:
        rng = np.random.default_rng(seed)
        initial_theta = rng.random(num_theta)
    else:
        initial_theta = np.asarray(initial_theta, dtype=float)
        if initial_theta.shape != (num_theta,):
            raise ValueError(f"initial_theta must have shape {(num_theta,)}, got {initial_theta.shape}")

    x0 = np.concatenate(
        [
            np.asarray(initial_theta, dtype=float),
            [float(initial_r1), float(initial_r2), float(initial_angle_deg)],
        ]
    )

    history_r1: list[float] = []
    history_r2: list[float] = []
    history_angle: list[float] = []
    history_raw: list[float] = []
    history_cost: list[float] = []

    def _window_penalty(value: float, window: tuple[float, float] | None) -> float:
        if window is None:
            return 0.0
        lo, hi = map(float, window)
        if lo >= hi:
            raise ValueError("window must satisfy lo < hi")
        if value < lo:
            return float(penalty_strength) * (lo - value) ** 2
        if value > hi:
            return float(penalty_strength) * (value - hi) ** 2
        return 0.0

    def _geometry_is_valid(r1: float, r2: float, angle_deg: float) -> bool:
        # keep PySCF happy and avoid degenerate/invalid angles
        return (r1 > 0.0) and (r2 > 0.0) and (0.0 < angle_deg < 180.0)

    def _evaluate_energy(theta: np.ndarray, r1: float, r2: float, angle_deg: float) -> float:
        atom_str = _water_atom_string(r1=r1, r2=r2, angle_deg=angle_deg)
        spec = MoleculeSpec(
            atom=atom_str,
            basis=basis,
            charge=charge,
            spin=spin,
            unit=unit,
        )
        problem = build_molecule_problem(
            spec,
            freeze_core=freeze_core,
            active_space=active_space,
            sanitize_active_space=True,
        )

        qubit_op = qubit_mapper.map(build_fermionic_hamiltonian(problem))
        if qubit_op.num_qubits != ref_qubit_op.num_qubits:
            raise ValueError(
                "Number of qubits changed across geometries; "
                "joint optimization requires a fixed qubit count. "
                f"ref={ref_qubit_op.num_qubits}, got={qubit_op.num_qubits}"
            )

        isa_observables = qubit_op.apply_layout(isa_ansatz.layout)
        job = estimator.run([(isa_ansatz, isa_observables, theta)])
        exp_val = job.result()[0].data.evs
        return float(interpret_expectation_value(exp_val, problem))

    def joint_cost(x: np.ndarray) -> float:
        theta = np.asarray(x[:-3], dtype=float)
        r1 = float(x[-3])
        r2 = float(x[-2])
        angle_deg = float(x[-1])

        # Penalty terms (soft constraints)
        pen = 0.0
        pen += _window_penalty(r1, r1_window)
        pen += _window_penalty(r2, r2_window)
        pen += _window_penalty(angle_deg, angle_window_deg)

        # If geometry is invalid, don't call the electronic structure driver.
        if not _geometry_is_valid(r1, r2, angle_deg):
            raw_e = float("inf")
            cost_e = 1e6 + pen
        else:
            raw_e = _evaluate_energy(theta, r1, r2, angle_deg)
            cost_e = raw_e + pen

        history_r1.append(r1)
        history_r2.append(r2)
        history_angle.append(angle_deg)
        history_raw.append(raw_e)
        history_cost.append(cost_e)

        if verbose:
            k = len(history_cost)
            msg = (
                f"Eval {k:03d}: r1={r1:.6f}, r2={r2:.6f} {unit.name}, "
                f"angle={angle_deg:.3f} deg, E={raw_e:.10f} Ha"
            )
            if (r1_window is not None) or (r2_window is not None) or (angle_window_deg is not None):
                msg += f", cost={cost_e:.10f}"
            print(msg)

        return cost_e

    # Edge case: fixed-state circuits -> theta is empty; still optimize geometry.
    if num_theta == 0:
        x0_geom = np.array([float(initial_r1), float(initial_r2), float(initial_angle_deg)], dtype=float)

        def geom_cost(g: np.ndarray) -> float:
            return joint_cost(np.concatenate([np.array([], dtype=float), np.asarray(g, dtype=float)]))

        opt_result = optimizer.minimize(fun=geom_cost, x0=x0_geom)
        r1_opt, r2_opt, ang_opt = map(float, opt_result.x)
        e_opt_raw = float(_evaluate_energy(np.array([], dtype=float), r1_opt, r2_opt, ang_opt))
        return WaterJointOptimizationResult(
            result=opt_result,
            optimal_r1=r1_opt,
            optimal_r2=r2_opt,
            optimal_angle_deg=ang_opt,
            optimal_energy=e_opt_raw,
            optimal_theta=np.array([], dtype=float),
            history_r1=history_r1,
            history_r2=history_r2,
            history_angle_deg=history_angle,
            history_raw_energies=history_raw,
            history_cost_energies=history_cost,
        )

    opt_result = optimizer.minimize(fun=joint_cost, x0=x0)

    x_opt = np.asarray(opt_result.x, dtype=float)
    theta_opt = x_opt[:-3]
    r1_opt = float(x_opt[-3])
    r2_opt = float(x_opt[-2])
    ang_opt = float(x_opt[-1])

    e_opt_raw = float(_evaluate_energy(theta_opt, r1_opt, r2_opt, ang_opt))

    return WaterJointOptimizationResult(
        result=opt_result,
        optimal_r1=r1_opt,
        optimal_r2=r2_opt,
        optimal_angle_deg=ang_opt,
        optimal_energy=e_opt_raw,
        optimal_theta=np.array(theta_opt, copy=True),
        history_r1=history_r1,
        history_r2=history_r2,
        history_angle_deg=history_angle,
        history_raw_energies=history_raw,
        history_cost_energies=history_cost,
    )
