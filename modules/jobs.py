from __future__ import annotations

import json
import numpy as np

from typing import Any
from pathlib import Path
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms.optimizers import COBYLA

from modules.bond_scan import bond_scan_diatomic_vqe
from modules.joint_optimization import joint_optimize_diatomic_bond_length


DATA_DIR = Path("data")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def optimizer_metadata(optimizer_or_builder: Any) -> dict[str, Any]:
    opt_obj = (
        optimizer_or_builder()
        if callable(optimizer_or_builder)
        else optimizer_or_builder
    )
    metadata = {"name": type(opt_obj).__name__}
    settings = getattr(opt_obj, "settings", None)
    if isinstance(settings, dict):
        metadata["settings"] = {str(k): json_safe(v) for k, v in settings.items()}
    return metadata


def make_noisy_estimator(
    *, scale: float, p1_base: float, p2_base: float
) -> EstimatorV2:
    """Create an EstimatorV2 configured with a scaled depolarizing noise model."""
    noise_model = NoiseModel()
    p1 = float(scale) * p1_base
    p2 = float(scale) * p2_base

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p1, 1), ["x", "sx", "rx", "ry", "rz"]
    )
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz"])

    return EstimatorV2(
        options={
            "backend_options": {
                "method": "density_matrix",
                "noise_model": noise_model,
            }
        }
    )


def run_h2_joint_optimization() -> dict[str, Any]:
    """Run H2 joint optimization with EfficientSU2."""

    backend = AerSimulator()

    h2_config = {
        "atom1": "H",
        "atom2": "H",
        "initial_distance": 0.74,
        "distance_window": (0.4, 2.0),
        "penalty_strength": 100.0,
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "freeze_core": False,
        "active_space": (2, 2),
        "mapper": "JordanWigner",
        "ansatz_type": "UCCSD",
        "entanglement": "linear",
        "reps": 2,
        "optimizer_maxiter": 80,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-6,
        "seed": 42,
    }

    print("\n=== H2 joint optimization ===")
    optimizer = COBYLA(
        maxiter=h2_config["optimizer_maxiter"],
        rhobeg=h2_config["optimizer_rhobeg"],
        tol=h2_config["optimizer_tol"],
    )
    result = joint_optimize_diatomic_bond_length(
        atom1=h2_config["atom1"],
        atom2=h2_config["atom2"],
        initial_distance=h2_config["initial_distance"],
        distance_window=h2_config["distance_window"],
        penalty_strength=h2_config["penalty_strength"],
        basis=h2_config["basis"],
        charge=h2_config["charge"],
        spin=h2_config["spin"],
        freeze_core=h2_config["freeze_core"],
        active_space=h2_config["active_space"],
        mapper=h2_config["mapper"],
        ansatz_type=h2_config["ansatz_type"],
        entanglement=h2_config["entanglement"],
        reps=h2_config["reps"],
        optimizer=optimizer,
        maxiter=h2_config["optimizer_maxiter"],
        backend=backend,
        optimization_level=0,
        seed=h2_config["seed"],
        initial_theta=None,
        verbose=True,
    )

    print(
        f"H2 optimum: d={result.optimal_distance:.6f} Å, E={result.optimal_energy:.10f} Ha"
    )

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "optimal_distance": float(result.optimal_distance),
        "optimal_energy": float(result.optimal_energy),
        "history_cost_energies": [float(e) for e in result.history_cost_energies],
        "history_distances": [float(d) for d in result.history_distances],
        "history_raw_energies": [float(e) for e in result.history_raw_energies],
        "metadata": {
            "basis": h2_config["basis"],
            "mapper": h2_config["mapper"],
            "ansatz": h2_config["ansatz_type"],
            "reps": h2_config["reps"],
            "optimizer": optimizer_metadata(optimizer),
            "active_space": h2_config["active_space"],
            "freeze_core": h2_config["freeze_core"],
            "penalty_strength": h2_config["penalty_strength"],
            "distance_window": h2_config["distance_window"],
            "seed": h2_config["seed"],
        },
    }
    with open(DATA_DIR / "h2_joint_optimization.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return result


def run_h2_bond_scan() -> dict[str, Any]:
    """Run bond scan for H2."""

    backend = AerSimulator()

    h2_scan_config = {
        "atom1": "H",
        "atom2": "H",
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "freeze_core": False,
        "active_space": (2, 2),
        "mapper": "JordanWigner",
        "ansatz_method": "UCCSD",
        "entanglement": "linear",
        "reps": 2,
        "optimizer_maxiter": 150,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-6,
        "seed": 42,
        "distances": np.linspace(0.4, 1.0, 30),
    }

    distances = h2_scan_config["distances"]
    distance_grid_label = (
        f"np.linspace({distances[0]:.3f}, {distances[-1]:.3f}, {len(distances)})"
    )
    optimizer_builder = lambda: COBYLA(
        maxiter=h2_scan_config["optimizer_maxiter"],
        rhobeg=h2_scan_config["optimizer_rhobeg"],
        tol=h2_scan_config["optimizer_tol"],
    )

    print("\n=== Bond scan for H2 ===")
    res = bond_scan_diatomic_vqe(
        atom1=h2_scan_config["atom1"],
        atom2=h2_scan_config["atom2"],
        distances=distances,
        basis=h2_scan_config["basis"],
        charge=h2_scan_config["charge"],
        spin=h2_scan_config["spin"],
        freeze_core=h2_scan_config["freeze_core"],
        active_space=h2_scan_config["active_space"],
        mapper=h2_scan_config["mapper"],
        ansatz_method=h2_scan_config["ansatz_method"],
        entanglement=h2_scan_config["entanglement"],
        reps=h2_scan_config["reps"],
        optimizer_builder=optimizer_builder,
        maxiter=h2_scan_config["optimizer_maxiter"],
        backend=backend,
        optimization_level=0,
        seed=h2_scan_config["seed"],
        warm_start=True
    )

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "distances": [float(d) for d in res["distances"]],
        "energies": [float(e) for e in res["energies"]],
        "metadata": {
            "basis": h2_scan_config["basis"],
            "mapper": h2_scan_config["mapper"],
            "ansatz": h2_scan_config["ansatz_method"],
            "reps": h2_scan_config["reps"],
            "optimizer": optimizer_metadata(optimizer_builder),
            "active_space": h2_scan_config["active_space"],
            "freeze_core": h2_scan_config["freeze_core"],
            "seed": h2_scan_config["seed"],
            "distance_grid": distance_grid_label,
        },
    }
    with open(DATA_DIR / "h2_bond_scan.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return res