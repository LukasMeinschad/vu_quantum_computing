from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit

from modules.ansatz import build_uccsd_ansatz
from modules.bond_scan import bond_scan_diatomic_vqe
from modules.joint_optimization import (
    joint_optimize_diatomic_bond_length,
    joint_optimize_water_geometry,
)
from modules.molecule import MoleculeSpec, build_molecule_problem
from modules.qubit_hamiltonian import map_to_qubit_hamiltonian
from modules.vqe import run_vqe_single_point


IMAGES_DIR = Path("images")
DATA_DIR = Path("data")


def _make_noisy_estimator(
    *, scale: float, p1_base: float = 0.001, p2_base: float = 0.01
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


def run_h2_noise_benchmark() -> dict[str, Any]:
    """Run the H2 UCCSD noise sweep benchmark and save plots."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()
    jw_mapper = JordanWignerMapper()

    h2_spec = MoleculeSpec(
        atom="H 0 0 -0.37; H 0 0 0.37", basis="sto3g", charge=0, spin=0
    )
    h2_problem = build_molecule_problem(
        h2_spec,
        freeze_core=False,
        active_space=(2, 2),
        sanitize_active_space_flag=True,
    )
    h2_qubit_hamiltonian = map_to_qubit_hamiltonian(h2_problem, mapper="JordanWigner")
    h2_ansatz = build_uccsd_ansatz(
        h2_problem, qubit_mapper=jw_mapper, initial_state=None, reps=1
    )

    rng = np.random.default_rng(42)
    x0 = rng.random(h2_ansatz.num_parameters)

    maxiter = 120
    optimizer = COBYLA(maxiter=maxiter)

    ideal_estimator = EstimatorV2()
    print("\n=== H2: ideal VQE (UCCSD) ===")
    run_ideal = run_vqe_single_point(
        problem=h2_problem,
        qubit_hamiltonian=h2_qubit_hamiltonian,
        ansatz=h2_ansatz,
        optimizer=optimizer,
        initial_params=x0,
        backend=backend,
        optimization_level=0,
        seed=None,
        verbose=False,
        estimator=ideal_estimator,
    )
    e_ideal = float(run_ideal.result.fun)
    print(f"H2 ideal: E = {e_ideal:.10f} Ha")

    noise_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    noisy_runs: dict[float, Any] = {}
    noisy_finals: list[float] = []

    for scale in noise_scales:
        noisy_estimator = _make_noisy_estimator(
            scale=scale, p1_base=0.001, p2_base=0.01
        )
        print(f"\n=== H2: noisy VQE (UCCSD), scale={scale:g} ===")

        run_noisy = run_vqe_single_point(
            problem=h2_problem,
            qubit_hamiltonian=h2_qubit_hamiltonian,
            ansatz=h2_ansatz,
            optimizer=COBYLA(maxiter=maxiter),
            initial_params=x0,
            backend=backend,
            optimization_level=0,
            seed=None,
            verbose=False,
            estimator=noisy_estimator,
        )

        e_noisy = float(run_noisy.result.fun)
        noisy_runs[scale] = run_noisy
        noisy_finals.append(e_noisy)
        print(
            f"scale={scale:g}: E = {e_noisy:.10f} Ha  (Δ={e_noisy - e_ideal:+.6e} Ha)"
        )

    plt.figure(figsize=(7, 5))
    plt.plot(run_ideal.energies, label="ideal", linewidth=2.0)
    for scale in noise_scales:
        plt.plot(
            noisy_runs[scale].energies, label=f"noisy scale={scale:g}", linewidth=1.4
        )
    plt.xlabel("Cost function evaluations")
    plt.ylabel("Energy (Ha)")
    plt.title("H2 UCCSD VQE: ideal vs increasing noise")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "h2_uccsd_convergence_ideal_vs_noise_sweep.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(noise_scales, noisy_finals, marker="o", label="noisy final energy")
    plt.axhline(e_ideal, linestyle="--", linewidth=1.5, label="ideal final energy")
    plt.xlabel("Noise scale (multiplier on p1/p2)")
    plt.ylabel("Final VQE energy (Ha)")
    plt.title("H2 UCCSD VQE: final energy vs noise level")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "h2_uccsd_final_energy_vs_noise_scale.png", dpi=200)
    plt.close()

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "e_ideal": float(e_ideal),
        "noise_scales": [float(s) for s in noise_scales],
        "noisy_finals": [float(e) for e in noisy_finals],
        "ideal_convergence": [float(e) for e in run_ideal.energies],
        "noisy_convergence": {
            str(float(scale)): [float(e) for e in noisy_runs[scale].energies]
            for scale in noise_scales
        },
    }
    with open(DATA_DIR / "h2_noise_benchmark.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return {
        "problem": h2_problem,
        "qubit_hamiltonian": h2_qubit_hamiltonian,
        "ansatz": h2_ansatz,
        "ideal_run": run_ideal,
        "noisy_runs": noisy_runs,
        "noise_scales": noise_scales,
        "noisy_finals": noisy_finals,
        "e_ideal": e_ideal,
    }


def run_h2_joint_comparison() -> dict[str, Any]:
    """Compare joint optimization on H2 using EfficientSU2 vs UCCSD and plot results."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()

    h2_window = (0.4, 2.0)
    h2_d0 = 0.74

    print("\n=== H2 joint optimization: EfficientSU2 vs UCCSD ===")

    h2_eff = joint_optimize_diatomic_bond_length(
        atom1="H",
        atom2="H",
        initial_distance=h2_d0,
        distance_window=h2_window,
        penalty_strength=100.0,
        basis="sto3g",
        charge=0,
        spin=0,
        freeze_core=False,
        active_space=(2, 2),
        mapper="JordanWigner",
        ansatz_type="EfficientSU2",
        entanglement="linear",
        reps=2,
        optimizer=COBYLA(maxiter=80),
        maxiter=80,
        backend=backend,
        optimization_level=0,
        seed=42,
        initial_theta=None,
        verbose=False,
    )

    h2_uccsd = joint_optimize_diatomic_bond_length(
        atom1="H",
        atom2="H",
        initial_distance=h2_d0,
        distance_window=h2_window,
        penalty_strength=100.0,
        basis="sto3g",
        charge=0,
        spin=0,
        freeze_core=False,
        active_space=(2, 2),
        mapper="JordanWigner",
        ansatz_type="UCCSD",
        entanglement="linear",
        reps=1,
        optimizer=COBYLA(maxiter=80),
        maxiter=80,
        backend=backend,
        optimization_level=0,
        seed=42,
        initial_theta=None,
        verbose=False,
    )

    print(
        "EfficientSU2 optimum: "
        f"d={h2_eff.optimal_distance:.6f} Å, E={h2_eff.optimal_energy:.10f} Ha"
    )
    print(
        "UCCSD optimum       : "
        f"d={h2_uccsd.optimal_distance:.6f} Å, E={h2_uccsd.optimal_energy:.10f} Ha"
    )

    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axs[0].plot(
        h2_eff.history_cost_energies, label="EfficientSU2 (cost)", linewidth=1.8
    )
    axs[0].plot(h2_uccsd.history_cost_energies, label="UCCSD (cost)", linewidth=1.8)
    axs[0].set_ylabel("Energy (Ha)")
    axs[0].set_title("H2 joint optimization convergence")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(h2_eff.history_distances, label="EfficientSU2 distance", linewidth=1.6)
    axs[1].plot(h2_uccsd.history_distances, label="UCCSD distance", linewidth=1.6)
    axs[1].set_xlabel("Cost function evaluations")
    axs[1].set_ylabel("Bond distance (Å)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "h2_jointopt_convergence_effsu2_vs_uccsd.png", dpi=200)
    plt.close(fig)

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "effsu2": {
            "optimal_distance": float(h2_eff.optimal_distance),
            "optimal_energy": float(h2_eff.optimal_energy),
            "history_cost_energies": [float(e) for e in h2_eff.history_cost_energies],
            "history_distances": [float(d) for d in h2_eff.history_distances],
        },
        "uccsd": {
            "optimal_distance": float(h2_uccsd.optimal_distance),
            "optimal_energy": float(h2_uccsd.optimal_energy),
            "history_cost_energies": [float(e) for e in h2_uccsd.history_cost_energies],
            "history_distances": [float(d) for d in h2_uccsd.history_distances],
        },
    }
    with open(DATA_DIR / "h2_joint_comparison.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return {
        "eff": h2_eff,
        "uccsd": h2_uccsd,
    }


def run_diatomic_bond_scans() -> dict[str, Any]:
    """Run bond scans for H2, LiH, and HF using a shared configuration."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()

    scans = [
        (("H", "H"), (2, 2)),
        (("Li", "H"), (2, 3)),
        (("H", "F"), (8, 6)),
    ]

    distances = np.linspace(0.5, 1.8, 30)

    results: dict[str, Any] = {}

    for (atom1, atom2), active_space in scans:
        print(f"\n=== Bond scan for {atom1}{atom2} ===")
        res = bond_scan_diatomic_vqe(
            atom1=atom1,
            atom2=atom2,
            distances=distances,
            basis="sto3g",
            charge=0,
            spin=0,
            freeze_core=False,
            active_space=active_space,
            mapper="JordanWigner",
            ansatz_method="EfficientSU2",
            entanglement="linear",
            reps=2,
            optimizer_builder=lambda: COBYLA(maxiter=150),
            maxiter=150,
            backend=backend,
            optimization_level=0,
            seed=42,
            warm_start=True,
            plot=True,
            images_dir=IMAGES_DIR,
        )
        results[f"{atom1}{atom2}"] = res

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {}
    for key, res in results.items():
        data_to_save[key] = {
            "distances": [float(d) for d in res["distances"]],
            "energies": [float(e) for e in res["energies"]],
        }
    with open(DATA_DIR / "diatomic_bond_scans.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return results


def run_h2o_joint_optimization() -> Any:
    """Run H2O joint geometry + ansatz optimization with UCCSD."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()

    print("\n=== Joint optimization for H2O (r1, r2, angle) with UCCSD ===")

    water_kwargs = dict(
        initial_r1=1.0,
        initial_r2=1.0,
        initial_angle_deg=105.0,
        r1_window=(0.7, 1.3),
        r2_window=(0.7, 1.3),
        angle_window_deg=(80.0, 130.0),
        penalty_strength=100.0,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
        freeze_core=True,
        active_space=(8, 6),
        mapper="JordanWigner",
        backend=backend,
        optimization_level=0,
        seed=42,
        initial_theta=None,
        verbose=False,
    )

    water_uccsd = joint_optimize_water_geometry(
        **water_kwargs,
        ansatz_type="UCCSD",
        reps=1,
        optimizer=COBYLA(maxiter=120),
        maxiter=120,
    )

    print(
        "H2O UCCSD optimum       : "
        f"r1={water_uccsd.optimal_r1:.6f} Å, r2={water_uccsd.optimal_r2:.6f} Å, "
        f"angle={water_uccsd.optimal_angle_deg:.3f} deg, E={water_uccsd.optimal_energy:.10f} Ha"
    )

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "optimal_r1": float(water_uccsd.optimal_r1),
        "optimal_r2": float(water_uccsd.optimal_r2),
        "optimal_angle_deg": float(water_uccsd.optimal_angle_deg),
        "optimal_energy": float(water_uccsd.optimal_energy),
        "history_r1": [float(r) for r in water_uccsd.history_r1],
        "history_r2": [float(r) for r in water_uccsd.history_r2],
        "history_angle_deg": [float(a) for a in water_uccsd.history_angle_deg],
        "history_cost_energies": [float(e) for e in water_uccsd.history_cost_energies],
    }
    with open(DATA_DIR / "h2o_joint_optimization.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return water_uccsd
