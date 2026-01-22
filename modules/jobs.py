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
from modules.vqe import run_vqe_single_point, make_noisy_estimator


IMAGES_DIR = Path("images")
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


def run_h2_noise_benchmark() -> dict[str, Any]:
    """Run the H2 UCCSD noise sweep benchmark and save plots."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    h2_settings = {
        "basis": "sto3g",
        "freeze_core": False,
        "active_space": (2, 2),
        "mapper": "JordanWigner",
        "ansatz_reps": 1,
        "initial_params_seed": 42,
        "optimizer_maxiter": 120,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-6,
    }

    backend = AerSimulator()
    jw_mapper = JordanWignerMapper()

    h2_spec = MoleculeSpec(
        atom="H 0 0 -0.37; H 0 0 0.37", basis=h2_settings["basis"], charge=0, spin=0
    )
    h2_problem = build_molecule_problem(
        h2_spec,
        freeze_core=h2_settings["freeze_core"],
        active_space=h2_settings["active_space"],
        sanitize_active_space_flag=True,
    )
    h2_qubit_hamiltonian = map_to_qubit_hamiltonian(
        h2_problem, mapper=h2_settings["mapper"]
    )
    h2_ansatz = build_uccsd_ansatz(
        h2_problem,
        qubit_mapper=jw_mapper,
        initial_state=None,
        reps=h2_settings["ansatz_reps"],
    )

    rng = np.random.default_rng(h2_settings["initial_params_seed"])
    x0 = rng.random(h2_ansatz.num_parameters)

    maxiter = h2_settings["optimizer_maxiter"]
    optimizer = COBYLA(
        maxiter=maxiter,
        rhobeg=h2_settings["optimizer_rhobeg"],
        tol=h2_settings["optimizer_tol"],
    )

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
        noisy_estimator = make_noisy_estimator(scale=scale, p1_base=0.001, p2_base=0.01)
        print(f"\n=== H2: noisy VQE (UCCSD), scale={scale:g} ===")

        run_noisy = run_vqe_single_point(
            problem=h2_problem,
            qubit_hamiltonian=h2_qubit_hamiltonian,
            ansatz=h2_ansatz,
            optimizer=COBYLA(
                maxiter=maxiter,
                rhobeg=h2_settings["optimizer_rhobeg"],
                tol=h2_settings["optimizer_tol"],
            ),
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
        "metadata": {
            "basis": h2_spec.basis,
            "mapper": h2_settings["mapper"],
            "ansatz": "UCCSD",
            "reps": h2_ansatz.reps,
            "optimizer": optimizer_metadata(optimizer),
            "active_space": h2_settings["active_space"],
            "freeze_core": h2_settings["freeze_core"],
            "backend": "AerSimulator",
            "initial_params_seed": h2_settings["initial_params_seed"],
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

    shared = {
        "atom1": "H",
        "atom2": "H",
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "freeze_core": False,
        "active_space": (2, 2),
        "mapper": "JordanWigner",
        "entanglement": "linear",
        "penalty_strength": 100.0,
        "distance_window": (0.4, 2.0),
        "seed": 42,
        "maxiter": 80,
        "rhobeg": 1.0,
        "tol": 1e-6,
    }

    backend = AerSimulator()

    h2_window = shared["distance_window"]
    h2_d0 = 0.74

    print("\n=== H2 joint optimization: EfficientSU2 vs UCCSD ===")

    eff_optimizer = COBYLA(
        maxiter=shared["maxiter"], rhobeg=shared["rhobeg"], tol=shared["tol"]
    )

    h2_eff = joint_optimize_diatomic_bond_length(
        atom1=shared["atom1"],
        atom2=shared["atom2"],
        initial_distance=h2_d0,
        distance_window=shared["distance_window"],
        penalty_strength=shared["penalty_strength"],
        basis=shared["basis"],
        charge=shared["charge"],
        spin=shared["spin"],
        freeze_core=shared["freeze_core"],
        active_space=shared["active_space"],
        mapper=shared["mapper"],
        ansatz_type="EfficientSU2",
        entanglement=shared["entanglement"],
        reps=2,
        optimizer=eff_optimizer,
        maxiter=shared["maxiter"],
        backend=backend,
        optimization_level=0,
        seed=shared["seed"],
        initial_theta=None,
        verbose=False,
    )

    uccsd_optimizer = COBYLA(
        maxiter=shared["maxiter"], rhobeg=shared["rhobeg"], tol=shared["tol"]
    )

    h2_uccsd = joint_optimize_diatomic_bond_length(
        atom1=shared["atom1"],
        atom2=shared["atom2"],
        initial_distance=h2_d0,
        distance_window=shared["distance_window"],
        penalty_strength=shared["penalty_strength"],
        basis=shared["basis"],
        charge=shared["charge"],
        spin=shared["spin"],
        freeze_core=shared["freeze_core"],
        active_space=shared["active_space"],
        mapper=shared["mapper"],
        ansatz_type="UCCSD",
        entanglement=shared["entanglement"],
        reps=1,
        optimizer=uccsd_optimizer,
        maxiter=shared["maxiter"],
        backend=backend,
        optimization_level=0,
        seed=shared["seed"],
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
            "metadata": {
                "basis": "sto3g",
                "mapper": shared["mapper"],
                "ansatz": "EfficientSU2",
                "reps": 2,
                "optimizer": optimizer_metadata(eff_optimizer),
                "active_space": shared["active_space"],
                "freeze_core": shared["freeze_core"],
                "penalty_strength": shared["penalty_strength"],
                "distance_window": shared["distance_window"],
                "seed": shared["seed"],
            },
        },
        "uccsd": {
            "optimal_distance": float(h2_uccsd.optimal_distance),
            "optimal_energy": float(h2_uccsd.optimal_energy),
            "history_cost_energies": [float(e) for e in h2_uccsd.history_cost_energies],
            "history_distances": [float(d) for d in h2_uccsd.history_distances],
            "metadata": {
                "basis": "sto3g",
                "mapper": shared["mapper"],
                "ansatz": "UCCSD",
                "reps": 1,
                "optimizer": optimizer_metadata(uccsd_optimizer),
                "active_space": shared["active_space"],
                "freeze_core": shared["freeze_core"],
                "penalty_strength": shared["penalty_strength"],
                "distance_window": shared["distance_window"],
                "seed": shared["seed"],
            },
        },
    }
    with open(DATA_DIR / "h2_joint_comparison.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return {
        "eff": h2_eff,
        "uccsd": h2_uccsd,
    }


def run_h2_joint_optimization() -> dict[str, Any]:
    """Run H2 joint optimization with EfficientSU2."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    backend = AerSimulator()

    h2_config = {
        "atom1": "H",
        "atom2": "H",
        "initial_distance": 1.0,
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
        "reps": 1,
        "optimizer_maxiter": 300,
        "optimizer_rhobeg": 0.5,
        "optimizer_tol": 1e-4,
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

def run_convergence_benchmark() -> dict[str, Any]:

    # Settings
    atom1 = "H"
    atom2 = "H"
    distance = 0.74
    basis = "sto3g"
    charge = 0
    spin = 0
    freeze_core = False
    active_space = (2, 2)
    mapper = "JordanWigner"
    ansatz_method = "UCCSD"
    entanglement = "linear"
    reps = 5
    optimization_level = 3
    seed = 42
    use_sampler = False
    noisy_sampler = False
    shots = 1024
    noise_scale = 1.0
    p1_base = 0.001
    p2_base = 0.01
    readout_error = 0.0
    data_file_name = "vqe_conv_uccsd_jw_5.json"
    
    from qiskit_aer.primitives import SamplerV2
    from modules.ansatz import build_ansatz
    from qiskit_nature.second_q.mappers import (
        JordanWignerMapper,
        ParityMapper,
        BravyiKitaevMapper,
    )


    backend = AerSimulator()
    optimizer_builder = lambda: COBYLA(maxiter=10000, tol=1e-4)


    sampler = None
    if use_sampler:
        if int(shots) <= 0:
            raise ValueError("shots must be a positive integer")

        if noisy_sampler:
            from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

            noise_model = NoiseModel()
            p1 = float(noise_scale) * float(p1_base)
            p2 = float(noise_scale) * float(p2_base)

            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(p1, 1),
                ["x", "sx", "rx", "ry", "rz", "h", "s", "sdg"],
            )
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(p2, 2), ["cx", "cz"]
            )

            if float(readout_error) > 0.0:
                p = float(readout_error)
                ro = ReadoutError([[1 - p, p], [p, 1 - p]])
                noise_model.add_all_qubit_readout_error(ro)

            sampler = SamplerV2(
                options={"backend_options": {"noise_model": noise_model}}
            )
        else:
            sampler = SamplerV2()

    atom_str = f"{atom1} 0.0 0.0 0.0; {atom2} 0.0 0.0 {float(distance)}"

    spec = MoleculeSpec(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    problem_used = build_molecule_problem(
        spec,
        freeze_core=freeze_core,
        active_space=active_space,
        sanitize_active_space_flag=True,
    )

    qubit_hamiltonian = map_to_qubit_hamiltonian(problem_used, mapper=mapper)

    if ansatz_method in ("EfficientSU2", "TwoLocal", "RealAmplitudes"):
        ansatz = build_ansatz(
            ansatz_method,
            num_qubits=qubit_hamiltonian.num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    else:
        if mapper == "JordanWigner":
            qubit_mapper = JordanWignerMapper()
        elif mapper == "BravyiKitaev":
            qubit_mapper = BravyiKitaevMapper()
        elif mapper == "Parity":
            qubit_mapper = ParityMapper(num_particles=problem_used.num_particles)
        else:
            raise ValueError(f"Unsupported mapper: {mapper}")

        ansatz = build_ansatz(
            ansatz_method,
            problem=problem_used,
            qubit_mapper=qubit_mapper,
            reps=reps,
        )

    run = run_vqe_single_point(
        problem=problem_used,
        qubit_hamiltonian=qubit_hamiltonian,
        ansatz=ansatz,
        optimizer=optimizer_builder(),
        initial_params=None,
        backend=backend,
        optimization_level=optimization_level,
        seed=seed,
        verbose=False,
        sampler=sampler,
        shots=int(shots) if use_sampler else None,
    )
    
    convergence: list[list[float]] = []
    convergence.append(list(run.energies))

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "final_energy": float(run.result.fun),
        "convergence": [[float(e) for e in run.energies]],
        "num_evaluations": len(run.energies),
        "metadata": {
            "molecule": f"{atom1}-{atom2}",
            "distance": float(distance),
            "basis": basis,
            "charge": charge,
            "spin": spin,
            "mapper": mapper,
            "ansatz": ansatz_method,
            "reps": reps,
            "entanglement": entanglement,
            "optimizer": optimizer_metadata(optimizer_builder),
            "active_space": active_space,
            "freeze_core": freeze_core,
            "backend": "AerSimulator",
            "optimization_level": optimization_level,
            "seed": seed,
            "use_sampler": use_sampler,
            "noisy_sampler": noisy_sampler,
            "shots": int(shots) if use_sampler else None,
            "noise_scale": float(noise_scale) if noisy_sampler else None,
            "p1_base": float(p1_base) if noisy_sampler else None,
            "p2_base": float(p2_base) if noisy_sampler else None,
            "readout_error": float(readout_error) if noisy_sampler else None,
        },
    }
    with open(DATA_DIR / data_file_name, "w") as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"\nData saved to {DATA_DIR / data_file_name}")
    
    return {
        "run": run,
        "convergence": convergence,
        "final_energy": run.result.fun,
    }

def run_lih_joint_optimization() -> dict[str, Any]:
    """Run LiH joint optimization with EfficientSU2."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    backend = AerSimulator()

    lih_config = {
        "atom1": "Li",
        "atom2": "H",
        "initial_distance": 1.6,
        "distance_window": (0.8, 3.0),
        "penalty_strength": 100.0,
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "freeze_core": False,
        "active_space": (2, 3),
        "mapper": "JordanWigner",
        "ansatz_type": "UCCSD",
        "entanglement": "linear",
        "reps": 2,
        "optimizer_maxiter": 100,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-6,
        "seed": 42,
    }

    print("\n=== LiH joint optimization ===")
    optimizer = COBYLA(
        maxiter=lih_config["optimizer_maxiter"],
        rhobeg=lih_config["optimizer_rhobeg"],
        tol=lih_config["optimizer_tol"],
    )
    result = joint_optimize_diatomic_bond_length(
        atom1=lih_config["atom1"],
        atom2=lih_config["atom2"],
        initial_distance=lih_config["initial_distance"],
        distance_window=lih_config["distance_window"],
        penalty_strength=lih_config["penalty_strength"],
        basis=lih_config["basis"],
        charge=lih_config["charge"],
        spin=lih_config["spin"],
        freeze_core=lih_config["freeze_core"],
        active_space=lih_config["active_space"],
        mapper=lih_config["mapper"],
        ansatz_type=lih_config["ansatz_type"],
        entanglement=lih_config["entanglement"],
        reps=lih_config["reps"],
        optimizer=optimizer,
        maxiter=lih_config["optimizer_maxiter"],
        backend=backend,
        optimization_level=0,
        seed=lih_config["seed"],
        initial_theta=None,
        verbose=True,
    )

    print(
        f"LiH optimum: d={result.optimal_distance:.6f} Å, E={result.optimal_energy:.10f} Ha"
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
            "basis": lih_config["basis"],
            "mapper": lih_config["mapper"],
            "ansatz": lih_config["ansatz_type"],
            "reps": lih_config["reps"],
            "optimizer": optimizer_metadata(optimizer),
            "active_space": lih_config["active_space"],
            "freeze_core": lih_config["freeze_core"],
            "penalty_strength": lih_config["penalty_strength"],
            "distance_window": lih_config["distance_window"],
            "seed": lih_config["seed"],
        },
    }
    with open(DATA_DIR / "lih_joint_optimization.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return result


def run_hf_joint_optimization() -> dict[str, Any]:
    """Run HF joint optimization with EfficientSU2."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    backend = AerSimulator()

    hf_config = {
        "atom1": "H",
        "atom2": "F",
        "initial_distance": 0.92,
        "distance_window": (0.5, 2.0),
        "penalty_strength": 100.0,
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "freeze_core": True,
        "active_space": (8, 5),
        "mapper": "JordanWigner",
        "ansatz_type": "UCCSD",
        "entanglement": "full",
        "reps": 4,
        "optimizer_maxiter": 500,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-6,
        "seed": 42,
    }

    print("\n=== HF joint optimization ===")
    optimizer = COBYLA(
        maxiter=hf_config["optimizer_maxiter"],
        rhobeg=hf_config["optimizer_rhobeg"],
        tol=hf_config["optimizer_tol"],
    )
    result = joint_optimize_diatomic_bond_length(
        atom1=hf_config["atom1"],
        atom2=hf_config["atom2"],
        initial_distance=hf_config["initial_distance"],
        distance_window=hf_config["distance_window"],
        penalty_strength=hf_config["penalty_strength"],
        basis=hf_config["basis"],
        charge=hf_config["charge"],
        spin=hf_config["spin"],
        freeze_core=hf_config["freeze_core"],
        active_space=hf_config["active_space"],
        mapper=hf_config["mapper"],
        ansatz_type=hf_config["ansatz_type"],
        entanglement=hf_config["entanglement"],
        reps=hf_config["reps"],
        optimizer=optimizer,
        maxiter=hf_config["optimizer_maxiter"],
        backend=backend,
        optimization_level=0,
        seed=hf_config["seed"],
        initial_theta=None,
        verbose=True,
    )

    print(
        f"HF optimum: d={result.optimal_distance:.6f} Å, E={result.optimal_energy:.10f} Ha"
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
            "basis": hf_config["basis"],
            "mapper": hf_config["mapper"],
            "ansatz": hf_config["ansatz_type"],
            "reps": hf_config["reps"],
            "optimizer": optimizer_metadata(optimizer),
            "active_space": hf_config["active_space"],
            "freeze_core": hf_config["freeze_core"],
            "penalty_strength": hf_config["penalty_strength"],
            "distance_window": hf_config["distance_window"],
            "seed": hf_config["seed"],
        },
    }
    with open(DATA_DIR / "hf_joint_optimization.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return result


def run_h2_bond_scan() -> dict[str, Any]:
    """Run bond scan for H2."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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
        "reps": 1,
        "optimizer_maxiter": 150,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-5,
        "seed": 42,
        "distances": np.linspace(0.25, 2.2, 80),
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
        warm_start=True,
        plot=True,
        images_dir=IMAGES_DIR,
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


def run_lih_bond_scan() -> dict[str, Any]:
    """Run bond scan for LiH."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()
    lih_scan_config = {
        "atom1": "Li",
        "atom2": "H",
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "freeze_core": False,
        "active_space": (2, 3),
        "mapper": "JordanWigner",
        "ansatz_method": "UCCSD",
        "entanglement": "linear",
        "reps": 2,
        "optimizer_maxiter": 150,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-6,
        "seed": 42,
        "distances": np.linspace(1.4, 1.9, 10),
    }

    distances = lih_scan_config["distances"]
    distance_grid_label = (
        f"np.linspace({distances[0]:.3f}, {distances[-1]:.3f}, {len(distances)})"
    )
    optimizer_builder = lambda: COBYLA(
        maxiter=lih_scan_config["optimizer_maxiter"],
        rhobeg=lih_scan_config["optimizer_rhobeg"],
        tol=lih_scan_config["optimizer_tol"],
    )

    print("\n=== Bond scan for LiH ===")
    res = bond_scan_diatomic_vqe(
        atom1=lih_scan_config["atom1"],
        atom2=lih_scan_config["atom2"],
        distances=distances,
        basis=lih_scan_config["basis"],
        charge=lih_scan_config["charge"],
        spin=lih_scan_config["spin"],
        freeze_core=lih_scan_config["freeze_core"],
        active_space=lih_scan_config["active_space"],
        mapper=lih_scan_config["mapper"],
        ansatz_method=lih_scan_config["ansatz_method"],
        entanglement=lih_scan_config["entanglement"],
        reps=lih_scan_config["reps"],
        optimizer_builder=optimizer_builder,
        maxiter=lih_scan_config["optimizer_maxiter"],
        backend=backend,
        optimization_level=0,
        seed=lih_scan_config["seed"],
        warm_start=True,
        plot=True,
        images_dir=IMAGES_DIR,
    )

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "distances": [float(d) for d in res["distances"]],
        "energies": [float(e) for e in res["energies"]],
        "metadata": {
            "basis": lih_scan_config["basis"],
            "mapper": lih_scan_config["mapper"],
            "ansatz": lih_scan_config["ansatz_method"],
            "reps": lih_scan_config["reps"],
            "optimizer": optimizer_metadata(optimizer_builder),
            "active_space": lih_scan_config["active_space"],
            "freeze_core": lih_scan_config["freeze_core"],
            "seed": lih_scan_config["seed"],
            "distance_grid": distance_grid_label,
        },
    }
    with open(DATA_DIR / "lih_bond_scan.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return res


def run_hf_bond_scan() -> dict[str, Any]:
    """Run bond scan for HF."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()
    hf_scan_config = {
        "atom1": "H",
        "atom2": "F",
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "freeze_core": True,
        "active_space": (8, 5),
        "mapper": "JordanWigner",
        "ansatz_method": "UCCSD",
        "entanglement": "linear",
        "reps": 2,
        "optimizer_maxiter": 1000,
        "optimizer_rhobeg": 1.0,
        "optimizer_tol": 1e-6,
        "seed": 42,
        "distances": np.linspace(0.5, 1.8, 30),
    }

    distances = hf_scan_config["distances"]
    distance_grid_label = (
        f"np.linspace({distances[0]:.3f}, {distances[-1]:.3f}, {len(distances)})"
    )
    optimizer_builder = lambda: COBYLA(
        maxiter=hf_scan_config["optimizer_maxiter"],
        rhobeg=hf_scan_config["optimizer_rhobeg"],
        tol=hf_scan_config["optimizer_tol"],
    )

    print("\n=== Bond scan for HF ===")
    res = bond_scan_diatomic_vqe(
        atom1=hf_scan_config["atom1"],
        atom2=hf_scan_config["atom2"],
        distances=distances,
        basis=hf_scan_config["basis"],
        charge=hf_scan_config["charge"],
        spin=hf_scan_config["spin"],
        freeze_core=hf_scan_config["freeze_core"],
        active_space=hf_scan_config["active_space"],
        mapper=hf_scan_config["mapper"],
        ansatz_method=hf_scan_config["ansatz_method"],
        entanglement=hf_scan_config["entanglement"],
        reps=hf_scan_config["reps"],
        optimizer_builder=optimizer_builder,
        maxiter=hf_scan_config["optimizer_maxiter"],
        backend=backend,
        optimization_level=0,
        seed=hf_scan_config["seed"],
        warm_start=True,
        plot=True,
        images_dir=IMAGES_DIR,
    )

    # Save data to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "distances": [float(d) for d in res["distances"]],
        "energies": [float(e) for e in res["energies"]],
        "metadata": {
            "basis": hf_scan_config["basis"],
            "mapper": hf_scan_config["mapper"],
            "ansatz": hf_scan_config["ansatz_method"],
            "reps": hf_scan_config["reps"],
            "optimizer": optimizer_metadata(optimizer_builder),
            "active_space": hf_scan_config["active_space"],
            "freeze_core": hf_scan_config["freeze_core"],
            "seed": hf_scan_config["seed"],
            "distance_grid": distance_grid_label,
        },
    }
    with open(DATA_DIR / "hf_bond_scan.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return res


def run_h2o_joint_optimization() -> Any:
    """Run H2O joint geometry + ansatz optimization with UCCSD."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()

    water_config = {
        "initial_r1": 1.0,
        "initial_r2": 1.0,
        "initial_angle_deg": 105.0,
        "r1_window": (0.7, 1.3),
        "r2_window": (0.7, 1.3),
        "angle_window_deg": (80.0, 130.0),
        "penalty_strength": 100.0,
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "unit": DistanceUnit.ANGSTROM,
        "freeze_core": True,
        "active_space": (8, 6),
        "mapper": "JordanWigner",
        "backend": backend,
        "optimization_level": 0,
        "seed": 42,
        "initial_theta": None,
        "verbose": False,
        "ansatz_type": "UCCSD",
        "reps": 1,
        "optimizer_maxiter": 120,
        "optimizer_tol": 1e-6,
        "optimizer_rhobeg": 1.0,
    }

    print("\n=== Joint optimization for H2O (r1, r2, angle) with UCCSD ===")

    water_kwargs = dict(
        initial_r1=water_config["initial_r1"],
        initial_r2=water_config["initial_r2"],
        initial_angle_deg=water_config["initial_angle_deg"],
        r1_window=water_config["r1_window"],
        r2_window=water_config["r2_window"],
        angle_window_deg=water_config["angle_window_deg"],
        penalty_strength=water_config["penalty_strength"],
        basis=water_config["basis"],
        charge=water_config["charge"],
        spin=water_config["spin"],
        unit=water_config["unit"],
        freeze_core=water_config["freeze_core"],
        active_space=water_config["active_space"],
        mapper=water_config["mapper"],
        backend=water_config["backend"],
        optimization_level=water_config["optimization_level"],
        seed=water_config["seed"],
        initial_theta=water_config["initial_theta"],
        verbose=water_config["verbose"],
    )

    water_optimizer = COBYLA(
        maxiter=water_config["optimizer_maxiter"],
        rhobeg=water_config["optimizer_rhobeg"],
        tol=water_config["optimizer_tol"],
    )

    water_uccsd = joint_optimize_water_geometry(
        **water_kwargs,
        ansatz_type=water_config["ansatz_type"],
        reps=water_config["reps"],
        optimizer=water_optimizer,
        maxiter=water_config["optimizer_maxiter"],
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
        "metadata": {
            "basis": water_config["basis"],
            "mapper": water_config["mapper"],
            "ansatz": water_config["ansatz_type"],
            "reps": water_config["reps"],
            "optimizer": optimizer_metadata(water_optimizer),
            "active_space": water_config["active_space"],
            "freeze_core": water_config["freeze_core"],
            "seed": water_config["seed"],
            "penalty_strength": water_config["penalty_strength"],
            "geometry_windows": {
                "r1_window": water_config["r1_window"],
                "r2_window": water_config["r2_window"],
                "angle_window_deg": water_config["angle_window_deg"],
            },
        },
    }
    with open(DATA_DIR / "h2o_joint_optimization.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

    return water_uccsd
