from __future__ import annotations

from pathlib import Path
import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper

import matplotlib.pyplot as plt

from modules.molecule import MoleculeSpec, build_molecule_problem
from modules.qubit_hamiltonian import map_to_qubit_hamiltonian
from modules.ansatz import build_uccsd_ansatz
from modules.vqe import run_vqe_single_point

from modules.bond_scan import bond_scan_diatomic_vqe
from modules.joint_optimization import (
    joint_optimize_diatomic_bond_length,
    joint_optimize_water_geometry,
)


if __name__ == "__main__":
    Path("images").mkdir(parents=True, exist_ok=True)

    backend = AerSimulator()
    jw_mapper = JordanWignerMapper()

    # -------------------------------------------------------------------------
    # Ideal (exact) vs incrementally noisier Aer simulation for H2 (single-point)
    # -------------------------------------------------------------------------

    def make_noisy_estimator(
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

    # Single-point H2 at ~0.74 Å (symmetric on z-axis)
    h2_spec = MoleculeSpec(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    h2_problem = build_molecule_problem(
        h2_spec,
        freeze_core=False,
        active_space=(2, 2),
        sanitize_active_space=True,
    )
    h2_qubit_hamiltonian = map_to_qubit_hamiltonian(h2_problem, mapper="JordanWigner")
    h2_ansatz = build_uccsd_ansatz(
        h2_problem, qubit_mapper=jw_mapper, initial_state=None, reps=1
    )

    # Fair comparison: same initial parameters for all runs
    rng = np.random.default_rng(42)
    x0 = rng.random(h2_ansatz.num_parameters)

    maxiter = 120
    optimizer = COBYLA(maxiter=maxiter)

    # Ideal baseline
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

    # Incremental noise scales (0.0 would duplicate ideal; start at small >0)
    noise_scales = [0.25, 0.5, 1.0, 2.0, 4.0]

    noisy_runs = {}
    noisy_finals = []

    for s in noise_scales:
        noisy_estimator = make_noisy_estimator(scale=s, p1_base=0.001, p2_base=0.01)
        print(f"\n=== H2: noisy VQE (UCCSD), scale={s:g} ===")

        run_noisy = run_vqe_single_point(
            problem=h2_problem,
            qubit_hamiltonian=h2_qubit_hamiltonian,
            ansatz=h2_ansatz,
            optimizer=COBYLA(maxiter=maxiter),  # fresh optimizer per run
            initial_params=x0,
            backend=backend,  # nur fürs Transpiling/PassManager
            optimization_level=0,
            seed=None,
            verbose=False,
            estimator=noisy_estimator,
        )

        e_noisy = float(run_noisy.result.fun)
        noisy_runs[s] = run_noisy
        noisy_finals.append(e_noisy)
        print(f"scale={s:g}: E = {e_noisy:.10f} Ha  (Δ={e_noisy - e_ideal:+.6e} Ha)")

    # --- Plot 1: Convergence curves (ideal vs multiple noise levels)
    plt.figure(figsize=(7, 5))
    plt.plot(run_ideal.energies, label="ideal", linewidth=2.0)
    for s in noise_scales:
        plt.plot(noisy_runs[s].energies, label=f"noisy scale={s:g}", linewidth=1.4)

    plt.xlabel("Cost function evaluations")
    plt.ylabel("Energy (Ha)")
    plt.title("H2 UCCSD VQE: ideal vs increasing noise")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/h2_uccsd_convergence_ideal_vs_noise_sweep.png", dpi=200)
    plt.close()

    # --- Plot 2: Final energy vs noise scale (+ ideal reference line)
    plt.figure(figsize=(6, 4))
    plt.plot(noise_scales, noisy_finals, marker="o", label="noisy final energy")
    plt.axhline(e_ideal, linestyle="--", linewidth=1.5, label="ideal final energy")

    plt.xlabel("Noise scale (multiplier on p1/p2)")
    plt.ylabel("Final VQE energy (Ha)")
    plt.title("H2 UCCSD VQE: final energy vs noise level")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/h2_uccsd_final_energy_vs_noise_scale.png", dpi=200)
    plt.close()

    """  
    This section is for running the desired bond scans for the diatomic molecules
    """

    # Configure the bond-scan(s) you want to run here.
    # Each entry: ((atom1, atom2), active_space)
    scans = [
        (("H", "H"), (2, 2)),
        (("Li", "H"), (2, 3)),
        (("F", "H"), (8, 6)),
    ]

    #    distances = np.linspace(0.5, 1.8, 30)
    #
    #    for (atom1, atom2), active_space in scans:
    #        print(f"\n=== Bond scan for {atom1}{atom2} ===")
    #        bond_scan_diatomic_vqe(
    #            atom1=atom1,
    #            atom2=atom2,
    #            distances=distances,
    #            basis="sto3g",
    #            charge=0,
    #            spin=0,
    #            unit=DistanceUnit.ANGSTROM,
    #            freeze_core=False,
    #            active_space=active_space,
    #            mapper="JordanWigner",
    #            ansatz_method="EfficientSU2",
    #            entanglement="linear",
    #            reps=2,
    #            optimizer_builder=lambda: COBYLA(maxiter=150),
    #            maxiter=150,
    #            backend=backend,
    #            optimization_level=0,
    #            seed=42,
    #            warm_start=True,
    #            plot=True,
    #            images_dir=Path("images"),
    #        )
    #
    """  
    This section is for running joint optimizations of
    (diatomic bond distance + ansatz parameters).
    """

    joint_jobs = [
        # Each entry: ((atom1, atom2), active_space, initial_distance, (d_min, d_max))
        (("H", "H"), (2, 2), 0.74, (0.4, 2.0)),
        (("Li", "H"), (2, 3), 1.60, (0.8, 3.0)),
        (("F", "H"), (8, 6), 0.92, (0.6, 2.0)),
    ]

    #    for (atom1, atom2), active_space, d0, window in joint_jobs:
    #        print(f"\n=== Joint optimization for {atom1}{atom2} ===")
    #        res = joint_optimize_diatomic_bond_length(
    #            atom1=atom1,
    #            atom2=atom2,
    #            initial_distance=d0,
    #            distance_window=window,
    #            penalty_strength=100.0,
    #            symmetric_geometry=True,
    #            basis="sto3g",
    #            charge=0,
    #            spin=0,
    #            freeze_core=False,
    #            active_space=active_space,
    #            mapper="JordanWigner",
    #            ansatz_type="EfficientSU2",
    #            entanglement="linear",
    #            reps=2,
    #            optimizer=COBYLA(maxiter=200),
    #            maxiter=200,
    #            backend=backend,
    #            optimization_level=0,
    #            seed=42,
    #            initial_theta=None,
    #            verbose=True,
    #        )
    #
    #        print(f"Optimized: d = {res.optimal_distance:.6f} Å, E = {res.optimal_energy:.10f} Ha")

    """  
    This section compares H2 joint geometry optimization convergence
    using EfficientSU2 vs UCCSD.
    """

    print("\n=== H2 joint optimization: EfficientSU2 vs UCCSD ===")

    h2_window = (0.4, 2.0)
    h2_d0 = 0.74

    # EfficientSU2 joint optimization
    h2_eff = joint_optimize_diatomic_bond_length(
        atom1="H",
        atom2="H",
        initial_distance=h2_d0,
        distance_window=h2_window,
        penalty_strength=100.0,
        symmetric_geometry=True,
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

    # UCCSD joint optimization
    h2_uccsd = joint_optimize_diatomic_bond_length(
        atom1="H",
        atom2="H",
        initial_distance=h2_d0,
        distance_window=h2_window,
        penalty_strength=100.0,
        symmetric_geometry=True,
        basis="sto3g",
        charge=0,
        spin=0,
        freeze_core=False,
        active_space=(2, 2),
        mapper="JordanWigner",
        ansatz_type="UCCSD",
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

    # Plot energy + distance histories
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
    plt.savefig("images/h2_jointopt_convergence_effsu2_vs_uccsd.png", dpi=200)
    plt.close(fig)

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
        ansatz_kind="UCCSD",
        reps=1,
        optimizer=COBYLA(maxiter=120),
        maxiter=120,
    )

    print(
        "H2O UCCSD optimum       : "
        f"r1={water_uccsd.optimal_r1:.6f} Å, r2={water_uccsd.optimal_r2:.6f} Å, "
        f"angle={water_uccsd.optimal_angle_deg:.3f} deg, E={water_uccsd.optimal_energy:.10f} Ha"
    )
