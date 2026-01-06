"""Test and benchmark helpers.

This module is intentionally lightweight and script-friendly: it provides small
helpers to compare VQE settings and save convergence plots into the project's
images/ directory.

Currently implemented
---------------------
- `compare_ansatz_types_h2`: compares the available ansatz builders from
  `modules.ansatz` on the H2 system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper

from modules.ansatz import (
    build_hartree_fock_initial_state,
    build_uccsd_ansatz,
    generate_ansatz,
)
from modules.molecule import (
    MoleculeSpec,
    build_molecule_problem,
    xyz_file_to_pyscf_atom_string,
)
from modules.qubit_hamiltonian import map_to_qubit_hamiltonian
from modules.vqe import run_vqe_single_point


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_convergence_plot(
    curves: dict[str, list[float]],
    *,
    title: str,
    save_path: str | Path,
    xlabel: str = "Cost function evaluations",
    ylabel: str = "VQE energy (Ha)",
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for label, energies in curves.items():
        if not energies:
            continue
        xs = list(range(1, len(energies) + 1))
        plt.plot(xs, energies, linewidth=1.5, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def compare_ansatz_types_h2(
    *,
    xyz_path: str | Path = Path("test_molecules") / "h2.xyz",
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    active_space: tuple[int, int] = (2, 2),
    mapper_name: str = "JordanWigner",
    entanglement: str = "linear",
    reps: int = 2,
    maxiter: int = 120,
    optimization_level: int = 0,
    seed: int = 42,
    images_dir: str | Path = "images",
    verbose: bool = False,
) -> dict[str, Any]:
    """Compare all ansatz types implemented in modules.ansatz on H2.

    Includes:
    - EfficientSU2
    - TwoLocal
    - RealAmplitudes
    - HartreeFock (evaluated once; no optimization)
    - UCCSD (with HartreeFock initial state)

    Saves a single comparison convergence plot into images/.
    """

    images_dir = ensure_dir(images_dir)
    xyz_path = Path(xyz_path)

    atom_str = xyz_file_to_pyscf_atom_string(xyz_path)
    spec = MoleculeSpec(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    problem = build_molecule_problem(
        spec,
        freeze_core=False,
        active_space=active_space,
        sanitize_active_space=True,
    )

    qubit_hamiltonian = map_to_qubit_hamiltonian(problem, mapper=mapper_name)

    # For chemistry ansatz we need the mapper object as well
    jw_mapper = JordanWignerMapper()

    curves: dict[str, list[float]] = {}
    results: dict[str, Any] = {}

    # Hardware-efficient ansatzes
    for type in ("EfficientSU2", "TwoLocal", "RealAmplitudes"):
        ansatz = generate_ansatz(
            num_qubits=qubit_hamiltonian.num_qubits,
            method=type,
            reps=reps,
            entanglement=entanglement,
        )
        run = run_vqe_single_point(
            problem=problem,
            qubit_hamiltonian=qubit_hamiltonian,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=maxiter),
            backend=None,
            optimization_level=optimization_level,
            seed=seed,
            verbose=verbose,
        )
        curves[f"{type} (reps={reps})"] = run.energies
        results[type] = run

    # Hartree-Fock (no parameters) – evaluated once
    hf = build_hartree_fock_initial_state(problem, qubit_mapper=jw_mapper)
    hf_run = run_vqe_single_point(
        problem=problem,
        qubit_hamiltonian=qubit_hamiltonian,
        ansatz=hf,
        optimizer=COBYLA(maxiter=maxiter),
        backend=None,
        optimization_level=optimization_level,
        seed=seed,
        verbose=verbose,
    )
    curves["HartreeFock (eval once)"] = hf_run.energies
    results["HartreeFock"] = hf_run

    # UCCSD with HF initial state
    uccsd = build_uccsd_ansatz(problem, qubit_mapper=jw_mapper, reps=reps)
    uccsd_run = run_vqe_single_point(
        problem=problem,
        qubit_hamiltonian=qubit_hamiltonian,
        ansatz=uccsd,
        optimizer=COBYLA(maxiter=maxiter),
        backend=None,
        optimization_level=optimization_level,
        seed=seed,
        verbose=verbose,
    )
    curves[f"UCCSD (reps={reps})"] = uccsd_run.energies
    results["UCCSD"] = uccsd_run

    plot_path = _save_convergence_plot(
        curves,
        title=f"H2 ansatz comparison (basis={basis}, mapper={mapper_name})",
        save_path=images_dir / f"h2_ansatz_comparison_{basis}_{mapper_name}.png",
    )

    return {
        "problem": problem,
        "qubit_hamiltonian": qubit_hamiltonian,
        "curves": curves,
        "plot_path": plot_path,
        "results": results,
    }


def compare_optimizers_h2_uccsd(
    *,
    xyz_path: str | Path = Path("test_molecules") / "h2.xyz",
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    active_space: tuple[int, int] = (2, 2),
    mapper_name: str = "JordanWigner",
    uccsd_reps: int = 1,
    maxiter: int = 120,
    optimization_level: int = 0,
    seed: int = 42,
    images_dir: str | Path = "images",
    verbose: bool = False,
) -> dict[str, Any]:
    """Compare different optimizers on *the same* H2 UCCSD VQE setup.

    This is meant to show how optimizer choice changes convergence behavior.

    Notes
    -----
    - We use the same initial parameter vector for all optimizers for a fair
      comparison.
    - The stored energies are per cost-function evaluation (not necessarily per
      optimizer iteration for all optimizers).
    """

    from qiskit_algorithms.optimizers import COBYLA, POWELL, SLSQP

    images_dir = ensure_dir(images_dir)
    xyz_path = Path(xyz_path)

    atom_str = xyz_file_to_pyscf_atom_string(xyz_path)
    spec = MoleculeSpec(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    problem = build_molecule_problem(
        spec,
        freeze_core=False,
        active_space=active_space,
        sanitize_active_space=True,
    )

    qubit_hamiltonian = map_to_qubit_hamiltonian(problem, mapper=mapper_name)
    qubit_mapper_obj = JordanWignerMapper()
    uccsd = build_uccsd_ansatz(problem, qubit_mapper=qubit_mapper_obj, reps=uccsd_reps)

    rng = np.random.default_rng(seed)
    x0 = rng.random(uccsd.num_parameters)

    optimizer_builders = {
        "COBYLA": lambda: COBYLA(maxiter=maxiter),
        "SLSQP": lambda: SLSQP(maxiter=maxiter),
        "POWELL": lambda: POWELL(maxiter=maxiter),
    }

    curves: dict[str, list[float]] = {}
    results: dict[str, Any] = {}

    for opt_name, build_opt in optimizer_builders.items():
        run = run_vqe_single_point(
            problem=problem,
            qubit_hamiltonian=qubit_hamiltonian,
            ansatz=uccsd,
            optimizer=build_opt(),
            initial_params=np.array(x0, copy=True),
            backend=None,
            optimization_level=optimization_level,
            seed=None,
            verbose=verbose,
        )
        curves[opt_name] = run.energies
        results[opt_name] = run

    plot_path = _save_convergence_plot(
        curves,
        title=f"H2 UCCSD optimizer comparison (basis={basis}, mapper={mapper_name})",
        save_path=images_dir
        / f"h2_uccsd_optimizer_comparison_{basis}_{mapper_name}.png",
    )

    return {
        "problem": problem,
        "qubit_hamiltonian": qubit_hamiltonian,
        "ansatz": uccsd,
        "curves": curves,
        "plot_path": plot_path,
        "results": results,
    }


def compare_entanglement_and_reps_h2(
    *,
    xyz_path: str | Path = Path("test_molecules") / "h2.xyz",
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    active_space: tuple[int, int] = (2, 2),
    mapper_name: str = "JordanWigner",
    method: str = "EfficientSU2",
    entanglements: tuple[str, ...] = ("linear", "full"),
    reps_list: tuple[int, ...] = (1, 2, 3),
    maxiter: int = 120,
    optimization_level: int = 0,
    seed: int = 42,
    images_dir: str | Path = "images",
    verbose: bool = False,
) -> dict[str, Any]:
    """Compare the influence of entanglement pattern and reps on convergence.

    This uses a hardware-efficient ansatz (default: EfficientSU2) because those
    settings are defined there.
    """

    images_dir = ensure_dir(images_dir)
    xyz_path = Path(xyz_path)

    atom_str = xyz_file_to_pyscf_atom_string(xyz_path)
    spec = MoleculeSpec(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    problem = build_molecule_problem(
        spec,
        freeze_core=False,
        active_space=active_space,
        sanitize_active_space=True,
    )

    qubit_hamiltonian = map_to_qubit_hamiltonian(problem, mapper=mapper_name)

    curves: dict[str, list[float]] = {}
    results: dict[str, Any] = {}

    for ent in entanglements:
        for reps in reps_list:
            label = f"{method}: ent={ent}, reps={reps}"
            ansatz = generate_ansatz(
                num_qubits=qubit_hamiltonian.num_qubits,
                method=method,
                reps=reps,
                entanglement=ent,
            )

            # Seeded initialization per configuration (parameter count changes with reps)
            rng = np.random.default_rng(seed)
            x0 = rng.random(ansatz.num_parameters)

            run = run_vqe_single_point(
                problem=problem,
                qubit_hamiltonian=qubit_hamiltonian,
                ansatz=ansatz,
                optimizer=COBYLA(maxiter=maxiter),
                initial_params=x0,
                backend=None,
                optimization_level=optimization_level,
                seed=None,
                verbose=verbose,
            )

            curves[label] = run.energies
            results[label] = run

    plot_path = _save_convergence_plot(
        curves,
        title=f"H2 {method}: entanglement & reps comparison (basis={basis}, mapper={mapper_name})",
        save_path=images_dir
        / f"h2_{method}_entanglement_reps_{basis}_{mapper_name}.png",
    )

    return {
        "problem": problem,
        "qubit_hamiltonian": qubit_hamiltonian,
        "curves": curves,
        "plot_path": plot_path,
        "results": results,
    }
