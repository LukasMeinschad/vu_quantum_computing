"""Bond-scan utilities.

This module implements a VQE bond scan for *diatomic* molecules.

For each bond distance it:
1) builds an ElectronicStructureProblem via `modules.molecule.build_molecule_problem`
2) maps the fermionic Hamiltonian to a qubit operator via `modules.qubit_hamiltonian`
3) builds an ansatz circuit via `modules.ansatz.generate_ansatz`
4) runs a single-point VQE via `modules.vqe.run_vqe_single_point`

It can optionally warm-start subsequent geometries with the previously-optimized
parameters and it saves plots into the images/ folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.units import DistanceUnit

from modules.ansatz import generate_ansatz
from modules.molecule import MoleculeSpec, build_molecule_problem
from modules.qubit_hamiltonian import map_to_qubit_hamiltonian
from modules.vqe import run_vqe_single_point


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _diatomic_atom_string(
    atom1: str,
    atom2: str,
    distance: float,
    *,
    axis: str = "x",
    symmetric: bool = True,
) -> str:
    """Create a PySCF atom string for a diatomic molecule.

    Parameters
    ----------
    symmetric:
        If True, place atoms at +/- distance/2 around origin.
        If False, place atoms at 0 and +distance.
    """

    axis = axis.lower()
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


def bond_scan_diatomic_vqe(
    *,
    atom1: str,
    atom2: str,
    distances: np.ndarray | list[float],
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    unit: DistanceUnit = DistanceUnit.ANGSTROM,
    freeze_core: bool = False,
    active_space: tuple[int, int] | None = None,  # (num_electrons, num_spatial_orbitals)
    mapper: str = "JordanWigner",
    ansatz_method: str = "EfficientSU2",
    entanglement: str = "linear",
    reps: int = 1,
    optimizer_builder=None,  # callable like: lambda: COBYLA(maxiter=200)
    maxiter: int = 200,
    backend=None,
    optimization_level: int = 0,
    seed: int = 42,
    warm_start: bool = True,
    plot: bool = True,
    images_dir: str | Path = "images",
    axis: str = "x",
    symmetric_geometry: bool = True,
):
    """Run a VQE bond scan for a diatomic molecule.

    Returns
    -------
    dict
        distances, energies, convergence, optimal_params
    """

    images_dir = _ensure_dir(images_dir)
    if backend is None:
        backend = AerSimulator()
    if optimizer_builder is None:
        optimizer_builder = lambda: COBYLA(maxiter=maxiter)

    distances = np.array(distances, dtype=float)

    scan_energies: list[float] = []
    scan_convergence: list[list[float]] = []
    scan_params: list[np.ndarray] = []

    prev_params = None
    prev_num_params = None

    for i, d in enumerate(distances):
        atom_str = _diatomic_atom_string(
            atom1,
            atom2,
            float(d),
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
        problem_used = build_molecule_problem(
            spec,
            freeze_core=freeze_core,
            active_space=active_space,
            sanitize_active_space=True,
        )

        qubit_hamiltonian = map_to_qubit_hamiltonian(problem_used, mapper=mapper)

        ansatz = generate_ansatz(
            qubit_hamiltonian.num_qubits,
            entanglement=entanglement,
            reps=reps,
            method=ansatz_method,
        )

        init = None
        if warm_start and (prev_params is not None) and (prev_num_params == ansatz.num_parameters):
            init = prev_params

        run = run_vqe_single_point(
            problem=problem_used,
            qubit_hamiltonian=qubit_hamiltonian,
            ansatz=ansatz,
            optimizer=optimizer_builder(),
            initial_params=init,
            backend=backend,
            optimization_level=optimization_level,
            seed=seed if init is None else None,
            verbose=False,
        )

        e_final = float(run.result.fun)
        scan_energies.append(e_final)
        scan_convergence.append(list(run.energies))
        scan_params.append(np.array(run.result.x, copy=True))

        prev_params = scan_params[-1]
        prev_num_params = ansatz.num_parameters

        print(f"[{i+1:02d}/{len(distances):02d}] d = {d:.6f} Å  ->  E = {e_final:.10f} Ha")

    if plot:
        # 1) energy vs distance
        plt.figure(figsize=(6, 4))
        plt.plot(distances, scan_energies, marker="o")
        plt.xlabel("Bond distance (Å)")
        plt.ylabel("VQE energy (Ha)")
        plt.title(f"{atom1}{atom2} VQE bond scan ({basis}, {ansatz_method}, reps={reps}, {mapper})")
        plt.grid(True)
        plt.tight_layout()
        out1 = images_dir / f"bond_scan_{atom1}{atom2}_{basis}_{ansatz_method}_reps-{reps}_{mapper}.png"
        plt.savefig(out1, dpi=200)
        plt.close()

        # 2) convergence per distance
        plt.figure(figsize=(8, 6))
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=float(np.min(distances)), vmax=float(np.max(distances)))

        for d, energies in zip(distances, scan_convergence):
            xs = range(len(energies))
            plt.plot(xs, energies, color=cmap(norm(float(d))), linewidth=1.2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        plt.xlabel("Cost function evaluations")
        plt.ylabel("VQE energy (Ha)")
        plt.title(f"VQE convergence per distance ({atom1}{atom2}, {ansatz_method}, reps={reps})")
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Bond distance (Å)")
        plt.grid(True)
        plt.tight_layout()
        out2 = images_dir / f"bond_scan_convergence_{atom1}{atom2}_{basis}_{ansatz_method}_reps-{reps}_{mapper}.png"
        plt.savefig(out2, dpi=200)
        plt.close()

    return {
        "distances": distances,
        "energies": np.array(scan_energies, dtype=float),
        "convergence": scan_convergence,
        "optimal_params": scan_params,
    }
