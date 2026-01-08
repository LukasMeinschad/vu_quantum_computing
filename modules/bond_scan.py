"""Bond-scan utilities.

This module implements a VQE bond scan for *diatomic* molecules.

For each bond distance it:
1) builds an ElectronicStructureProblem via `modules.molecule.build_molecule_problem`
2) maps the fermionic Hamiltonian to a qubit operator via `modules.qubit_hamiltonian`
3) builds an ansatz circuit via `modules.ansatz.build_ansatz` (supports both hardware-efficient and chemistry-inspired)
4) runs a single-point VQE via `modules.vqe.run_vqe_single_point`

It can optionally warm-start subsequent geometries with the previously-optimized
parameters and it saves plots into the images/ folder.

Supported ansatz types:
- Hardware-efficient: EfficientSU2, TwoLocal, RealAmplitudes
- Chemistry-inspired: UCCSD
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA

from modules.ansatz import build_ansatz
from modules.results_io import results_print
from modules.molecule import MoleculeSpec, build_molecule_problem
from modules.qubit_hamiltonian import map_to_qubit_hamiltonian
from modules.vqe import run_vqe_single_point
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)


def bond_scan_diatomic_vqe(
    *,
    atom1: str,
    atom2: str,
    distances: np.ndarray | list[float],
    basis: str,
    charge: int,
    spin: int,
    freeze_core: bool,
    active_space: tuple[int, int] | None,  # (num_electrons, num_spatial_orbitals)
    mapper: str,
    ansatz_method: str,
    entanglement: str,
    reps: int,
    optimizer_builder,  # callable like: lambda: COBYLA(maxiter=200)
    maxiter: int,
    backend,
    optimization_level: int,
    seed: int,
    warm_start: bool,
):
    """Run a VQE bond scan for a diatomic molecule.

    For each bond distance, builds the molecular Hamiltonian, maps it to qubits,
    constructs an ansatz, and runs VQE optimization. Optionally warm-starts each
    geometry with the previous optimal parameters if the ansatz dimension is unchanged.

    Parameters
    ----------
    atom1 : str
        Symbol of the first atom (e.g., 'H', 'Li').
    atom2 : str
        Symbol of the second atom.
    distances : np.ndarray | list[float]
        Bond distances (in Angstroms) to scan.
    basis : str
        Basis set for PySCF (e.g., 'sto3g', '6-31g').
    charge : int
        Total molecular charge.
    spin : int
        Number of unpaired electrons (2S).
    freeze_core : bool
        Whether to freeze core orbitals.
    active_space : tuple[int, int] | None
        Active space as (num_electrons, num_spatial_orbitals), or None for full space.
    mapper : str
        Qubit mapper: 'JordanWigner', 'BravyiKitaev', or 'Parity'.
    ansatz_method : str
        Ansatz type: 'EfficientSU2', 'TwoLocal', 'RealAmplitudes', or 'UCCSD'.
    entanglement : str
        Entanglement pattern for hardware-efficient ansätze (e.g., 'linear', 'full').
        Ignored for chemistry-inspired ansätze like UCCSD.
    reps : int
        Number of repetitions/layers in the ansatz.
    optimizer_builder : callable
        Function returning an optimizer instance (e.g., lambda: COBYLA(maxiter=200)).
    maxiter : int
        Maximum optimizer iterations (used as fallback if optimizer_builder is None).
    backend : Backend
        Qiskit backend for simulation (e.g., AerSimulator()).
    optimization_level : int
        Qiskit transpiler optimization level (0-3).
    seed : int
        Random seed for initial parameters (ignored when warm-starting).
    warm_start : bool
        If True, initialize each geometry with the previous optimal parameters.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'distances': np.ndarray of bond distances
        - 'energies': np.ndarray of final VQE energies (Ha)
        - 'convergence': list of lists, each containing the energy trajectory
        - 'optimal_params': list of np.ndarray, final parameters for each distance
    """

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
        atom_str = f"{atom1} 0.0 0.0 0.0; {atom2} 0.0 0.0 {float(d)}"

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

        init = None
        if (
            warm_start
            and (prev_params is not None)
            and (prev_num_params == ansatz.num_parameters)
        ):
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

        results_print(
            f"[{i+1:02d}/{len(distances):02d}] d = {d:.6f} Å  ->  E = {e_final:.10f} Ha"
        )

    return {
        "distances": distances,
        "energies": np.array(scan_energies, dtype=float),
        "convergence": scan_convergence,
        "optimal_params": scan_params,
    }
