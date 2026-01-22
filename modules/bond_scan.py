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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.units import DistanceUnit

from modules.ansatz import build_ansatz
from modules.molecule import MoleculeSpec, build_molecule_problem
from modules.qubit_hamiltonian import map_to_qubit_hamiltonian
from modules.vqe import run_vqe_single_point
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    plot: bool,
    images_dir: str | Path,
    use_sampler: bool = True,
    shots: int = 200,
    noisy_sampler: bool = True,
    noise_scale: float = 1.0,
    p1_base: float = 0.001,
    p2_base: float = 0.01,
    readout_error: float = 0.0,
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
    plot : bool
        If True, save energy and convergence plots to images_dir.
    images_dir : str | Path
        Directory for saving plots.
    use_sampler : bool
        If True, evaluates energies via `SamplerV2` with finite shots.
    shots : int
        Number of shots per Pauli term measurement when `use_sampler=True`.
    noisy_sampler : bool
        If True (and `use_sampler=True`), runs the sampler with an Aer noise model.
    noise_scale : float
        Multiplier applied to base noise probabilities.
    p1_base : float
        Base 1-qubit depolarizing probability (scaled by `noise_scale`).
    p2_base : float
        Base 2-qubit depolarizing probability (scaled by `noise_scale`).
    readout_error : float
        Symmetric bit-flip readout error probability per qubit (0 disables).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'distances': np.ndarray of bond distances
        - 'energies': np.ndarray of final VQE energies (Ha)
        - 'convergence': list of lists, each containing the energy trajectory
        - 'optimal_params': list of np.ndarray, final parameters for each distance
    """

    images_dir = ensure_dir(images_dir)

    if backend is None:
        backend = AerSimulator()

    if optimizer_builder is None:
        optimizer_builder = lambda: COBYLA(maxiter=maxiter)

    distances = np.array(distances, dtype=float)

    scan_energies: list[float] = []
    scan_convergence: list[list[float]] = []
    scan_params: list[np.ndarray] = []

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
            sampler=sampler,
            shots=int(shots) if use_sampler else None,
        )

        e_final = float(run.result.fun)
        scan_energies.append(e_final)
        scan_convergence.append(list(run.energies))
        scan_params.append(np.array(run.result.x, copy=True))

        prev_params = scan_params[-1]
        prev_num_params = ansatz.num_parameters

        print(
            f"[{i+1:02d}/{len(distances):02d}] d = {d:.6f} Å  ->  E = {e_final:.10f} Ha"
        )

    if plot:
        # 1) energy vs distance
        plt.figure(figsize=(6, 4))
        plt.plot(distances, scan_energies, marker="o")
        plt.xlabel("Bond distance (Å)")
        plt.ylabel("VQE energy (Ha)")
        plt.title(
            f"{atom1}{atom2} VQE bond scan ({basis}, {ansatz_method}, reps={reps}, {mapper})"
        )
        plt.grid(True)
        plt.tight_layout()
        out1 = (
            images_dir
            / f"bond_scan_{atom1}{atom2}_{basis}_{ansatz_method}_reps-{reps}_{mapper}.png"
        )
        plt.savefig(out1, dpi=200)
        plt.close()

        # 2) convergence per distance
        plt.figure(figsize=(8, 6))
        cmap = plt.cm.viridis
        norm = plt.Normalize(
            vmin=float(np.min(distances)), vmax=float(np.max(distances))
        )

        for d, energies in zip(distances, scan_convergence):
            xs = range(len(energies))
            plt.plot(xs, energies, color=cmap(norm(float(d))), linewidth=1.2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        plt.xlabel("Cost function evaluations")
        plt.ylabel("VQE energy (Ha)")
        plt.title(
            f"VQE convergence per distance ({atom1}{atom2}, {ansatz_method}, reps={reps})"
        )
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Bond distance (Å)")
        plt.grid(True)
        plt.tight_layout()
        out2 = (
            images_dir
            / f"bond_scan_convergence_{atom1}{atom2}_{basis}_{ansatz_method}_reps-{reps}_{mapper}.png"
        )
        plt.savefig(out2, dpi=200)
        plt.close()

    return {
        "distances": distances,
        "energies": np.array(scan_energies, dtype=float),
        "convergence": scan_convergence,
        "optimal_params": scan_params,
    }
