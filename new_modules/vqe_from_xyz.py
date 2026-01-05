import os
from typing import List, Tuple, Callable, Dict, Optional

import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import MinimumEigensolverResult
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# ---------------------------------------------------------------------------
# Basic XYZ handling for (primarily) diatomic molecules
# ---------------------------------------------------------------------------


def read_xyz(path: str) -> Tuple[List[str], np.ndarray]:
    """Read an XYZ file.

    Parameters
    ----------
    path:
        Path to the .xyz file.

    Returns
    -------
    symbols:
        List of atomic symbols.
    coordinates:
        Cartesian coordinates in Angstrom as an (N, 3) array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"XYZ file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    try:
        n_atoms = int(lines[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("First line of XYZ must be number of atoms") from exc

    if len(lines) < 2 + n_atoms:
        raise ValueError("XYZ file does not contain enough atom lines")

    symbols: List[str] = []
    coords: List[List[float]] = []

    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: '{line}'")
        symbol = parts[0]
        x, y, z = map(float, parts[1:4])
        symbols.append(symbol)
        coords.append([x, y, z])

    return symbols, np.array(coords, dtype=float)


def diatomic_bond_distance(coordinates: np.ndarray) -> float:
    """Return the internuclear distance for a diatomic geometry.

    Assumes the first two atoms define the bond of interest.
    """
    if coordinates.shape[0] < 2:
        raise ValueError("Diatomic distance requires at least two atoms in XYZ")

    r1 = coordinates[0]
    r2 = coordinates[1]
    return float(np.linalg.norm(r2 - r1))


def build_pyscf_driver_for_diatomic(
    symbols: List[str],
    distance: float,
    basis: str,
    charge: int,
    spin: int,
) -> PySCFDriver:
    """Build a PySCFDriver for a diatomic molecule at a given distance.

    The two atoms are placed along the z-axis, centered at the origin.
    Translation does not affect the energies, only the internuclear distance.
    """
    if len(symbols) < 2:
        raise ValueError("At least two atoms required for diatomic driver")

    s1, s2 = symbols[0], symbols[1]
    half = 0.5 * distance

    atom_str = (
        f""" {s1} 0.000000 0.000000 {-half:.8f}\n{s2} 0.000000 0.000000 {half:.8f} """
    )

    driver = PySCFDriver(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
    )
    return driver


# ---------------------------------------------------------------------------
# Qiskit Nature problem construction from XYZ
# ---------------------------------------------------------------------------


def build_electronic_structure_problem_from_xyz(
    xyz_path: str,
    distance: Optional[float],
    basis: str,
    charge: int,
    spin: int,
):
    """Create an ElectronicStructureProblem for a (typically) diatomic XYZ.

    If ``distance`` is None, the distance between the first two atoms in the
    XYZ is used. Otherwise, the first two atoms are re-placed at the specified
    internuclear separation along the z-axis.

    Returns
    -------
    molecule_problem:
        Qiskit Nature electronic structure problem.
    symbols:
        List of atomic symbols from the XYZ.
    distance_used:
        Internuclear distance actually used (in Angstrom).
    """
    symbols, coords = read_xyz(xyz_path)
    if distance is None:
        distance = diatomic_bond_distance(coords)

    driver = build_pyscf_driver_for_diatomic(
        symbols=symbols,
        distance=distance,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    molecule_problem = driver.run()
    return molecule_problem, symbols, float(distance)


def apply_active_space(
    molecule_problem,
    num_electrons: Optional[int],
    num_spatial_orbitals: Optional[int],
):
    """Optionally reduce the problem to an active space.

    If ``num_electrons`` and ``num_spatial_orbitals`` are both None, the
    untransformed problem is returned.
    """
    if num_electrons is None and num_spatial_orbitals is None:
        return molecule_problem

    if num_electrons is None or num_spatial_orbitals is None:
        raise ValueError(
            "Both num_electrons and num_spatial_orbitals must be provided "
            "for an active space transformation."
        )

    transformer = ActiveSpaceTransformer(
        num_electrons=num_electrons,
        num_spatial_orbitals=num_spatial_orbitals,
    )
    return transformer.transform(molecule_problem)


def build_qubit_hamiltonian(reduced_problem):
    """Construct the qubit Hamiltonian, ansatz and observables as in the notebook."""
    # Second-quantized Hamiltonian
    second_q_hamiltonian = reduced_problem.second_q_ops()[0]

    # Parity mapping with particle number information (enables qubit tapering)
    parity_mapper = ParityMapper(num_particles=reduced_problem.num_particles)
    qubit_op = parity_mapper.map(second_q_hamiltonian)

    # Variational ansatz
    ansatz = EfficientSU2(num_qubits=qubit_op.num_qubits, entanglement="linear", reps=1)

    # Backend and pass manager
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    isa_ansatz = pm.run(ansatz)

    # Match observable layout to transpiled ansatz
    isa_observables = qubit_op.apply_layout(isa_ansatz.layout)

    return isa_ansatz, isa_observables, reduced_problem


# ---------------------------------------------------------------------------
# Energy evaluation and optimization
# ---------------------------------------------------------------------------


def interpret_exp_val(exp_val, problem) -> float:
    """Interpret an expectation value as a physical total energy.

    Mirrors the logic from the notebook: wrap the expectation value in a
    MinimumEigensolverResult and use the problem's interpret method.
    """
    sol = MinimumEigensolverResult()
    sol.eigenvalue = np.real(exp_val)
    return float(problem.interpret(sol).total_energies[0])


def make_energy_cost_function(
    isa_ansatz,
    isa_observables,
    reduced_problem,
    estimator: Optional[EstimatorV2],
) -> Callable[[np.ndarray], float]:
    """Create an Estimator-based energy cost function.

    Parameters mirror the construction used throughout the notebook.
    """
    if estimator is None:
        estimator = EstimatorV2()

    def energy_cost(params: np.ndarray) -> float:
        estimator_job = estimator.run([(isa_ansatz, isa_observables, params)])
        estimator_exp_val = estimator_job.result()[0].data.evs
        return interpret_exp_val(estimator_exp_val, reduced_problem)

    return energy_cost


def minimize_vqe_energy(
    energy_cost: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    maxiter: int,
    tol: float,
    callback: Optional[Callable[[np.ndarray, float], None]],
) -> Dict[str, object]:
    """Run COBYLA to minimize the provided VQE energy function.

    Parameters
    ----------
    energy_cost:
        Callable returning energy for a parameter vector.
    initial_params:
        Initial ansatz parameters.
    maxiter:
        Maximum number of COBYLA iterations.
    tol:
        Optimizer tolerance.
    callback:
        Optional function ``callback(params, energy)`` called after each
        energy evaluation performed inside the optimizer callback.
    """

    def _optimizer_callback(xk: np.ndarray) -> None:
        value = energy_cost(xk)
        if callback is not None:
            callback(xk, value)

    optimizer = COBYLA(maxiter=maxiter, callback=_optimizer_callback, tol=tol)
    result = optimizer.minimize(energy_cost, initial_params)

    return {
        "energy": float(result.fun),
        "params": np.array(result.x, dtype=float),
        "optimizer_result": result,
    }


# ---------------------------------------------------------------------------
# High-level workflows: bond scan and joint optimization
# ---------------------------------------------------------------------------


def vqe_bond_scan_from_xyz(
    xyz_path: str,
    distances: np.ndarray,
    basis: str,
    charge: int,
    spin: int,
    num_electrons: Optional[int],
    num_spatial_orbitals: Optional[int],
    maxiter: int,
    tol: float,
    warm_start: bool,
    random_seed: Optional[int],
) -> Dict[str, object]:
    """Perform a VQE bond scan for a diatomic molecule specified by an XYZ file.

    Parameters
    ----------
    xyz_path:
        Path to the diatomic XYZ file.
    distances:
        1D array of internuclear distances (Angstrom) to scan.
    basis, charge, spin:
        Electronic structure settings for the PySCFDriver.
    num_electrons, num_spatial_orbitals:
        Optional active space specification. If omitted, the full problem is
        used. If provided, both must be set.
    maxiter, tol:
        COBYLA optimizer settings.
    warm_start:
        If True, use the optimal parameters from the previous distance as the
        initial guess for the next distance.
    random_seed:
        Seed for the random initial parameters at the first distance.

    Returns
    -------
    A dictionary containing
    - "distances": array of distances
    - "energies": list of optimal VQE energies
    - "convergence": list of per-distance energy traces
    - "final_params": optimal parameters at the last distance
    """
    distances = np.asarray(distances, dtype=float)

    if random_seed is not None:
        np.random.seed(random_seed)

    scan_energies: List[float] = []
    scan_convergence: List[List[float]] = []
    prev_opt_params: Optional[np.ndarray] = None

    for d in distances:
        molecule_problem, _symbols, _ = build_electronic_structure_problem_from_xyz(
            xyz_path=xyz_path,
            distance=float(d),
            basis=basis,
            charge=charge,
            spin=spin,
        )

        reduced_problem = apply_active_space(
            molecule_problem,
            num_electrons=num_electrons,
            num_spatial_orbitals=num_spatial_orbitals,
        )

        isa_ansatz, isa_observables, reduced_problem = build_qubit_hamiltonian(
            reduced_problem
        )
        energy_cost = make_energy_cost_function(
            isa_ansatz=isa_ansatz,
            isa_observables=isa_observables,
            reduced_problem=reduced_problem,
            estimator=None,
        )

        energies_this_dist: List[float] = []

        def _cb(params: np.ndarray, energy: float) -> None:
            energies_this_dist.append(energy)

        if prev_opt_params is None or not warm_start:
            initial_params = np.random.rand(isa_ansatz.num_parameters)
        else:
            initial_params = prev_opt_params

        res = minimize_vqe_energy(
            energy_cost=energy_cost,
            initial_params=initial_params,
            maxiter=maxiter,
            tol=tol,
            callback=_cb,
        )

        energy = res["energy"]
        params_opt = res["params"]

        scan_energies.append(energy)
        scan_convergence.append(energies_this_dist)
        prev_opt_params = params_opt

    return {
        "distances": distances,
        "energies": scan_energies,
        "convergence": scan_convergence,
        "final_params": prev_opt_params,
    }


def vqe_energy_with_distance_from_xyz(
    xyz_path: str,
    theta: np.ndarray,
    distance: float,
    basis: str,
    charge: int,
    spin: int,
    num_electrons: Optional[int],
    num_spatial_orbitals: Optional[int],
) -> float:
    """Utility: VQE ground-state energy at a given distance for fixed parameters."""
    molecule_problem, _symbols, _ = build_electronic_structure_problem_from_xyz(
        xyz_path=xyz_path,
        distance=float(distance),
        basis=basis,
        charge=charge,
        spin=spin,
    )

    reduced_problem = apply_active_space(
        molecule_problem,
        num_electrons=num_electrons,
        num_spatial_orbitals=num_spatial_orbitals,
    )

    isa_ansatz, isa_observables, reduced_problem = build_qubit_hamiltonian(
        reduced_problem
    )

    if theta.shape[0] != isa_ansatz.num_parameters:
        raise ValueError(
            "Parameter vector length does not match ansatz.num_parameters "
            f"({theta.shape[0]} vs {isa_ansatz.num_parameters})."
        )

    energy_cost = make_energy_cost_function(
        isa_ansatz=isa_ansatz,
        isa_observables=isa_observables,
        reduced_problem=reduced_problem,
        estimator=None,
    )

    return float(energy_cost(theta))


def joint_geometry_ansatz_optimization_from_xyz(
    xyz_path: str,
    initial_distance: float,
    d_min: float,
    d_max: float,
    penalty_strength: float,
    basis: str,
    charge: int,
    spin: int,
    num_electrons: Optional[int],
    num_spatial_orbitals: Optional[int],
    maxiter: int,
    tol: float,
    random_seed: Optional[int],
) -> Dict[str, object]:
    """Jointly optimize the Be–H-like distance and ansatz parameters.

    This follows the logic of the joint optimization section of the notebook,
    generalized to any diatomic specified by an XYZ file. The last element of
    the optimization vector is the bond distance; the remaining entries are the
    ansatz parameters.

    Returns a dictionary with the optimized parameters, distance, energy, and
    evaluation history.
    """
    # Use a reference geometry to determine ansatz size
    ref_problem, _symbols, _ = build_electronic_structure_problem_from_xyz(
        xyz_path=xyz_path,
        distance=float(initial_distance),
        basis=basis,
        charge=charge,
        spin=spin,
    )

    ref_reduced_problem = apply_active_space(
        ref_problem,
        num_electrons=num_electrons,
        num_spatial_orbitals=num_spatial_orbitals,
    )

    ref_second_q_hamiltonian = ref_reduced_problem.second_q_ops()[0]
    ref_parity_mapper = ParityMapper(num_particles=ref_reduced_problem.num_particles)
    ref_qubit_op = ref_parity_mapper.map(ref_second_q_hamiltonian)

    ref_ansatz = EfficientSU2(
        num_qubits=ref_qubit_op.num_qubits,
        entanglement="linear",
        reps=1,
    )
    num_theta = ref_ansatz.num_parameters

    if random_seed is not None:
        np.random.seed(random_seed)

    joint_eval_count = 0
    joint_raw_energies: List[float] = []
    joint_penalized_energies: List[float] = []
    joint_distances: List[float] = []

    def vqe_energy_theta_d(theta: np.ndarray, distance: float) -> float:
        return vqe_energy_with_distance_from_xyz(
            xyz_path=xyz_path,
            theta=theta,
            distance=distance,
            basis=basis,
            charge=charge,
            spin=spin,
            num_electrons=num_electrons,
            num_spatial_orbitals=num_spatial_orbitals,
        )

    def joint_cost(x: np.ndarray) -> float:
        nonlocal joint_eval_count

        theta = x[:-1]
        distance = float(x[-1])

        raw_energy = vqe_energy_theta_d(theta, distance)
        energy = raw_energy
        penalty = 0.0

        if distance < d_min:
            penalty = penalty_strength * (d_min - distance) ** 2
        elif distance > d_max:
            penalty = penalty_strength * (distance - d_max) ** 2

        energy += penalty

        joint_eval_count += 1
        joint_raw_energies.append(raw_energy)
        joint_penalized_energies.append(energy)
        joint_distances.append(distance)

        return energy

    # Initial parameters: random ansatz parameters plus starting distance
    initial_theta = np.random.rand(num_theta)
    x0 = np.concatenate([initial_theta, [float(initial_distance)]])

    optimizer = SLSQP(maxiter=maxiter, ftol=tol)
    opt_result = optimizer.minimize(joint_cost, x0)

    opt_theta = np.array(opt_result.x[:-1], dtype=float)
    opt_distance = float(opt_result.x[-1])
    opt_energy = float(opt_result.fun)

    return {
        "opt_theta": opt_theta,
        "opt_distance": opt_distance,
        "opt_energy": opt_energy,
        "raw_energies": joint_raw_energies,
        "penalized_energies": joint_penalized_energies,
        "distances": joint_distances,
        "eval_count": joint_eval_count,
        "optimizer_result": opt_result,
    }
