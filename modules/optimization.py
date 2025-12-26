import numpy as np
import matplotlib

# Use a non-interactive backend to avoid Tkinter GUI issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import modules.hamiltonian as hamiltonian


def cost_func(params, ansatz, H, estimator):
    """
    Cost function for VQE optimization using Estimator primitive

    Args:
        params: Parameter values for the ansatz circuit
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        H: Qubit Hamiltonian as SparsePauliOp
        estimator: Qiskit Estimator primitive object
    Returns:
        energy: Expected energy value for given parameters
    """
    pub = (ansatz, [H], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    return energy


def vqe_single_point(
    ansatz,
    H,
    backend,
    out_file,
    initial_params=None,
    method="COBYLA",
    options={"maxiter": 100},
):
    """
    Run VQE optimization using simulator or real backend

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        H: Qubit Hamiltonian as SparsePauliOp
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        initial_params: Initial parameter values for the ansatz
        method (str): Optimization method (e.g., "COBYLA", "Nelder-Mead", etc.)
        options (dict): Options for the optimizer
    """
    if initial_params is None:
        initial_params = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    # Create simulator backend
    if backend is None:
        backend = AerSimulator()
    elif isinstance(backend, AerSimulator):
        backend_sim = backend
    else:
        # Real backend, create simulator
        backend_sim = AerSimulator.from_backend(backend)

    estimator = BackendEstimatorV2(backend=backend_sim)

    # Transpile the ansatz for the backend
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend
    )  # Make Optimized Circuit, with longer transpile time
    ansatz_isa = pm.run(ansatz)

    with open(out_file, "a") as f:
        f.write("\n=== VQE Single Point Optimization ===\n")
        f.write(f"Backend: {backend_sim.name}\n")
        f.write(f"Method: {method}, Options: {options}\n")
        f.write(f"Initial parameters shape: {initial_params.shape}\n")
        f.write(f"Number of Ansatz parameters: {ansatz.num_parameters}\n")

    # Track optimization progress
    iteration_count = [0]
    energy_history = []

    def callback(xk):
        iteration_count[0] += 1
        energy = cost_func(xk, ansatz_isa, H, estimator)
        energy_history.append(energy)
        if iteration_count[0] % 10 == 0:
            # Log every 10 iterations to output file
            with open(out_file, "a") as f:
                f.write(
                    f"Iteration {iteration_count[0]}: Energy = {energy:.6f} Hartree\n"
                )

    result = minimize(
        fun=cost_func,
        x0=initial_params,
        args=(ansatz_isa, H, estimator),
        method=method,
        options=options,
        callback=callback,
    )

    with open(out_file, "a") as f:
        f.write(f"\nOptimization completed in {iteration_count[0]} iterations.\n")
        f.write(f"Final energy: {result.fun:.6f} Hartree\n")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, iteration_count[0] + 1), energy_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Energy / Hartree")
    plt.title("VQE Energy Convergence")
    plt.grid()
    plt.savefig("images/vqe_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()

    return result, energy_history


def vqe_for_geometry(
    distance,
    ansatz,
    backend,
    out_file,
    initial_params=None,
    method="COBYLA",
    options={"maxiter": 100},
):
    """
    Run VQE for H2 molecule at given bond distance

    Args:
        distance (float): Bond distance in Angstrom
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        initial_params: Initial parameter values for the ansatz
        method (str): Optimization method (e.g., "COBYLA", "Nelder-Mead", etc.)
        options (dict): Options for the optimizer
    """
    H = hamiltonian.build_hamiltonian_with_geometry(distance)

    result, _ = vqe_single_point(
        ansatz=ansatz,
        H=H,
        backend=backend,
        out_file=out_file,
        initial_params=initial_params,
        method=method,
        options=options,
    )
    return result, result.fun


def bond_scan(
    ansatz,
    backend,
    out_file,
    distance_range=(0.5, 3.0),
    num_points=20,
    method="COBYLA",
    options={"maxiter": 100},
):
    """
    Perform the geometry optimization for H2 molecule over a range of bond distances

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        distance_range (tuple): Range of bond distances (min, max) in Angstrom
        num_points (int): Number of points in the distance range
        method (str): Optimization method (e.g., "COBYLA", "Nelder-Mead", etc.)
        options (dict): Options for the optimizer

        Returns:
            distances: List of bond distances
            energies: List of corresponding VQE energies
            optimal_distance: Bond distance with minimum energy
            optimal_energy: Minimum energy found
    """
    distances = np.linspace(distance_range[0], distance_range[1], num_points)
    energies = []

    with open(out_file, "a") as f:
        f.write("\n=== Bond Scan ===\n")
        f.write(
            f"Scanning {num_points} points from {distance_range[0]} Å to {distance_range[1]} Å\n"
        )

    # Use previous optimized parameters as initial guess for next point
    initial_params = None

    for i, dist in enumerate(distances):
        with open(out_file, "a") as f:
            f.write(f"\n--- Point {i+1}/{num_points}: Distance = {dist:.3f} Å ---\n")

        try:
            result, energy = vqe_for_geometry(
                distance=dist,
                ansatz=ansatz,
                backend=backend,
                out_file=out_file,
                initial_params=initial_params,
                method=method,
                options=options,
            )
            energies.append(energy)
            initial_params = result.x  # Update initial params for next iteration
            with open(out_file, "a") as f:
                f.write(f"VQE Energy at {dist:.3f} Å: {energy:.6f} Ha\n")
        except Exception as e:
            with open(out_file, "a") as f:
                f.write(f"Error at distance {dist:.3f} Å: {e}\n")
            energies.append(None)

    energies = np.array(energies, dtype=object)
    # Find optimal geometry among successful points
    valid_idx = np.where(energies != None)[0]

    if valid_idx.size == 0:
        # No successful VQE evaluations; log and raise a clear error
        with open(out_file, "a") as f:
            f.write(
                "\nGeometry optimization failed: no valid VQE energies were obtained for any distance.\n"
            )
        raise RuntimeError(
            "Geometry optimization failed: no valid VQE energies were obtained for any distance."
        )

    min_idx = valid_idx[np.argmin(energies[valid_idx].astype(float))]
    optimal_distance = distances[min_idx]
    optimal_energy = float(energies[min_idx])
    with open(out_file, "a") as f:
        f.write(
            f"\nOptimal bond distance: {optimal_distance:.3f} Å with energy {optimal_energy:.6f} Hartree\n"
        )

    # Plot PES
    valid_distances = distances[valid_idx]
    valid_energies = energies[valid_idx]
    plt.figure(figsize=(8, 5))
    plt.plot(valid_distances, valid_energies, marker="o")
    plt.xlabel("Bond Distance / Å")
    plt.ylabel("Energy / Hartree")
    plt.title("Potential Energy Surface of H2 Molecule")
    plt.grid()
    plt.savefig("images/h2_pes.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    results = np.column_stack((distances, energies))
    np.savetxt(
        "bond_scan.dat", results, header="Distance(Angstrom) Energy(Hartree)"
    )
    with open(out_file, "a") as f:
        f.write("Geometry optimization results saved to bond_scan.dat\n")

    return distances, energies, optimal_distance, optimal_energy
