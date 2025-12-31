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


def vqe_single_point_optimized(
    ansatz,
    H,
    backend,
    out_file,
    initial_params=None,
    stage1_method="COBYLA",
    stage1_options={"maxiter": 100, "rhobeg": 0.1},
    stage2_method="Powell",
    stage2_options={"maxiter": 50, "ftol": 1e-5, "xtol": 1e-5},
):
    """
    Optimized VQE single point function with a two stage optimization process

    + Stage 1: COBYLA for rough optimization
    + Stage 2: Powell for fine optimization

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        H: Qubit Hamiltonian as SparsePauliOp
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        initial_params: Initial parameter values for the ansatz
        stage1_method (str): Optimization method for stage 1 (e.g., "COBYLA")
        stage1_options (dict): Options for the optimizer in stage 1
        stage2_method (str): Optimization method for stage 2 (e.g., "Powell")
        stage2_options (dict): Options for the optimizer in stage 2
    """
    if initial_params is None:
        initial_params = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    # Setup backend and estimator
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

    # Track optimization progress
    stage1_history = []
    stage2_history = []
    with open(out_file, "a") as f:
        f.write("\n=== VQE Single Point Optimization (Two-Stage) ===\n")
        f.write(f"Backend: {backend_sim.name}\n")
        f.write(f"Stage 1 Method: {stage1_method}, Options: {stage1_options}\n")
        f.write(f"Stage 2 Method: {stage2_method}, Options: {stage2_options}\n")

    # ======== Stage 1: Rough Optimization ========
    def callback_stage1(xk):
        energy = cost_func(xk, ansatz_isa, H, BackendEstimatorV2(backend=backend_sim))
        stage1_history.append(energy)
        if len(stage1_history) % 10 == 0:
            with open(out_file, "a") as f:
                f.write(
                    f"Stage 1 - Iteration {len(stage1_history)}: Energy = {energy:.6f} Hartree\n"
                )

    result_stage1 = minimize(
        fun=cost_func,
        x0=initial_params,
        args=(ansatz_isa, H, estimator),
        method=stage1_method,
        options=stage1_options,
        callback=callback_stage1,
    )

    print(
        f"Stage 1 completed in {len(stage1_history)} iterations. Energy: {result_stage1.fun:.6f} Ha"
    )

    # ======== Stage 2: Fine Optimization ========
    def callback_stage2(xk):
        energy = cost_func(xk, ansatz_isa, H, BackendEstimatorV2(backend=backend_sim))
        stage2_history.append(energy)
        if len(stage2_history) % 10 == 0:
            with open(out_file, "a") as f:
                f.write(
                    f"Stage 2 - Iteration {len(stage2_history)}: Energy = {energy:.6f} Hartree\n"
                )

    result_stage2 = minimize(
        fun=cost_func,
        x0=result_stage1.x,
        args=(ansatz_isa, H, estimator),
        method=stage2_method,
        options=stage2_options,
        callback=callback_stage2,
    )

    with open(out_file, "a") as f:
        f.write(f"\nOptimization completed.\n")
        f.write(f"Final energy: {result_stage2.fun:.6f} Hartree\n")

    print(
        f"Stage 2 completed in {len(stage2_history)} iterations. Energy: {result_stage2.fun:.6f} Ha"
    )
    # Combine histories

    total_history = stage1_history + stage2_history
    energy_improvement = total_history[0] - total_history[-1]

    plt.figure(figsize=(8, 5))
    # Plot stage 1 history
    plt.plot(
        range(1, len(stage1_history) + 1),
        stage1_history,
        marker="o",
        label="Stage 1 (COBYLA)",
    )
    # Plot stage 2 history with offset
    offset = len(stage1_history)
    plt.plot(
        range(offset + 1, offset + len(stage2_history) + 1),
        stage2_history,
        marker="o",
        label="Stage 2 (Powell)",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Energy / Hartree")
    plt.title(
        f"VQE Energy Convergence (Total Improvement: {energy_improvement:.6f} Ha)"
    )
    plt.legend()
    plt.grid()
    plt.savefig("images/vqe_convergence_two_stage.png", dpi=300, bbox_inches="tight")
    plt.close()

    return result_stage2, total_history


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
    np.savetxt("bond_scan.dat", results, header="Distance(Angstrom) Energy(Hartree)")
    with open(out_file, "a") as f:
        f.write("Geometry optimization results saved to bond_scan.dat\n")

    return distances, energies, optimal_distance, optimal_energy


"""" 
Since this bond scan function is not really an optimization we compute the 
gradients according to our hamiltonian for our H2 Molecule

Since i don't know how to compute those analytically :) i will use finite differences
"""


def finite_diff_hamiltonian(distance, delta=1e-5):
    """
    Helper function to use central difference to the gradient with respect to the geometry as a parameter
    """
    H_plus = hamiltonian.build_hamiltonian_with_geometry(distance + delta)
    H_minus = hamiltonian.build_hamiltonian_with_geometry(distance - delta)
    dH_ddistance = (H_plus - H_minus) / (2 * delta)
    return dH_ddistance


def grad_geometry(params, distance, ansatz, backend, estimator=None):
    """
    Calculate nuclear gradient using finite differences

    Args:
        params: Parameter values for the ansatz circuit
        distance: Bond distance in Angstrom
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        estimator: Qiskit Estimator primitive object
    """
    if estimator is None:
        if backend is None:
            backend = AerSimulator()
        elif isinstance(backend, AerSimulator):
            backend_sim = backend
        else:
            # Real backend, create simulator
            backend_sim = AerSimulator.from_backend(backend)

        estimator = BackendEstimatorV2(backend=backend_sim)

    grad_H = finite_diff_hamiltonian(distance)
    # Evaluate the expectation value of the gradient Hamiltonian
    pub = (ansatz, [grad_H], [params])
    result = estimator.run(pubs=[pub]).result()
    gradient = result[0].data.evs[0]

    return gradient


def line_search_backtracking(
    params,
    distance,
    grad,
    ansatz,
    backend,
    estimator,
    initial_step=0.1,
    alpha=1e-4,
    beta=0.7,
    max_iter=20,
):
    """
    Implementation of a simple backtracking line search to find the optimal step size and speed up convergence

    Algorithm:
        1. Start with an initial step size
        2. While the Armijo condition is not satisfied and max iterations not reached:
            a. Reduce the step size by multiplying with beta
            b. Check the Armijo condition:
                f(x + step * d) <= f(x) + alpha * step * grad^T * d
    Args:
        params: Current parameter values for the ansatz circuit
        distance: Current bond distance in Angstrom
        grad: Gradient value at current parameters
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        estimator: Qiskit Estimator primitive object
        initial_step (float): Initial step size for line search
        alpha (float): Parameter for Armijo condition
        beta (float): Step size reduction factor
        max_iter (int): Maximum number of iterations for line search
    """
    H_current = hamiltonian.build_hamiltonian_with_geometry(distance)
    energy_current = cost_func(params, ansatz, H_current, estimator)

    step = initial_step

    for _ in range(max_iter):
        # Try new distance with current step size
        distance_new = distance - step * grad

        # Ensure distance stays positive
        if distance_new <= 0.1:
            step *= beta
            continue
        H_new = hamiltonian.build_hamiltonian_with_geometry(distance_new)
        energy_new = cost_func(params, ansatz, H_new, estimator)

        # Armijo conditon: sufficient decrease
        if energy_new <= energy_current - alpha * step * grad**2:
            return step  # Found suitable step size
        else:
            # Reduce step size
            step *= beta
    return step  # Return the last step size if max iterations reached


def geometry_optimization_diatomic(
    ansatz,
    backend,
    out_file,
    initial_distance=0.74,
    initial_params=None,
    max_iterations=36,
    convergence_threshold=1e-4,
    method="COBYLA",
    step_method="backtracking",
    options={"maxiter": 100},
):
    """
    Simultaneous optimization of circuit parameters and bond distances for H2 molecule

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        initial_distance (float): Initial bond distance in Angstrom
        initial_params: Initial parameter values for the ansatz
        max_iterations (int): Maximum number of optimization iterations
        convergence_threshold (float): Convergence threshold for energy change
        method (str): Optimization method (e.g., "COBYLA", "Nelder-Mead", etc.)
        options (dict): Options for the optimizer
    """
    if initial_params is None:
        initial_params = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    current_params = initial_params.copy()
    current_distance = initial_distance

    # Setup backend and estimator
    if backend is None:
        backend = AerSimulator()
    elif isinstance(backend, AerSimulator):
        backend_sim = backend
    else:
        backend_sim = AerSimulator.from_backend(backend)

    estimator = BackendEstimatorV2(backend=backend_sim)

    # Transpile the ansatz
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend
    )  # Make Optimized Circuit, with longer transpile time
    ansatz_isa = pm.run(ansatz)

    # Storage for results
    energies = []
    distances = []
    nuclear_gradients = []

    with open(out_file, "a") as f:
        f.write("\n=== Geometry Optimization ===\n")
        f.write(f"Initial bond distance: {initial_distance:.6f} Å\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write(f"Convergence threshold: {convergence_threshold} Hartree\n")
        f.write(f"Optimization method: {method}\n")

    for step in range(max_iterations):
        # Build hamiltonian for current geometry
        H_current = hamiltonian.build_hamiltonian_with_geometry(current_distance)

        # Minimize circuit parameters for current geometry
        result = minimize(
            fun=cost_func,
            x0=current_params,
            args=(ansatz_isa, H_current, estimator),
            method=method,
            options=options,
        )
        current_params = result.x
        current_energy = result.fun

        # Compute nuclear gradient
        grad_nuclear = grad_geometry(
            current_params, current_distance, ansatz_isa, backend, estimator
        )

        # Store results
        energies.append(current_energy)
        distances.append(current_distance)
        nuclear_gradients.append(grad_nuclear)

        # Log progress
        if step % 4 == 0:
            with open(out_file, "a") as f:
                f.write(f"Step {step}: E = {current_energy:.8f} Ha,")
                f.write(f" bond length = {current_distance:.6f} Å,")
                f.write(f" nuclear gradient = {grad_nuclear:.6f} Ha/Å\n")

        # Check the convergence
        if abs(grad_nuclear) < convergence_threshold:
            with open(out_file, "a") as f:
                f.write(
                    f"Converged at step {step} with energy {current_energy:.8f} Ha\n"
                )
            break
        # Update geometry based in gradient

        # THOMAS HOFER WOULD BE PROUD

        if step_method == "backtracking":
            step_size = line_search_backtracking(
                current_params,
                current_distance,
                grad_nuclear,
                ansatz_isa,
                backend,
                estimator,
                initial_step=0.1,
                alpha=1e-4,
                beta=0.7,
                max_iter=20,
            )
            current_distance -= step_size * grad_nuclear
        else:
            raise ValueError(f"Unknown step method: {step_method}")

    # Final logging
    with open(out_file, "a") as f:
        f.write(f"\nOptimization completed in {step+1} steps.\n")
        f.write(f"Final bond distance: {current_distance:.6f} Å\n")
        f.write(f"Final energy: {current_energy:.8f} Ha\n")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(range(1, len(energies) + 1), energies, marker="o")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Energy / Hartree")
    ax1.set_title("Energy Convergence")
    ax1.grid()
    ax2.plot(range(1, len(distances) + 1), distances, marker="o", color="orange")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Bond Distance / Å")
    ax2.set_title("Bond Distance Convergence")
    ax2.grid()
    plt.savefig(
        "images/geometry_optimization_convergence.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    return energies, distances, current_params, distances[-1]


def finite_diff_triatomic_gradient(
    geometry_params, param_name, atom_labels, delta=1e-5, **kwargs
):
    """
    Compute gradient of Hamiltonian with respect to a geometry parameter using finite differences

    Args:
        geometry_params: Current geometry parameters dictionary
        param_name: Name of parameter to differentiate ("R1", "R2", "theta")
        atom_labels: List of three atom symbols
        delta: Finite difference step size
        **kwargs: Additional arguments for build_triatomic_hamiltonian

    Returns:
        dH_dparam: Gradient of Hamiltonian with respect to parameter
    """
    # Forward step
    params_plus = geometry_params.copy()
    params_plus[param_name] = geometry_params[param_name] + delta
    ecore_plus, h1e_plus, h2e_plus = hamiltonian.build_triatomic_hamiltonian(
        params_plus, atom_labels, **kwargs
    )
    H_plus = hamiltonian.build_hamiltonian(
        ecore_plus, h1e_plus, h2e_plus, mapping_method="jordan_wigner"
    )

    # Backward step
    params_minus = geometry_params.copy()
    params_minus[param_name] = geometry_params[param_name] - delta
    ecore_minus, h1e_minus, h2e_minus = hamiltonian.build_triatomic_hamiltonian(
        params_minus, atom_labels, **kwargs
    )
    H_minus = hamiltonian.build_hamiltonian(
        ecore_minus, h1e_minus, h2e_minus, mapping_method="jordan_wigner"
    )

    # Central difference
    dH_dparam = (H_plus - H_minus) / (2 * delta)
    return dH_dparam


def geometry_optimization_triatomic(
    ansatz,
    backend,
    out_file,
    atom_labels,
    initial_geometry,
    initial_params=None,
    max_iterations=50,
    convergence_threshold=1e-4,
    method="COBYLA",
    step_size=0.05,
    basis="sto-3g",
    ncas=4,
    nelecas=(2, 2),
    options={"maxiter": 100},
):
    """
    Simultaneous optimization of circuit parameters and molecular geometry for a three-atomic molecule

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        atom_labels: List of three atom symbols, e.g., ["O", "H", "H"]
        initial_geometry: Dictionary with initial geometry parameters
            For H2O: {"R1": 0.96, "R2": 0.96, "theta": 104.5}
            For linear: {"R1": 1.0, "R2": 1.0}
        initial_params: Initial parameter values for the ansatz
        max_iterations: Maximum number of optimization iterations
        convergence_threshold: Convergence threshold for gradient norm
        method: Optimization method for circuit parameters
        step_size: Step size for geometry updates
        basis: Basis set for quantum chemistry calculations
        ncas: Number of active space orbitals
        nelecas: Number of active space electrons (alpha, beta)
        options: Options for the circuit parameter optimizer

    Returns:
        energies: List of energies at each iteration
        geometries: List of geometry parameter dictionaries
        optimal_params: Final optimized circuit parameters
        optimal_geometry: Final optimized geometry parameters
    """
    if initial_params is None:
        initial_params = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    current_params = initial_params.copy()
    current_geometry = initial_geometry.copy()

    # Setup backend and estimator
    if backend is None:
        backend = AerSimulator()
    elif isinstance(backend, AerSimulator):
        backend_sim = backend
    else:
        backend_sim = AerSimulator.from_backend(backend)

    estimator = BackendEstimatorV2(backend=backend_sim)

    # Transpile ansatz
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)

    # Storage for results
    energies = []
    geometries = []
    gradients_history = []

    with open(out_file, "a") as f:
        f.write("\n=== Geometry Optimization ===\n")
        f.write(f"Atoms: {atom_labels}\n")
        f.write(f"Initial geometry: {initial_geometry}\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write(f"Convergence threshold: {convergence_threshold}\n")
        f.write(f"Active space: {ncas} orbitals, {nelecas} electrons\n\n")

    for iteration in range(max_iterations):
        # Build Hamiltonian for current geometry
        ecore, h1e, h2e = hamiltonian.build_triatomic_hamiltonian(
            current_geometry, atom_labels, basis=basis, ncas=ncas, nelecas=nelecas
        )
        H_current = hamiltonian.build_hamiltonian(
            ecore, h1e, h2e, mapping_method="jordan_wigner"
        )

        # Optimize circuit parameters for current geometry
        result = minimize(
            fun=cost_func,
            x0=current_params,
            args=(ansatz_isa, H_current, estimator),
            method=method,
            options=options,
        )
        current_params = result.x
        current_energy = result.fun

        # Compute gradients with respect to all geometry parameters
        gradients = {}
        for param_name in current_geometry.keys():
            dH = finite_diff_triatomic_gradient(
                current_geometry,
                param_name,
                atom_labels,
                basis=basis,
                ncas=ncas,
                nelecas=nelecas,
            )
            # Evaluate gradient expectation value
            pub = (ansatz_isa, [dH], [current_params])
            grad_result = estimator.run(pubs=[pub]).result()
            gradients[param_name] = grad_result[0].data.evs[0]

        # Store results
        energies.append(current_energy)
        geometries.append(current_geometry.copy())
        gradients_history.append(gradients.copy())

        # Compute gradient norm for convergence check
        grad_norm = np.sqrt(sum(g**2 for g in gradients.values()))

        # Log progress
        with open(out_file, "a") as f:
            f.write(f"Iteration {iteration}:\n")
            f.write(f"  Energy: {current_energy:.8f} Ha\n")
            f.write(f"  Geometry: {current_geometry}\n")
            f.write(f"  Gradients: {gradients}\n")
            f.write(f"  Gradient norm: {grad_norm:.6e}\n\n")

        # Check convergence
        if grad_norm < convergence_threshold:
            with open(out_file, "a") as f:
                f.write(f"Converged at iteration {iteration}!\n")
                f.write(f"Final energy: {current_energy:.8f} Ha\n")
                f.write(f"Final geometry: {current_geometry}\n")
            break

        # Update geometry using gradient descent
        for param_name in current_geometry.keys():
            # Use adaptive step size based on gradient magnitude
            adaptive_step = step_size / (1 + abs(gradients[param_name]))
            current_geometry[param_name] -= adaptive_step * gradients[param_name]

            # Apply constraints to keep geometry physically reasonable
            if param_name in ["R1", "R2"]:
                current_geometry[param_name] = max(
                    0.4, min(3.0, current_geometry[param_name])
                )
            elif param_name == "theta":
                current_geometry[param_name] = max(
                    60.0, min(180.0, current_geometry[param_name])
                )

    # Final output
    with open(out_file, "a") as f:
        f.write(f"\nOptimization completed in {iteration + 1} iterations.\n")
        f.write(f"Final geometry: {current_geometry}\n")
        f.write(f"Final energy: {current_energy:.8f} Ha\n")

    # Plot convergence
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Energy convergence
    axes[0].plot(range(len(energies)), energies, marker="o", linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Energy / Hartree")
    axes[0].set_title("Energy Convergence")
    axes[0].grid(True)

    # Geometry parameters convergence
    for param_name in initial_geometry.keys():
        param_values = [geom[param_name] for geom in geometries]
        label = f"{param_name} ({'Å' if param_name.startswith('R') else '°'})"
        axes[1].plot(
            range(len(param_values)), param_values, marker="o", label=label, linewidth=2
        )

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Geometry Parameters")
    axes[1].set_title("Geometry Convergence")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(
        "images/triatomic_geometry_optimization.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    return energies, geometries, current_params, current_geometry
