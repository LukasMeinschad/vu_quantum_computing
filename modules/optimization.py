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
    options={"tol": 1e-6, "maxiter": 200},
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
        initial_params = np.zeros(ansatz.num_parameters)  # HF state

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


def vqe_single_point_two_step(
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
    VQE single point calculation with a two stage optimization process

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
    mol,  # Hinzufügen des mol-Objekts
    initial_params=None,
    method="COBYLA",
    options={"maxiter": 100},
    ncas=2,
    nelecas=(1, 1),
    mapping_method="jordan_wigner",
):
    """
    Run VQE for a diatomic molecule at a given bond distance.
    """
    # Moleküleigenschaften aus dem mol-Objekt extrahieren
    atom_labels = [mol.atom_symbol(i) for i in range(mol.natm)]

    ecore, h1e, h2e = hamiltonian.build_fermionic_hamiltonian_diatomic(
        distance,
        atom_labels=atom_labels,
        basis=mol.basis,
        spin=mol.spin,
        charge=mol.charge,
        symmetry=mol.symmetry,
        ncas=ncas,
        nelecas=nelecas,
    )

    H = hamiltonian.build_qubit_hamiltonian(
        ecore, h1e, h2e, mapping_method=mapping_method
    )

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
    mol,
    distance_range,
    num_points,
    method,
    options,
    ncas,
    nelecas,
    mapping_method,
):
    """
    Perform a potential energy surface scan for a diatomic molecule.
    """
    distances = np.linspace(distance_range[0], distance_range[1], num_points)
    energies = []

    with open(out_file, "a") as f:
        f.write("\n=== Bond Scan (Potential Energy Surface) ===\n")
        f.write(
            f"Scanning {num_points} points from {distance_range[0]} Å to {distance_range[1]} Å\n"
        )

    # Start with HF parameters (zeros) for the first point
    initial_params = np.zeros(ansatz.num_parameters)

    for i, dist in enumerate(distances):
        with open(out_file, "a") as f:
            f.write(f"\n--- Point {i+1}/{num_points}: Distance = {dist:.3f} Å ---\n")

        try:
            result, energy = vqe_for_geometry(
                distance=dist,
                ansatz=ansatz,
                backend=backend,
                out_file=out_file,
                mol=mol,
                initial_params=initial_params,
                method=method,
                options=options,
                ncas=ncas,
                nelecas=nelecas,
                mapping_method=mapping_method,
            )
            energies.append(energy)
            # Use the optimized parameters from the last point as the initial guess for the next
            initial_params = result.x
            with open(out_file, "a") as f:
                f.write(f"VQE Energy at {dist:.3f} Å: {energy:.8f} Ha\n")
        except Exception as e:
            with open(out_file, "a") as f:
                f.write(f"Error at distance {dist:.3f} Å: {e}\n")
            energies.append(None)
            # Reset initial_params if a point fails
            initial_params = np.zeros(ansatz.num_parameters)

    energies = np.array(energies, dtype=object)
    valid_idx = np.where(energies != None)[0]

    if valid_idx.size == 0:
        error_msg = "Bond scan failed: no valid VQE energies were obtained."
        with open(out_file, "a") as f:
            f.write(f"\n{error_msg}\n")
        raise RuntimeError(error_msg)

    min_idx = valid_idx[np.argmin(energies[valid_idx].astype(float))]
    optimal_distance = distances[min_idx]
    optimal_energy = float(energies[min_idx])
    with open(out_file, "a") as f:
        f.write(
            f"\nOptimal bond distance from scan: {optimal_distance:.4f} Å with energy {optimal_energy:.8f} Hartree\n"
        )

    # Plot PES
    valid_distances = distances[valid_idx]
    valid_energies = energies[valid_idx].astype(float)
    plt.figure(figsize=(10, 6))
    plt.plot(valid_distances, valid_energies, marker="o", linestyle="-")
    plt.xlabel("Bond Distance / Å")
    plt.ylabel("Energy / Hartree")
    plt.title(
        f"Potential Energy Surface of {mol.atom_pure_symbol(0)}-{mol.atom_pure_symbol(1)}"
    )
    plt.grid(True)
    plt.savefig("images/pes_scan.png", dpi=300, bbox_inches="tight")
    plt.close()

    results = np.column_stack((distances, energies))
    np.savetxt(
        "bond_scan_results.dat", results, header="Distance(Angstrom) Energy(Hartree)"
    )
    with open(out_file, "a") as f:
        f.write("Bond scan results saved to bond_scan_results.dat\n")

    return distances, energies, optimal_distance, optimal_energy


"""" 
Since this bond scan function is not really an optimization we compute the 
gradients according to our hamiltonian for our H2 Molecule

Since i don't know how to compute those analytically :) i will use finite differences
"""

def grad_geometry(
    params,
    distance,
    ansatz,
    backend,
    atom_labels,
    estimator,
    delta,
    basis,
    spin,
    charge,
    symmetry,
    ncas,
    nelecas,
    mapping_method,
):
    """
    Calculate nuclear gradient dE/dR using finite differences.

    We compute: grad = (E(R+δ) - E(R-δ)) / (2δ)

    Args:
        params: Current parameter values for the ansatz circuit
        distance (float): Current bond distance in Angstrom
        ansatz: Qiskit QuantumCircuit object
        backend: Qiskit backend
        atom_labels (list[str]): List of two atom symbols
        estimator: Qiskit Estimator primitive
        delta (float): Finite difference step size in Angstrom
    """
    if estimator is None:
        if backend is None:
            backend = AerSimulator()
        elif isinstance(backend, AerSimulator):
            backend_sim = backend
        else:
            backend_sim = AerSimulator.from_backend(backend)
        estimator = BackendEstimatorV2(backend=backend_sim)

    # Compute energy at R + delta
    ecore, h1e, h2e = hamiltonian.build_fermionic_hamiltonian_diatomic(
        distance + delta,
        atom_labels=atom_labels,
        basis=basis,
        spin=spin,
        charge=charge,
        symmetry=symmetry,
        ncas=ncas,
        nelecas=nelecas,
    )

    H_plus = hamiltonian.build_qubit_hamiltonian(
        ecore, h1e, h2e, mapping_method=mapping_method
    )

    energy_plus = cost_func(params, ansatz, H_plus, estimator)

    # Compute energy at R - delta

    ecore, h1e, h2e = hamiltonian.build_fermionic_hamiltonian_diatomic(
        distance - delta,
        atom_labels=atom_labels,
        basis=basis,
        spin=spin,
        charge=charge,
        symmetry=symmetry,
        ncas=ncas,
        nelecas=nelecas,
    )

    H_minus = hamiltonian.build_qubit_hamiltonian(
        ecore, h1e, h2e, mapping_method=mapping_method
    )

    energy_minus = cost_func(params, ansatz, H_minus, estimator)

    # Central difference
    gradient = (energy_plus - energy_minus) / (2 * delta)

    print(f"    R = {distance:.6f} Å")
    print(f"    E(R+δ) = {energy_plus:.8f} Ha at R={distance+delta:.6f} Å")
    print(f"    E(R-δ) = {energy_minus:.8f} Ha at R={distance-delta:.6f} Å")
    print(f"    dE/dR = {gradient:.8f} Ha/Å")

    return gradient


def line_search_backtracking(
    params,
    distance,
    grad,
    ansatz,
    estimator,
    atom_labels,
    initial_step=0.1,
    alpha=1e-4,
    beta=0.7,
    max_iter=20,
    basis="sto-3g",
    spin=0,
    charge=0,
    symmetry=True,
    ncas=2,
    nelecas=(1, 1),
    mapping_method="jordan_wigner",
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
        estimator: Qiskit Estimator primitive object
        atom_labels (list[str]): List of two atom symbols, e.g., ["H", "H"] or ["Li", "H"]
        initial_step (float): Initial step size for line search
        alpha (float): Parameter for Armijo condition
        beta (float): Step size reduction factor
        max_iter (int): Maximum number of iterations for line search
        basis (str): Basis set for calculation
        spin (int): Spin multiplicity
        charge (int): Molecular charge
        symmetry (bool): Whether to use molecular symmetry
        ncas (int): Number of active space orbitals
        nelecas (tuple): Number of active space electrons (alpha, beta)
        mapping_method (str): Fermion-to-qubit mapping method
    """
    # BUILD WITH ALL PARAMETERS
    ecore, h1e, h2e = hamiltonian.build_fermionic_hamiltonian_diatomic(
        distance,
        atom_labels=atom_labels,
        basis=basis,
        spin=spin,
        charge=charge,
        symmetry=symmetry,
        ncas=ncas,
        nelecas=nelecas,
    )

    H_current = hamiltonian.build_qubit_hamiltonian(
        ecore, h1e, h2e, mapping_method=mapping_method
    )

    energy_current = cost_func(params, ansatz, H_current, estimator)

    step = initial_step

    for _ in range(max_iter):
        # Try new distance with current step size
        distance_new = distance - step * grad

        # Ensure distance stays positive
        if distance_new <= 0.1:
            step *= beta
            continue

        # BUILD WITH ALL PARAMETERS
        ecore, h1e, h2e = hamiltonian.build_fermionic_hamiltonian_diatomic(
            distance_new,
            atom_labels=atom_labels,
            basis=basis,
            spin=spin,
            charge=charge,
            symmetry=symmetry,
            ncas=ncas,
            nelecas=nelecas,
        )

        H_new = hamiltonian.build_qubit_hamiltonian(
            ecore, h1e, h2e, mapping_method=mapping_method
        )

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
    mol,
    initial_params=None,
    max_iterations=36,
    convergence_threshold=1e-4,
    method="COBYLA",
    step_method="backtracking",
    options={"maxiter": 100},
    # We need a two stage optimization here without this the algorithm just doesnt converge at all
    use_two_stage_vqe=True,
    stage1_maxiter=150,
    stage2_maxiter=100,
    # Molecule specific parameters
    ncas=2,
    nelecas=(1, 1),
    mapping_method="jordan_wigner",
):
    """
    Simultaneous optimization of circuit parameters and bond distances for diatomic molecules

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        mol: PySCF Mole object containing molecule information (basis, spin, charge, symmetry, atoms, coordinates)
        initial_params: Initial parameter values for the ansatz
        max_iterations (int): Maximum number of optimization iterations
        convergence_threshold (float): Convergence threshold for gradient norm
        method (str): Optimization method (e.g., "COBYLA", "Nelder-Mead", etc.)
        step_method (str): Step size method ("backtracking")
        options (dict): Options for the optimizer
        use_two_stage_vqe (bool): Whether to use two-stage VQE optimization
        stage1_maxiter (int): Max iterations for stage 1 VQE
        stage2_maxiter (int): Max iterations for stage 2 VQE
        ncas (int): Number of active space orbitals for CASCI
        nelecas (tuple): Number of active space electrons (alpha, beta) for CASCI
        mapping_method (str): Fermion-to-qubit mapping method
    """
    if initial_params is None:
        initial_params = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    # Extract molecule properties from mol object
    basis = mol.basis
    spin = mol.spin
    charge = mol.charge
    symmetry = mol.symmetry

    # Extract atom labels and initial distance from mol
    atom_labels = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = np.array([mol.atom[i][1] for i in range(mol.natm)])
    initial_distance = np.linalg.norm(coords[1] - coords[0])

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
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)

    # Storage for results
    energies = []
    distances = []
    nuclear_gradients = []
    vqe_histories = []  # Store convergence history for each VQE run

    params_history = [current_params.copy()]  # Make storage for parameters

    with open(out_file, "a") as f:
        f.write("\n=== Geometry Optimization ===\n")
        f.write(f"Initial bond distance: {initial_distance:.6f} Å\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write(f"Convergence threshold: {convergence_threshold} Hartree\n")
        f.write(f"Optimization method: {method}\n")

    for step in range(max_iterations):

        ecore, h1e, h2e = hamiltonian.build_fermionic_hamiltonian_diatomic(
            current_distance,
            atom_labels=atom_labels,
            basis=basis,
            spin=spin,
            charge=charge,
            symmetry=symmetry,
            ncas=ncas,
            nelecas=nelecas,
        )

        H_current = hamiltonian.build_qubit_hamiltonian(
            ecore, h1e, h2e, mapping_method=mapping_method
        )

        vqe_history = []
        iteration_count = [0]

        def callback_vqe(xk):
            iteration_count[0] += 1
            energy = cost_func(xk, ansatz_isa, H_current, estimator)
            vqe_history.append(energy)

        if use_two_stage_vqe:
            # Stage 1: COBYLA with higher tolerance
            result_stage1 = minimize(
                fun=cost_func,
                x0=current_params,
                args=(ansatz_isa, H_current, estimator),
                method="COBYLA",
                options={"maxiter": stage1_maxiter, "rhobeg": 0.5, "tol": 1e-4},
                callback=callback_vqe,
            )
            # Stage 2: SLSQP for fine optimization
            result = minimize(
                fun=cost_func,
                x0=result_stage1.x,
                args=(ansatz_isa, H_current, estimator),
                method="Powell",
                options={"maxiter": stage2_maxiter, "ftol": 1e-2, "xtol": 1e-2},
                callback=callback_vqe,
            )
        else:
            # Single step optimization
            result = minimize(
                fun=cost_func,
                x0=current_params,
                args=(ansatz_isa, H_current, estimator),
                method=method,
                options=options,
                callback=callback_vqe,
            )

        current_params = result.x
        current_energy = result.fun

        vqe_histories.append(
            {
                "step": step,
                "distance": current_distance,
                "history": vqe_history.copy(),
                "final_energy": current_energy,
                "iterations": iteration_count[0],
            }
        )

        # Check the VQE convergence quality
        if len(vqe_history) > 2:
            vqe_variance = np.var(vqe_history[-10:])  # Last ten iterations
            print(
                f"Step {step}: VQE energy variance in last 10 iterations: {vqe_variance:.2e}"
            )
            if vqe_variance > 1e-4:
                print(
                    f"Warning: VQE did not converge well at step {step}, variance: {vqe_variance:.2e}"
                )

        # Compute nuclear gradient
        grad_nuclear = grad_geometry(
            current_params,
            current_distance,
            ansatz_isa,
            backend,
            atom_labels=atom_labels,
            estimator=estimator,
            delta=0.0053,
            basis=basis,
            spin=spin,
            charge=charge,
            symmetry=symmetry,
            ncas=ncas,
            nelecas=nelecas,
            mapping_method=mapping_method,
        )

        # Store results
        energies.append(current_energy)
        distances.append(current_distance)
        nuclear_gradients.append(grad_nuclear)

        # Log progress
        if step % 1 == 0:
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

        # Update geometry based on gradient
        if step_method == "backtracking":
            step_size = line_search_backtracking(
                current_params,
                current_distance,
                grad_nuclear,
                ansatz_isa,
                estimator,
                atom_labels=atom_labels,
                initial_step=0.1,
                alpha=1e-4,
                beta=0.7,
                max_iter=20,
                basis=basis,
                spin=spin,
                charge=charge,
                symmetry=symmetry,
                ncas=ncas,
                nelecas=nelecas,
                mapping_method=mapping_method,
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

    # Plot VQE convergence for each geometry step
    n_steps = len(vqe_histories)
    n_cols = min(3, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_steps == 1:
        axes = np.array([[axes]])  # Make it 2D array for consistency

    for idx, vqe_data in enumerate(vqe_histories):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        history = vqe_data["history"]
        ax.plot(range(1, len(history) + 1), history, marker="o")
        ax.set_xlabel("VQE Iteration")
        ax.set_ylabel("Energy / Hartree")
        ax.set_title(
            f"Step {vqe_data['step']} (d={vqe_data['distance']:.3f} Å): "
            f"E={vqe_data['final_energy']:.6f} Ha in {vqe_data['iterations']} iters"
        )
        ax.grid()

    # Hide any unused subplots
    for idx in range(n_steps, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("images/vqe_convergence_per_step.png", dpi=300, bbox_inches="tight")
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
    H_plus = hamiltonian.build_qubit_hamiltonian(
        ecore_plus, h1e_plus, h2e_plus, mapping_method="jordan_wigner"
    )

    # Backward step
    params_minus = geometry_params.copy()
    params_minus[param_name] = geometry_params[param_name] - delta
    ecore_minus, h1e_minus, h2e_minus = hamiltonian.build_triatomic_hamiltonian(
        params_minus, atom_labels, **kwargs
    )
    H_minus = hamiltonian.build_qubit_hamiltonian(
        ecore_minus, h1e_minus, h2e_minus, mapping_method="jordan_wigner"
    )

    # Central difference
    dH_dparam = (H_plus - H_minus) / (2 * delta)
    return dH_dparam


def geometry_optimization_triatomic(
    ansatz,
    backend,
    out_file,
    mol,
    initial_params=None,
    max_iterations=50,
    convergence_threshold=1e-4,
    method="COBYLA",
    step_size=0.05,
    ncas=4,
    nelecas=(2, 2),
    mapping_method="jordan_wigner",
    options={"maxiter": 100},
):
    """
    Simultaneous optimization of circuit parameters and molecular geometry for a three-atomic molecule

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        mol: PySCF Mole object containing molecule information (basis, spin, charge, symmetry, atoms, coordinates)
        initial_params: Initial parameter values for the ansatz
        max_iterations: Maximum number of optimization iterations
        convergence_threshold: Convergence threshold for gradient norm
        method: Optimization method for circuit parameters
        step_size: Step size for geometry updates
        ncas: Number of active space orbitals
        nelecas: Number of active space electrons (alpha, beta)
        mapping_method: Fermion-to-qubit mapping method
        options: Options for the circuit parameter optimizer

    Returns:
        energies: List of energies at each iteration
        geometries: List of geometry parameter dictionaries
        optimal_params: Final optimized circuit parameters
        optimal_geometry: Final optimized geometry parameters
    """
    if initial_params is None:
        initial_params = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    # Extract molecule properties from mol object
    basis = mol.basis
    spin = mol.spin
    charge = mol.charge
    symmetry = mol.symmetry

    # Extract atom labels and initial geometry from mol
    atom_labels = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = np.array([mol.atom[i][1] for i in range(mol.natm)])

    # Calculate initial geometry parameters
    vec1 = coords[1] - coords[0]
    vec2 = coords[2] - coords[0]
    R1 = np.linalg.norm(vec1)
    R2 = np.linalg.norm(vec2)
    cos_theta = np.dot(vec1, vec2) / (R1 * R2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi

    initial_geometry = {
        "R1": R1,
        "R2": R2,
        "theta": theta,
    }

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
        H_current = hamiltonian.build_qubit_hamiltonian(
            ecore, h1e, h2e, mapping_method=mapping_method
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
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Energy convergence
    axes[0].plot(range(len(energies)), energies, marker="o", linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Energy / Hartree")
    axes[0].set_title("Energy Convergence")
    axes[0].grid(True)

    # Bond distances convergence (R1 and R2)
    R1_values = [geom["R1"] for geom in geometries]
    axes[1].plot(range(len(R1_values)), R1_values, marker="o", label="R1", linewidth=2)
    R2_values = [geom["R2"] for geom in geometries]
    axes[1].plot(range(len(R2_values)), R2_values, marker="o", label="R2", linewidth=2)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Bond Distance / Å")
    axes[1].set_title("Bond Distance Convergence")
    axes[1].legend()
    axes[1].grid(True)

    # Bond angle convergence (theta)
    if "theta" in initial_geometry:
        theta_values = [geom["theta"] for geom in geometries]
        axes[2].plot(
            range(len(theta_values)),
            theta_values,
            marker="o",
            label="theta",
            linewidth=2,
            color="green",
        )
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Angle / °")
        axes[2].set_title("Angle Convergence")
        axes[2].legend()
        axes[2].grid(True)
    else:
        # For linear molecules without theta, hide the third subplot
        axes[2].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "images/triatomic_geometry_optimization.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    return energies, geometries, current_params, current_geometry


"""  
Joint Optimization Methods the main difference here is that we build a cost function
that is both dependent on the geometry and the ansatz parameters and then evaluate the gradient of this thing

they also do this in this paper 
https://doi.org/10.48550/arXiv.2106.13840
"""


def joint_optimization_diatomic(
    ansatz,
    backend,
    out_file,
    mol,
    initial_params=None,
    initial_distance=None,
    max_iterations=100,
    convergence_threshold=1e-4,
    learning_rate=0.2,  # Parameter können sich schneller anpassen
    learning_rate_geom=0.05,  # Geometrie muss sich langsamer anpassen
    ncas=2,
    nelecas=(1, 1),
    mapping_method="jordan_wigner",
):
    """
    Joint optimization of circuit parameters and geometries that avoids the nested VQE that we have above

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
        out_file: Path to output file for logging results
        mol: PySCF Mole object containing molecule information (basis, spin, charge, symmetry, atoms, coordinates)
        initial_params: Initial parameter values for the ansatz
        initial_distance: Initial bond distance in Angstrom
        max_iterations: Maximum number of optimization iterations
        convergence_threshold: Convergence threshold for gradient norm
        learning_rate: Learning rate for circuit parameter updates
        learning_rate_geom: Learning rate for geometry updates
        ncas: Number of active space orbitals for CASCI
        nelecas: Number of active space electrons (alpha, beta) for CASCI
        mapping_method: Fermion-to-qubit mapping method
    """
    # KORREKTUR 1: Immer im Hartree-Fock-Zustand (Null-Parameter) starten
    if initial_params is None:
        initial_params = np.zeros(ansatz.num_parameters)

    # Extract the molecule properties
    basis = mol.basis
    spin = mol.spin
    charge = mol.charge
    symmetry = mol.symmetry
    atom_labels = [mol.atom_symbol(i) for i in range(mol.natm)]

    if initial_distance is None:
        coords = np.array([mol.atom[i][1] for i in range(mol.natm)])
        initial_distance = np.linalg.norm(coords[1] - coords[0])

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

    # Transpile the Ansatz
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)

    # Storage section for results
    energies = []
    distances = []
    params_history = []
    params_gradients = []
    geom_gradients = []

    with open(out_file, "a") as f:
        f.write("\n === Joint Geometry Optimization (Parameters + Geometry) ===\n")
        f.write(f"Initial bond distance: {initial_distance:.6f} Å\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write(f"Convergence threshold: {convergence_threshold} Hartree\n")
        f.write(f"Learning rate (params): {learning_rate}\n")
        f.write(f"Learning rate (geometry): {learning_rate_geom}\n")

    for iteration in range(max_iterations):
        # Build the hamiltonian for the current geometry
        ecore, h1e, h2e = hamiltonian.build_fermionic_hamiltonian_diatomic(
            current_distance,
            atom_labels=atom_labels,
            basis=basis,
            spin=spin,
            charge=charge,
            symmetry=symmetry,
            ncas=ncas,
            nelecas=nelecas,
        )

        H_current = hamiltonian.build_qubit_hamiltonian(
            ecore, h1e, h2e, mapping_method=mapping_method
        )

        # Compute the current energy
        current_energy = cost_func(current_params, ansatz_isa, H_current, estimator)

        # Compute gradient with to geometry using grad_geometry
        grad_nuclear = grad_geometry(
            current_params,
            current_distance,
            ansatz_isa,
            backend,
            atom_labels=atom_labels,
            estimator=estimator,
            delta=0.0053,
            basis=basis,
            spin=spin,
            charge=charge,
            symmetry=symmetry,
            ncas=ncas,
            nelecas=nelecas,
            mapping_method=mapping_method,
        )
        # Compute gradient with respect to parameters using parameter shift rule
        grad_params = parameter_shift_gradient(
            current_params,
            ansatz_isa,
            H_current,
            estimator,
        )
        # Store the current results
        energies.append(current_energy)
        distances.append(current_distance)
        params_history.append(current_params.copy())
        params_gradients.append(grad_params.copy())
        geom_gradients.append(grad_nuclear)

        # Compute gradient norms as a check for convergence
        grad_params_norm = np.linalg.norm(grad_params)
        grad_geom_norm = abs(grad_nuclear)
        total_grad_norm = np.sqrt(grad_params_norm**2 + grad_geom_norm**2)

        # Log the progress
        if iteration % 5 == 0:
            with open(out_file, "a") as f:
                f.write(f"Iteration {iteration}: E = {current_energy:.8f} Ha,")
                f.write(f" bond length = {current_distance:.6f} Å,")
                f.write(f" ||grad_params|| = {grad_params_norm:.6f},")
                f.write(f" |grad_geom| = {grad_geom_norm:.6f},")
                f.write(f" ||total_grad|| = {total_grad_norm:.6f}\n")
        # Check for convergence
        if total_grad_norm < convergence_threshold:
            with open(out_file, "a") as f:
                f.write(
                    f"Converged at iteration {iteration} with energy {current_energy:.8f} Ha\n"
                )
            break

        # KORREKTUR 2: Gradienten-Clipping für Stabilität
        grad_params_clipped = np.clip(grad_params, -1.0, 1.0)
        grad_nuclear_clipped = np.clip(grad_nuclear, -1.0, 1.0)

        # Update parameters and geometry using gradient descent
        current_params -= learning_rate * grad_params_clipped
        current_distance -= learning_rate_geom * grad_nuclear_clipped

        # KORREKTUR 3: Sinnvolle Grenzen für den Abstand
        current_distance = np.clip(current_distance, 0.4, 2.5)

    # Final logging
    with open(out_file, "a") as f:
        f.write(f"\n Optimization completed in {iteration+1} iterations.\n")
        f.write(f"Final bond distance: {current_distance:.6f} Å\n")
        f.write(f"Final energy: {current_energy:.8f} Ha\n")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Energy convergence
    axes[0, 0].plot(range(1, len(energies) + 1), energies, marker="o")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Energy / Hartree")
    axes[0, 0].set_title("Energy Convergence")
    axes[0, 0].grid()
    # Distance convergence
    axes[0, 1].plot(range(1, len(distances) + 1), distances, marker="o", color="orange")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Bond Distance / Å")
    axes[0, 1].set_title("Bond Distance Convergence")
    axes[0, 1].grid()
    # Parameter gradient norm convergence
    param_grad_norms = [np.linalg.norm(g) for g in params_gradients]
    axes[1, 0].plot(
        range(1, len(param_grad_norms) + 1), param_grad_norms, marker="o", color="green"
    )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("||grad_params||")
    axes[1, 0].set_title("Parameter Gradient Norm Convergence")
    axes[1, 0].grid()
    # Geometry gradient convergence
    axes[1, 1].plot(
        range(1, len(geom_gradients) + 1), geom_gradients, marker="o", color="red"
    )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("|grad_geom|")
    axes[1, 1].set_title("Geometry Gradient Convergence")
    axes[1, 1].grid()
    plt.tight_layout()
    plt.savefig(
        "images/joint_geometry_optimization_convergence.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    return energies, distances, current_params, distances[-1]


def parameter_shift_gradient(
    params,
    ansatz,
    H,
    estimator,
    shift=np.pi / 2,
):
    """
    Computes the gradient of the cost function using the Parameter-Shift Rule

    Given a cost function f = f(θ), the gradient with respect to a parameter θ_i is given by:
        df/dθ_i = (f(θ + s) - f(θ - s)) / 2
    where s is the shift amount (commonly π/2 for many gates).
    """
    gradient = np.zeros_like(params)
    for i in range(len(params)):
        # Compute all the partial derivatives
        params_plus = params.copy()
        params_plus[i] += shift
        energy_plus = cost_func(params_plus, ansatz, H, estimator)

        # Shift parameter backwards
        params_minus = params.copy()
        params_minus[i] -= shift
        energy_minus = cost_func(params_minus, ansatz, H, estimator)

        # Compute gradient
        gradient[i] = (energy_plus - energy_minus) / (2 * np.sin(shift))
    return gradient
