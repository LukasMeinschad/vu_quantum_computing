import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


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

def optimize_vqe(ansatz, H, backend, initial_params=None, method = "COBYLA", options={"maxiter":100}):
    """  
    Run VQE optimization using simulator or real backend

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        H: Qubit Hamiltonian as SparsePauliOp
        backend: Qiskit backend (simulator or real device)
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
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend) # Make Optimized Circuit, with longer transpile time
    ansatz_isa = pm.run(ansatz)

    print(f"\n === VQE Optimization ===")
    print(f"Backend: {backend_sim.name}")
    print(f"Method: {method}, Options: {options}")
    print(f"Initial parameters shape: {initial_params.shape}")
    print(f"Number of Ansatz parameters: {ansatz.num_parameters}")

    # Track optimization progress 
    iteration_count = [0]
    energy_history = []

    def callback(xk):
        iteration_count[0] += 1
        energy = cost_func(xk, ansatz_isa, H, estimator)
        energy_history.append(energy)
        if iteration_count[0] % 10 == 0:
            # Make print every 10 iterations
            print(f"Iteration {iteration_count[0]}: Energy = {energy:.6f} Ha")
    
    # run optimization
    result = minimize(
        fun=cost_func,
        x0 = initial_params,
        args = (ansatz_isa, H, estimator),
        method = method,
        options=options,
        callback=callback
    )

    print(f"\nOptimization completed in {iteration_count[0]} iterations.")
    print(f"Final energy: {result.fun:.6f} Ha")

    # Plot convergence
    plt.figure(figsize=(8,5))
    plt.plot(range(1, iteration_count[0]+1), energy_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title("VQE Energy Convergence")
    plt.grid()
    plt.savefig("vqe_convergence.png", dpi=300, bbox_inches="tight")
    print("VQE convergence plot saved to vqe_convergence.png")
    #plt.show()
    plt.close()

    return result, energy_history

def vqe_for_geometry(distance, ansatz, backend, initial_params=None, method="COBYLA", options={"maxiter":100}):
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

    result, _ = optimize_vqe(
        ansatz = ansatz,
        H = H,
        backend = backend,
        initial_params = initial_params,
        method = method,
        options = options
    )
    return result, result.fun

def optimize_geometry(ansatz, backend, distance_range=(0.5,3.0), num_points=20,
                      method = "COBYLA", options={"maxiter":100}):
    """
    Perform the geometry optimization for H2 molecule over a range of bond distances

    Args:
        ansatz: Qiskit QuantumCircuit object representing the ansatz
        backend: Qiskit backend (simulator or real device)
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

    print(f"\n === Geometry Optimization ===")
    print(f"Scanning {num_points} points from {distance_range[0]} Å to {distance_range[1]} Å")

    # Use previous optimized parameters as initial guess for next point
    initial_params = None

    for i, dist in enumerate(distances):
        print(f"\n--- Point {i+1}/{num_points}: Distance = {dist:.3f} Å ---")
        
        
        try:
            result, energy = vqe_for_geometry(
                distance = dist,
                ansatz = ansatz,
                backend = backend,
                initial_params = initial_params,
                method = method,
                options = options
            )
            energies.append(energy)
            initial_params = result.x  # Update initial params for next iteration
            print(f"VQE Energy at {dist:.3f} Å: {energy:.6f} Ha")
        except Exception as e:
            print(f"Error at distance {dist:.3f} Å: {e}")
            energies.append(None)

    energies = np.array(energies)
    # Find optimal geometry 
    valid_idx = np.where(energies != None)[0]
    min_idx = valid_idx[np.argmin(energies[valid_idx])]
    optimal_distance = distances[min_idx]
    optimal_energy = energies[min_idx]
    print(f"\nOptimal bond distance: {optimal_distance:.3f} Å with energy {optimal_energy:.6f} Ha")

    # Plot PES
    # Filter out None values for plotting
    valid_distances = distances[valid_idx]
    valid_energies = energies[valid_idx]
    plt.figure(figsize=(8,5))
    plt.plot(valid_distances, valid_energies, marker='o')
    plt.xlabel("Bond Distance (Å)")
    plt.ylabel("Energy (Ha)")
    plt.title("Potential Energy Surface of H2 Molecule")
    plt.grid()
    plt.savefig("h2_pes.png", dpi=300, bbox_inches="tight")
    print("Potential Energy Surface plot saved to h2_pes.png")
    plt.show()
    return distances, energies, optimal_distance, optimal_energy 