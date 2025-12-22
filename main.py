import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2, TwoLocal, pauli_two_design 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit_aer import AerSimulator 
from qiskit.primitives import BackendEstimatorV2 
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Modules Import
import modules.molecule as molecule 
import modules.hamiltonian as hamiltonian




def cholesky(V, eps=1e-5):
    """
    Cholesky decomposition of two-electron integrals
    
    Args:
        V: Two-electron integral tensor (no, no, no, no)
        eps: Threshold for decomposition accuracy
    
    Returns:
        L: Cholesky vectors (no, no, ng)
        ng: Number of Cholesky vectors
    
    References:
        - https://arxiv.org/pdf/1711.02242.pdf section B2
        - https://arxiv.org/abs/1808.02625
        - https://arxiv.org/abs/2104.08957
    """
    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "Accuracy of Cholesky decomposition:",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng


def identity(n):
    """  
    Builds the Identity operator for n qubits

    Returns:
        SparsePauliOp: Identity operator on n qubits I^n
    """
    return SparsePauliOp.from_list([("I" * n, 1.0 )])


def bravyi_kitaev_update_set(n, j):
    """  
    Helper function to calculate the update Set U(j) for Bravyi-Kitaev mapping

    The update set U(j) contains indices of qubits that store occupation information
    that must be updated when orbital j changes

    Args:
        n (int): Number of spin-orbitals
        j (int): Orbital index
    
    Returns:
        list: Indices in the update set
    """
    update_set = [j]
    
    # Find parent nodes in the binary tree
    # Add the bit length that j covers
    k = j
    while k < n - 1:
        # Find the next update position by flipping the rightmost 0 bit
        # This follows the binary tree structure
        k = k + ((k + 1) & (~k))
        if k < n:
            update_set.append(k)
        else:
            break
    
    return update_set

def bravyi_kitaev_parity_set(n, j):
    """
    Calculate the parity set P(j) for Bravyi-Kitaev mapping

    The parity set P(j) contains indices of qubits that must be checked
    to determine the parity of orbitals with indices less than j

    Args:
        n (int): Number of spin-orbitals
        j (int): Orbital index

    Returns:
        list: List of qubit indices in the parity set P(j)
    """
    if j == 0:
        return []
    
    parity_set = []
    
    # Find all positions that store parity information needed for position j
    # We need to traverse back through the binary tree
    k = j - 1
    
    # Keep removing the rightmost set bit to find parent nodes
    while k >= 0:
        parity_set.append(k)
        # Remove rightmost set bit: k & (k-1)
        k_new = k & (k - 1)
        if k_new == k or k == 0:
            break
        k = k_new - 1
        if k < 0:
            break
    
    return parity_set



def creators_destructors(n, mapping="jordan_wigner"):
    """  
    Generate creation and annihilation operators using different mappings
    
    Jordan-Wigner Transformation for fermionic operators:
    - Creation operator a_j^dagger: (X_p - iY_p)/2 x Z_1 x Z_2 ... x Z_(j-1)
    - Annihilation operator a_j: (X_p + iY_p)/2 x Z_1 x Z_2 ... x Z_(j-1)

    Note that the Z chain encodes the fermionic anticommutation relations

    Bravyi-Kitaev Transformation:
    - Stores occupation and parity information in logarithmic number of qubits
    - Creation operator: a_j^dagger = 1/2 (X_U(j) x X_j x Z_P(j) - iX_U(j) x Y_j x Z_P(j))
    - Annihilation operator: a_j = 1/2 (X_U(j) x X_j x Z_P(j) + iX_U(j) x Y_j x Z_P(j))
    
    Args:
        n (int): Number of spin-orbitals
        mapping (str): Mapping method ("jordan_wigner", "parity", "bravyi_kitaev")
    
    Returns:
        creators (list): List of creation operators as SparsePauliOp
        destructors (list): List of annihilation operators as SparsePauliOp
    """
    c_list = []
    if mapping == "jordan_wigner":
        for p in range(n):
            # Construct Pauli String for position p
            # Left side: Identity operators for qubits > p
            # Right side: Z operators for qubits < p
            if p == 0:
                ell, r = "I" *(n-1), ""
            elif p == n-1:
                ell, r = "", "Z" * (n-1)
            else:
                ell, r = "I" * (n-p-1), "Z" * p

            # Creation operator a_p^dagger
            cp = SparsePauliOp.from_list([
                (ell + "X" + r, 0.5),
                (ell + "Y" + r, -0.5j)
            ])
            c_list.append(cp)
    elif mapping == "bravyi_kitaev":
        for j in range(n):
            U_j = bravyi_kitaev_update_set(n,j)
            P_j = bravyi_kitaev_parity_set(n,j)

            # Build pauli string for this orbital
            # Initialize with identity
            pauli_x = ["I"] * n
            pauli_y = ["I"] * n

            # Apply X to update set (all qubits in U(j))
            for idx in U_j:
                pauli_x[n-1 - idx] = "X"
                pauli_y[n-1 - idx] = "X"
            
            # Apply central operator at position j 
            pauli_x[n-1 - j] = "X"
            pauli_y[n-1 - j] = "Y"

            # Apply Z to parity set P(j)
            for idx in P_j:
                pauli_x[n-1 - idx] = "Z"
                pauli_y[n-1 - idx] = "Z"

            pauli_str_x = "".join(pauli_x)
            pauli_str_y = "".join(pauli_y)

            # Creation operator a_j^dagger
            cp = SparsePauliOp.from_list([
                (pauli_str_x, 0.5),
                (pauli_str_y, -0.5j)
            ])
            c_list.append(cp)
    
    else:
        raise ValueError(f"Mapping {mapping} not implemented.")

    # Annhilation operatos are Hermitian adjoints
    d_list = [c.adjoint() for c in c_list]
    return c_list, d_list

def build_hamiltonian(ecore: float, h1e: np.ndarray, h2e: np.ndarray, mapping="jordan_wigner") -> SparsePauliOp:
    """ 
    Buils the qubit Hamiltonian from fermionic integrals

    Fermionic Hamiltonian is:
    H = E_core + sum_{pq} h1e_{pq} a_p^† a_q + 0.5 sum_{pqrs} h2e_{pqrs} a_p^† a_q^† a_r a_s
    """

    ncas, _ = h1e.shape

    # Get creation and annihilation operators
    # 2 * ncas to account for spin-orbitals
    C,D = creators_destructors(ncas * 2, mapping=mapping)

    # Build excitation operators c_p^† c_r for all p,r
    # Exc[p][r-p] represents excitation from orbital r to orbital p
    # Do this for both spins
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]] # Number of operators if p == r
        for r in range(p+1, ncas):
            Excp.append(
                C[p] @ D[r] # \alpha spin p -> r
                + C[ncas + p] @ D[ncas + r] # \beta spin p -> r
                + C[r] @ D[p] # \alpha spin r -> p
                + C[ncas + r] @ D[ncas + p] # \beta spin r -> p
            )
        Exc.append(Excp)

    # Low-rank decomposition of h2e
    Lop, ng = cholesky(h2e, eps=1e-5)

    # t1e = h1e - 1/2 * Coloumb integral
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)

    # Initialize Hamiltonian with core energy
    H = ecore * identity(ncas * 2)

    # Add one-body-terms
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p,r] * Exc[p][r - p]

    # Add two body terms
    for g in range(ng):
        Lg = 0 * identity(ncas * 2)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p,r,g] * Exc[p][r - p]
        # Square operator
        H += 0.5 * (Lg @ Lg)

    # Combine like terms and remove small coefficients
    return H.chop().simplify()


def create_ansatz(num_qubits, ansatz_type="efficient_su2", reps=1, entanglement="linear"):
    """ 
    Create a variational ansatz circuit for VQE 
    
    Args:
        num_qubits (int): Number of qubits
        ansatz_type (str): Type of ansatz ("efficient_su2", "two_local")
        reps (int): Number of repetitions of the ansatz layers
        entanglement (str): Entanglement pattern ("linear", "full", etc.)
    """
    if ansatz_type == "efficient_su2":
        # EfficientSU2: RY + RZ Rotations + CNOT entanglement
        ansatz = EfficientSU2(
            num_qubits=num_qubits,
            reps =  reps,
            entanglement=entanglement,
            insert_barriers=True
        )
    elif ansatz_type == "two_local":
        # Two Local: with customizable rotation and entanglement gates
        rotation_blocks = ['ry']
        entanglement_blocks = ['crx']
        ansatz = TwoLocal(
            num_qubits = num_qubits,
            rotation_blocks = rotation_blocks,
            entanglement_blocks = entanglement_blocks,
            entanglement = entanglement,
            reps = reps,
            insert_barriers = True
        )
    elif ansatz_type == "pauli_two_design":
        ansatz = pauli_two_design(
            num_qubits = num_qubits,
            reps = reps,
            insert_barriers = True
        )

    
    print(f"\n === Ansatz: {ansatz_type} ===")
    print(f"Number of qubits: {num_qubits}")
    print(f"Number of parameters: {ansatz.num_parameters}")
    print(f"Depth: {ansatz.depth()}")

    return ansatz

def visualize_ansatz(ansatz,save_path=None):
    """  
    Visualize the ansatz circuit

    Args:
        ansatz: Qiskit QuantumCircuit object
        save_path (str): If provided, saves the circuit diagram to the given path
    """
    fig = ansatz.decompose().draw(output="mpl", fold=-1, style="iqp")
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Ansatz circuit diagram saved to {save_path}")
    
    # Otherwise save in current directory
    fig.savefig("ansatz_circuit.png", dpi=300, bbox_inches="tight")


    return fig

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


def ham_terms(x: float):
    """    
    Build Hamiltonian terms for H2 molecules based on bond distance x (in Angstrom)

    Args:
        x (float): Bond distance in Angstrom

    Returns:
        h1e: One-electron integrals
        h2e: Two-electron integrals
        ecore: Core energy
    """
    distance = x 
    a = distance / 2 
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom = [["H", (0,0,-a)], ["H", (0,0,a)]],
        basis = "sto-6g",
        spin = 0,
        charge = 0,
        symmetry = True
    )

    mf = scf.RHF(mol)
    mf.kernel()

    if not mf.converged:
        raise ValueError("SCF calculation did not converge.")
    
    mx = mcscf.CASCI(mf, ncas=2, nelecas=(1,1))
    casci_energy = mx.kernel()
    if casci_energy is None:
        raise ValueError("CASCI calculation did not converge.")
    
    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)
    return ecore, h1e, h2e

def build_hamiltonian_with_geometry(distx: float) -> SparsePauliOp:
    """  
    Build qubit Hamiltonian for H2 molecule at given bond distance

    Args:
        distx (float): Bond distance in Angstrom
    Returns:
        H_qubit: Qubit Hamiltonian as SparsePauliOp
    """
    ecore, h1e, h2e = ham_terms(distx)

    ncas, _ = h1e.shape
    C,D = creators_destructors(ncas * 2, mapping="jordan_wigner")
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]]
        for r in range(p+1, ncas):
            Excp.append(
                C[p] @ D[r]
                + C[ncas + p] @ D[ncas + r]
                + C[r] @ D[p]
                + C[ncas + r] @ D[ncas + p]
            )
        Exc.append(Excp)
    # Low-rank decomposition of h2e
    Lop, ng = cholesky(h2e, eps=1e-5)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
    H = ecore * identity(ncas * 2)
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p,r] * Exc[p][r - p]
    # Add two body terms
    for g in range(ng):
        Lg = 0 * identity(ncas * 2)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p,r,g] * Exc[p][r - p]
        H += 0.5 * (Lg @ Lg)
    return H.chop().simplify()

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
    H = build_hamiltonian_with_geometry(distance)

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


# ==== Helper Functions --> Make Module some other time ====

def build_hamiltonian_helper(ecore,h1e,h2e,C,D,ncas):
    """   
    Helper function to test the differences of the Jordan-Wigner and Bravyi-Kitaev mappings
    """
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]]
        for r in range(p+1, ncas):
            Excp.append(
                C[p] @ D[r]
                + C[ncas + p] @ D[ncas + r]
                + C[r] @ D[p]
                + C[ncas + r] @ D[ncas + p]
            )
        Exc.append(Excp)
    # Low-rank decomposition of h2e
    Lop, ng = cholesky(h2e, eps=1e-5)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
    H = ecore * identity(ncas * 2)
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p,r] * Exc[p][r - p]
    # Add two body terms
    for g in range(ng):
        Lg = 0 * identity(ncas * 2)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p,r,g] * Exc[p][r - p]
        H += 0.5 * (Lg @ Lg)
    return H.chop().simplify()

def compare_mappings(ecore,h1e,h2e):
    """  
    Compares the Jordan-Wigner and Bravyi-Kitaev mapping for the same fermionic Hamiltonian
    """
    print("\n === Comparing Jordan-Wigner and Bravyi-Kitaev Mappings ===")

    print("\n-- Jordan-Wigner Mapping --")
    import time
    start_jw = time.time()
    ncas, _ = h1e.shape
    C_jw, D_jw = creators_destructors(ncas * 2, mapping="jordan_wigner")
    H_jw = build_hamiltonian_helper(ecore,h1e,h2e,C_jw,D_jw,ncas)
    end_jw = time.time()
    print(f"Jordan-Wigner Hamiltonian has {len(H_jw.paulis)} terms")
    print(f"Time taken for Jordan-Wigner: {end_jw - start_jw:.4f} seconds")

    print("\n-- Bravyi-Kitaev Mapping --")
    start_bk = time.time()
    C_bk, D_bk = creators_destructors(ncas * 2, mapping="bravyi_kitaev")
    H_bk = build_hamiltonian_helper(ecore,h1e,h2e,C_bk,D_bk,ncas)
    end_bk = time.time()
    print(f"Bravyi-Kitaev Hamiltonian has {len(H_bk.paulis)} terms")
    print(f"Time taken for Bravyi-Kitaev: {end_bk - start_bk:.4f} seconds")

    # Compare Speedup and Term Reduction
    print(f"\nSpeedup (BK vs JW): {(end_jw - start_jw)/(end_bk - start_bk):.2f}x")
    print(f"Term Reduction (BK vs JW): {len(H_jw.paulis)/len(H_bk.paulis):.2f}x")

    return {
    "H_jw": H_jw,
    "H_bk": H_bk,
    "time_jw": end_jw - start_jw,
    "time_bk": end_bk - start_bk
    }


if __name__ == "__main__":
    out_file = "vqe_geomopt_results.txt"

    # Clear Output File
    with open(out_file, "w") as f:
        f.write("VQE Geometry Optimization Results\n")
        f.write("="*40 + "\n\n")



    h2_filepath = "/Users/lukas/Desktop/vu_quantum_computing/test_molecules/h2.xyz"
    mol = molecule.build_molecule_from_xyz(h2_filepath, basis="sto-3g", spin=0, charge=0, symmetry=True)
    mf = molecule.run_scf_calculation(mol, method="RHF")
    molecule.write_molecule_out(mol, out_file)
    molecule.write_energy_out(mf, out_file)

    ecore, h1e, h2e = hamiltonian.get_full_space_hamiltonian(mf)

    hamiltonian.write_hamiltonian_out(ecore, h1e, h2e, out_file)



    print(" === Full Fermionic Hamiltonian ===")
    h1e, h2e, ecore = hamiltonian.get_fermionic_hamiltonian(mf)
    print("Core energy: ", ecore)
    print(f"h1e shape: {h1e.shape}, h2e shape: {h2e.shape}")
    #print("One-electron integrals (h1e):\n", h1e)
    #print("Two-electron integrals (h2e) in MO basis:\n", h2e)


    print("\n === Active Space Fermionic Hamiltonian (CASCI) ===")
    ncas=2
    nelecas=(1,1)
    h1e, h2e, ecore = hamiltonian.get_fermionic_hamiltonian_active_space(mf, ncas=ncas, nelecas=nelecas)
    print("Core energy (active space): ", ecore)
    print(f"h1e shape: {h1e.shape}, h2e shape: {h2e.shape}")
    #print("One-electron integrals (h1e):\n", h1e)
    #print("Two-electron integrals (h2e) in MO basis:\n", h2e)


    # Comparison of Mappings
    print("\n" + "="*30 + "\n")
    comparison = compare_mappings(ecore,h1e,h2e)



    print("\n === Qubit Hamiltonian ===")
    H_qubit = build_hamiltonian(ecore, h1e, h2e, mapping="bravyi_kitaev")
    print("Qubit Hamiltonian:\n", H_qubit)
    print("Number of qubits:", H_qubit.num_qubits)
    print("Number of terms in Hamiltonian:", len(H_qubit.paulis))

    # Create ansatz circuit
    num_qubits = H_qubit.num_qubits
    ansatz = create_ansatz(num_qubits, ansatz_type="pauli_two_design")
    visualize_ansatz(ansatz, save_path="ansatz_circuit.png")

    # Random initial state
    x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    backend = AerSimulator()
    vqe_result, energy_history = optimize_vqe(
        ansatz,
        H_qubit,
        backend,
        initial_params=x0,
        method="COBYLA",
        options={"maxiter":100}
    )

    # Example Geometry Optimization for H2 molecule
    print("\n" + "="*30 + "\n")
    print("Starting Geometry Optimization for H2 molecule")
    print("\n" + "="*30 + "\n")
    num_qubits = 4  # 2 orbitals x 2 spins
    ansatz_geo = create_ansatz(num_qubits, ansatz_type="efficient_su2", reps=2)

    backend = AerSimulator()
    distances, energies, optimal_distance, optimal_energy = optimize_geometry(
        ansatz_geo,
        backend,
        distance_range=(0.2, 1.5),
        num_points=10,
        method="COBYLA",
        options={"maxiter":100}
    )
    print(f"Optimal bond distance: {optimal_distance:.3f} Å with energy {optimal_energy:.6f} Ha")

    # Save to text file
    results = np.column_stack((distances, energies))
    np.savetxt("h2_geometry_optimization.txt", results, header="Distance(Angstrom) Energy(Ha)")
    print("Geometry optimization results saved to h2_geometry_optimization.txt")

