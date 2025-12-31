import matplotlib.pyplot as plt
import numpy as np

import modules.hamiltonian as hamiltonian
import modules.ansatz as ansatz_module
import modules.optimization as optimization


def run_single_point_comparison(mf, backend, out_file, molecule="H2"):
    """   
    Function that compares VQE with full and active space Hamiltonians at fixed geoemtries

    Args:
        mf: Converged SCF object; mf = mean_field
        backend: Qiskit backend to run VQE on
        out_file: File to write results to
    """ 
    distance= 0.74 # Equilibrium bond length for H2 
    results = {}

    # 1. Full hamiltonian
    # TODO i think im completely stupid here 
    # I dont know how this transformation in case of a full hamiltonian work
    # Maybe we can figure this out together later
    # What i do in the meantime is just to use the CASICI hamiltonian with all orbitals included
    print("\nRunning VQE with full Hamiltonian...\n")
    ncas_full = mf.mo_coeff.shape[1]  # Use all orbitals for full Hamiltonian
    nelecas_full = (mf.mol.nelectron // 2, mf.mol.nelectron // 2)  # All electrons
    ecore_hf, h1e_hf, h2e_hf = hamiltonian.get_casci_hamiltonian(mf, ncas=ncas_full, nelecas=nelecas_full)
    H_qubit_hf = hamiltonian.build_hamiltonian(ecore_hf, h1e_hf, h2e_hf, mapping_method="jordan_wigner")
    num_qubits_hf = H_qubit_hf.num_qubits
    ansatz_hf = ansatz_module.create_ansatz(num_qubits_hf, ansatz_type="pauli_two_design", reps=3, entanglement="full")
    vqe_result_hf, energy_history_hf = optimization.vqe_single_point(
        ansatz_hf,
        H_qubit_hf,
        backend,
        out_file=out_file,
        method="COBYLA",
        options={"maxiter": 100},
    )
    results['Full Hamiltonian'] = vqe_result_hf
    
    # 2. Active space hamiltonian
    print("\nRunning VQE with active space Hamiltonian...\n")
    ncas = 2  # Number of active space orbitals for CASCI
    nelecas = (1, 1)  # Number of active space electrons (alpha, beta) for CASCI
    ecore_cas, h1e_cas, h2e_cas = hamiltonian.get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)
    H_qubit_cas = hamiltonian.build_hamiltonian(ecore_cas, h1e_cas, h2e_cas, mapping_method="jordan_wigner")
    num_qubits_cas = H_qubit_cas.num_qubits
    ansatz_cas = ansatz_module.create_ansatz(num_qubits_cas, ansatz_type="pauli_two_design", reps=3, entanglement="full")
    vqe_result_cas, energy_history_cas = optimization.vqe_single_point(
        ansatz_cas,
        H_qubit_cas,
        backend,
        out_file=out_file,
        method="COBYLA",
        options={"maxiter": 100},
    )
    results['Active Space Hamiltonian'] = vqe_result_cas
    
    # Compare energy convergence
    plt.figure(figsize=(8,6))
    plt.plot(energy_history_hf, label='Full Hamiltonian', marker='o')
    plt.plot(energy_history_cas, label='Active Space Hamiltonian', marker='x')

    # If molecule is H2 plot the Sto-3G  FullCI and HF energy
    full_ci_energy = -1.137284
    hf_energy =  -1.116759
    if molecule == "H2":
        plt.axhline(y=full_ci_energy, color='r', linestyle='--', label=f'Full-CI Energy (Sto-3G) {full_ci_energy:.6f} Ha')
        plt.axhline(y=hf_energy, color='g', linestyle='--', label=f'Hartree-Fock Energy (Sto-3G) {hf_energy:.6f} Ha')   


    plt.title('VQE Energy Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.legend()
    plt.grid()
    plt.savefig('images/vqe_energy_convergence_comparison.png')
    plt.close()

    print(f"Number of Qubits (Full Hamiltonian): {num_qubits_hf}")
    print(f"Number of Qubits (Active Space Hamiltonian): {num_qubits_cas}")

    
def initial_parameters_influence(mf, backend, out_file, molecule="H2"):
    """   
    Function that studies the influence of initial parameters on the VQE optimization
    
    Args:
        mf: Converged SCF object; mf = mean_field
        backend: Qiskit backend to run VQE on
        out_file: File to write results to
    """
    ncas = 2  # Number of active space orbitals for CASCI
    nelecas = (1, 1)  # Number of active space electrons (alpha, beta) for CASCI
    ecore, h1e, h2e = hamiltonian.get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)
    H_qubit = hamiltonian.build_hamiltonian(ecore, h1e, h2e, mapping_method="jordan_wigner")
    num_qubits = H_qubit.num_qubits
    ansatz = ansatz_module.create_ansatz(num_qubits, ansatz_type="pauli_two_design", reps=3, entanglement="full")
    
    # Randomize values in interval 2 * pi * uniform(0,1)
    n_initial_params = 5
    initial_parameters = [list(2 * np.pi * np.random.rand(ansatz.num_parameters)) for _ in range(n_initial_params)]
    initial_params_list = initial_parameters
    initial_params_list = np.array(initial_params_list) 
    results = {}
    for idx, initial_params in enumerate(initial_params_list):
        print(f"\nRunning VQE with initial parameters set {idx+1}...\n")
        vqe_result, energy_history = optimization.vqe_single_point(
            ansatz,
            H_qubit,
            backend,
            out_file=out_file,
            method="COBYLA",
            options={"maxiter": 100},
            initial_params=initial_params
        )
        results[f'Initial Params Set {idx+1}'] = (vqe_result, energy_history)
    # Plot energy convergence for different initial parameters
    plt.figure(figsize=(8,6))
    for label, (vqe_result, energy_history) in results.items():
        plt.plot(energy_history, label=label, marker='o')

    if molecule == "H2":
        full_ci_energy = -1.137284
        plt.axhline(y=full_ci_energy, color='r', linestyle='--', label=f'Full-CI Energy (Sto-3G) {full_ci_energy:.6f} Ha')

    
    plt.title('VQE Energy Convergence with Different Initial Parameters')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.legend()
    plt.grid()
    plt.savefig('images/vqe_initial_parameters_influence.png')
    plt.close()

def influence_ansatz_depth(mf, backend, out_file, molecule="H2"):
    """  
    Function that studies the influence of ansatz depth on the VQE optimization
    """
    ncas = 2  # Number of active space orbitals for CASCI
    nelecas = (1, 1)  # Number of active space electrons (alpha, beta) for CASCI
    ecore, h1e, h2e = hamiltonian.get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)
    H_qubit = hamiltonian.build_hamiltonian(ecore, h1e, h2e, mapping_method="jordan_wigner")
    num_qubits = H_qubit.num_qubits
    
    depths = [1, 2, 3, 4, 5]
    results = {}
    for depth in depths:
        print(f"\nRunning VQE with ansatz depth: {depth}...\n")
        ansatz = ansatz_module.create_ansatz(num_qubits, ansatz_type="pauli_two_design", reps=depth, entanglement="full")
        vqe_result, energy_history = optimization.vqe_single_point(
            ansatz,
            H_qubit,
            backend,
            out_file=out_file,
            method="COBYLA",
            options={"maxiter": 100},
        )
        results[f'Depth {depth}'] = (vqe_result, energy_history)
    
    # Plot energy convergence for different ansatz depths
    plt.figure(figsize=(8,6))
    for label, (vqe_result, energy_history) in results.items():
        plt.plot(energy_history, label=label, marker='o')

    if molecule == "H2":
        full_ci_energy = -1.137284
        plt.axhline(y=full_ci_energy, color='r', linestyle='--', label=f'Full-CI Energy (Sto-3G) {full_ci_energy:.6f} Ha')

    
    plt.title('VQE Energy Convergence with Different Ansatz Depths')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.legend()
    plt.grid()
    plt.savefig('images/vqe_ansatz_depth_influence.png')
    plt.close()

def influence_optimizer_choice(mf, backend, out_file, molecule="H2"):
    """    
    Function that studies the influence of optimizer choice on the VQE optimization

    We use the following optimizers:
    - COBYLA
    - Nelder-Mead
    - BFGS
    - SLSQP
    - Powell
    """
    ncas = 2  # Number of active space orbitals for CASCI
    nelecas = (1, 1)  # Number of active space electrons (alpha, beta) for CASCI
    ecore, h1e, h2e = hamiltonian.get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)
    H_qubit = hamiltonian.build_hamiltonian(ecore, h1e, h2e, mapping_method="jordan_wigner")
    num_qubits = H_qubit.num_qubits
    ansatz = ansatz_module.create_ansatz(num_qubits, ansatz_type="pauli_two_design", reps=3, entanglement="full")
    
    optimizers = ["COBYLA", "Nelder-Mead", "BFGS", "SLSQP", "Powell"]
    results = {}
    for optimizer in optimizers:
        print(f"\nRunning VQE with optimizer: {optimizer}...\n")
        vqe_result, energy_history = optimization.vqe_single_point(
            ansatz,
            H_qubit,
            backend,
            out_file=out_file,
            method=optimizer,
            options={"maxiter": 100},
        )
        results[optimizer] = (vqe_result, energy_history)
    
    # Plot energy convergence for different optimizers
    plt.figure(figsize=(8,6))
    for label, (vqe_result, energy_history) in results.items():
        plt.plot(energy_history, label=label, marker='o')

    if molecule == "H2":
        full_ci_energy = -1.137284
        plt.axhline(y=full_ci_energy, color='r', linestyle='--', label=f'Full-CI Energy (Sto-3G) {full_ci_energy:.6f} Ha')

    
    plt.title('VQE Energy Convergence with Different Optimizers')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.legend()
    plt.grid()
    plt.savefig('images/vqe_optimizer_influence.png')
    plt.close()
