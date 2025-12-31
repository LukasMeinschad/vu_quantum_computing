from qiskit_aer import AerSimulator

import modules.molecule as molecule
import modules.hamiltonian as hamiltonian
import modules.mapping as mapping
import modules.ansatz as ansatz_module
import modules.optimization as optimization
import modules.comparison as comparison


if __name__ == "__main__":

    # Set parameters
    out_file = "results.log"
    input_geometry = "./test_molecules/h2.xyz"
    spin = 0
    charge = 0
    symmetry = True
    basis_set = "sto-3g"
    ncas = 2  # Number of active space orbitals for CASCI
    nelecas = (1, 1)  # Number of active space electrons (alpha, beta) for CASCI
    mapping_method = "bravyi_kitaev"
    ansatz_type = "pauli_two_design"
    reps = 5 # Number of repetitions for the ansatz
    entanglement = "full" # Entanglement pattern for the ansatz

    backend = AerSimulator()

    with open(out_file, "w") as f:
        f.write("VQE Geometry Optimization Results\n")
        f.write("=" * 40 + "\n\n")

    # Build Molecule and Run SCF
    mol = molecule.build_molecule_from_xyz(
        input_geometry, basis=basis_set, spin=spin, charge=charge, symmetry=symmetry
    )
    mf = hamiltonian.run_scf_calculation(mol, method="RHF")
    molecule.write_molecule_out(mol, out_file)
    molecule.write_energy_out(mf, out_file)

    """  
    These test functions can be uncommented to run different comparison tests
    regarding the VQE optimization process.
    """

    # Compare VQE with full and active space Hamiltonians at fixed geometry
    #comparison.run_single_point_comparison(mf, backend, out_file, molecule="H2")

    # Compare the influence of initial parameters on VQE optimization
    # comparison.initial_parameters_influence(mf, backend, out_file)

    # Compare the influence of optimizer choice on VQE optimization
    # comparison.influence_optimizer_choice(mf, backend, out_file)

    # Compare the influence of ansatz depth on VQE optimization
    # comparison.influence_ansatz_depth(mf, backend, out_file)


    # Get Hartree-Fock Fermionic Hamiltonian
    ecore, h1e, h2e = hamiltonian.get_hf_hamiltonian(mf)
    hamiltonian.write_hamiltonian_out(ecore, h1e, h2e, out_file, label="Hartree-Fock")

    # Get Complete Active Space Fermionic Hamiltonian
    ecore, h1e, h2e = hamiltonian.get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)
    hamiltonian.write_hamiltonian_out(ecore, h1e, h2e, out_file, label="CASCI")

    # Comparison of Jordan-Wigner and Bravyi-Kitaev Mappings
    mapping.compare_mappings(ecore, h1e, h2e, out_file)

    # Build Qubit Hamiltonian
    H_qubit = hamiltonian.build_hamiltonian(ecore, h1e, h2e, mapping_method)
    hamiltonian.write_qubit_hamiltonian_out(H_qubit, out_file)

    # Create ansatz circuit
    num_qubits = H_qubit.num_qubits
    ansatz = ansatz_module.create_ansatz(num_qubits, ansatz_type=ansatz_type, reps=reps, entanglement=entanglement)
    ansatz_module.write_ansatz_out(ansatz, out_file)
    ansatz_module.visualize_ansatz(ansatz, save_path="images/ansatz_circuit.png")

    # Run VQE Optimization
    vqe_result, energy_history = optimization.vqe_single_point(
        ansatz,
        H_qubit,
        backend,
        out_file=out_file,
        method="COBYLA",
        options={"maxiter": 100},
    )

    # Try optimized VQE with 2 Stages
    vqe_results, energy_history = optimization.vqe_single_point_optimized(
        ansatz,
        H_qubit,
        backend,
        out_file=out_file
    )

    # Run Geometry Optimization
    ansatz_geo = ansatz_module.create_ansatz(
        num_qubits, ansatz_type=ansatz_type, reps=2
    )

    optimization.bond_scan(
        ansatz_geo,
        backend,
        out_file=out_file,
        distance_range=(0.2, 1.5),
        num_points=10,
        method="COBYLA",
        options={"maxiter": 100},
    )

    # New option with gradient evaluation
    print("\nRunning geometry optimization with gradient evaluation...\n")
    energies, distances, optimal_params, optimal_distance = optimization.geometry_optimization_sim(
        ansatz=ansatz_geo,
        backend = backend,
        out_file=out_file,
        initial_distance=0.8,
        method = "COBYLA",
        convergence_threshold=1e-4,
        step_method="backtracking",
        max_iterations=36,
        options={"maxiter": 100}
    ) 
