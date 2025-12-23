import numpy as np
from qiskit_aer import AerSimulator 

import modules.molecule as molecule 
import modules.hamiltonian as hamiltonian
import modules.mapping as mapping
import modules.ansatz as ansatz_module
import modules.optimization as optimization


if __name__ == "__main__":
    out_file = "vqe_geomopt_results.txt"

    # Clear Output File
    with open(out_file, "w") as f:
        f.write("VQE Geometry Optimization Results\n")
        f.write("="*40 + "\n\n")

    h2_filepath = "./test_molecules/h2.xyz"
    mol = molecule.build_molecule_from_xyz(h2_filepath, basis="sto-6g", spin=0, charge=0, symmetry=True)
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
    comparison = mapping.compare_mappings(ecore,h1e,h2e)

    print("\n === Qubit Hamiltonian ===")
    H_qubit = hamiltonian.build_hamiltonian(ecore, h1e, h2e, mapping="bravyi_kitaev")
    print("Qubit Hamiltonian:\n", H_qubit)
    print("Number of qubits:", H_qubit.num_qubits)
    print("Number of terms in Hamiltonian:", len(H_qubit.paulis))

    # Create ansatz circuit
    num_qubits = H_qubit.num_qubits
    ansatz = ansatz_module.create_ansatz(num_qubits, ansatz_type="pauli_two_design")
    ansatz_module.visualize_ansatz(ansatz, save_path="ansatz_circuit.png")

    # Random initial state
    x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)

    backend = AerSimulator()
    vqe_result, energy_history = optimization.optimize_vqe(
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
    ansatz_geo = ansatz_module.create_ansatz(num_qubits, ansatz_type="efficient_su2", reps=2)

    backend = AerSimulator()
    distances, energies, optimal_distance, optimal_energy = optimization.optimize_geometry(
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
