import numpy as np
from qiskit_aer import AerSimulator

import modules.molecule as molecule
import modules.hamiltonian as hamiltonian
import modules.ansatz as ansatz_module
import modules.optimization as optimization
import modules.comparison as comparison


if __name__ == "__main__":

    # Set Parameters
    out_file = "results.log"
    input_geom = "./test_molecules/h2.xyz"
    spin = 0
    charge = 0
    symmetry = True
    basis = "sto-3g"
    ncas = 2  # Number of active space orbitals for CASCI
    nelecas = (1, 1)  # Number of active space electrons (alpha, beta) for CASCI
    mapping_method = "bravyi_kitaev"
    ansatz_type = "pauli_two_design"
    reps = 5  # Number of repetitions for the ansatz
    entanglement = "full"  # Entanglement pattern for the ansatz
    vqe_optimizer = "Powell"  # Optimizer choice for VQE

    backend = AerSimulator()

    with open(out_file, "w") as f:
        f.write("VQE Geometry Optimization Results\n")
        f.write("=" * 40 + "\n\n")

    # Build Molecule and Run SCF
    mol = molecule.build_molecule_from_xyz(input_geom, basis, spin, charge, symmetry)
    mf = hamiltonian.run_scf_calculation(mol, method="RHF")
    molecule.write_molecule_out(mol, out_file)
    molecule.write_energy_out(mf, out_file)

    # Get Hartree-Fock Fermionic Hamiltonian
    ecore, h1e, h2e = hamiltonian.get_hf_hamiltonian(mf)
    hamiltonian.write_hamiltonian_out(ecore, h1e, h2e, out_file, label="Hartree-Fock")

    # Get Complete Active Space Fermionic Hamiltonian
    ecore, h1e, h2e = hamiltonian.get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)
    hamiltonian.write_hamiltonian_out(ecore, h1e, h2e, out_file, label="CASCI")

    """  
    These test functions can be uncommented to run different comparison tests
    regarding the VQE optimization process.
    """

    # Compare VQE with full and active space Hamiltonians at fixed geometry
    # comparison.run_single_point_comparison(mf, backend, out_file, molecule="H2")

    # Compare the influence of initial parameters on VQE optimization
    # comparison.initial_parameters_influence(mf, backend, out_file)

    # Compare the influence of optimizer choice on VQE optimization
    # comparison.influence_optimizer_choice(mf, backend, out_file)

    # Compare the influence of ansatz depth on VQE optimization
    # comparison.influence_ansatz_depth(mf, backend, out_file)

    # Comparison of Jordan-Wigner and Bravyi-Kitaev Mappings
    # comparison.compare_mappings(ecore, h1e, h2e, out_file)

    # Build Qubit Hamiltonian
    H_qubit = hamiltonian.build_hamiltonian(ecore, h1e, h2e, mapping_method)
    hamiltonian.write_qubit_hamiltonian_out(H_qubit, out_file)

    # Create Ansatz Circuit
    num_qubits = H_qubit.num_qubits
    ansatz = ansatz_module.create_ansatz(num_qubits, ansatz_type, reps, entanglement)
    ansatz_module.write_ansatz_out(ansatz, out_file)
    ansatz_module.visualize_ansatz(ansatz, save_path="images/ansatz_circuit.png")

    # Run Geometry Optimization
    if mol.natm == 2:
        coords = np.array([mol.atom[i][1] for i in range(mol.natm)])
        initial_distance = np.linalg.norm(coords[1] - coords[0])

        energies, distances, optimal_params, optimal_distance = (
            optimization.geometry_optimization_diatomic(
                ansatz=ansatz,
                backend=backend,
                out_file=out_file,
                initial_distance=initial_distance,
                method=vqe_optimizer,
                convergence_threshold=1e-3,
                step_method="backtracking",
                max_iterations=36,
                options={"maxiter": 100},
            )
        )
    elif mol.natm == 3:
        atom_labels = [mol.atom_symbol(i) for i in range(mol.natm)]
        coords = np.array([mol.atom[i][1] for i in range(mol.natm)])

        vec1 = coords[1] - coords[0]
        vec2 = coords[2] - coords[0]

        R1 = np.linalg.norm(vec1)
        R2 = np.linalg.norm(vec2)

        # Calculate bond angle at central atom (angle 1-0-2)
        cos_theta = np.dot(vec1, vec2) / (R1 * R2)
        theta = (
            np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi
        )  # Convert to degrees

        initial_geometry = {
            "R1": R1,
            "R2": R2,
            "theta": theta,
        }

        energies_h2o, geometries_h2o, optimal_params_h2o, optimal_geometry_h2o = (
            optimization.geometry_optimization_triatomic(
                ansatz=ansatz,
                backend=backend,
                out_file=out_file,
                atom_labels=atom_labels,
                initial_geometry=initial_geometry,
                max_iterations=30,
                convergence_threshold=1e-3,
                method=vqe_optimizer,
                step_size=0.05,
                basis=basis,
                ncas=ncas,
                nelecas=nelecas,
                options={"maxiter": 100},
            )
        )
    else:
        raise ValueError(
            "Geometry optimization currently supports only diatomic or triatomic molecules."
        )
