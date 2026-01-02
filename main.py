import numpy as np
from qiskit_aer import AerSimulator

import modules.molecule as molecule
import modules.hamiltonian as hamiltonian
import modules.ansatz as ansatz_module
import modules.optimization as optimization
import modules.comparison as comparison
import modules.args as args


if __name__ == "__main__":

    cli_args = args.parse_arguments()
    out_file = "results.log"

    if cli_args.molecule == "H2":
        input_geom = "./test_molecules/h2.xyz"
        spin = 0
        charge = 0
        symmetry = True
        basis = "sto-3g"
        ncas = 2  # Number of active space orbitals for CASCI
        nelecas = (1, 1)  # Number of active space electrons (alpha, beta) for CASCI
        mapping_method = "jordan_wigner"
        ansatz_type = "efficient_su2"
        reps = 1  # Number of repetitions for the ansatz
        entanglement = "linear"  # Entanglement pattern for the ansatz
        vqe_optimizer = "COBYLA"  # Optimizer choice for VQE
        max_vqe_iterations = 200
        max_geoopt_iterations = 20
        geoopt_convergence_threshold = 1e-2
        hf_initial_state = True

    elif cli_args.molecule == "LiH":
        input_geom = "./test_molecules/lih.xyz"
        spin = 0
        charge = 0
        symmetry = False
        basis = "sto-3g"
        ncas = 4  # Number of active space orbitals for CASCI
        nelecas = (2, 2)  # Number of active space electrons (alpha, beta) for CASCI
        mapping_method = "bravyi_kitaev"
        ansatz_type = "pauli_two_design"
        reps = 2  # Number of repetitions for the ansatz
        entanglement = "full"  # Entanglement pattern for the ansatz
        vqe_optimizer = "COBYLA"  # Optimizer choice for VQE
        max_vqe_iterations = 100
        max_geoopt_iterations = 8
        geoopt_convergence_threshold = 1e-3
        hf_initial_state = True

    elif cli_args.molecule == "HF":
        input_geom = "./test_molecules/hf.xyz"
        spin = 0
        charge = 0
        symmetry = True
        basis = "sto-3g"
        # HF molecule has 10 electrons so we have different choices for an active space
        ncas = 6  # Number of Active Orbitals
        nelecas = (5, 5)  # (alpha, beta) electrons - 10 total in active space
        mapping_method = "bravyi_kitaev"
        ansatz_type = "pauli_two_design"
        reps = 2  # Number of repetitions for the ansatz
        entanglement = "full"  # Entanglement pattern for the ansatz
        vqe_optimizer = "COBYLA"  # Optimizer choice for VQE
        max_vqe_iterations = 100
        max_geoopt_iterations = 8
        geoopt_convergence_threshold = 1e-3
        hf_initial_state = True

    elif cli_args.molecule == "H2O":
        input_geom = "./test_molecules/water.xyz"
        spin = 0
        charge = 0
        symmetry = True
        basis = "sto-3g"
        ncas = 7  # Number of active space orbitals for CASCI
        nelecas = (5, 5)  # (alpha, beta) electrons - 10 total
        mapping_method = "bravyi_kitaev"
        ansatz_type = "pauli_two_design"
        reps = 2  # Number of repetitions for the ansatz
        entanglement = "full"  # Entanglement pattern for the ansatz
        vqe_optimizer = "COBYLA"  # Optimizer choice for VQE
        max_vqe_iterations = 100
        max_geoopt_iterations = 8
        geoopt_convergence_threshold = 1e-3
        hf_initial_state = True

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
    use_hf_initial_state = hf_initial_state

    ansatz = ansatz_module.create_ansatz(
        num_qubits=num_qubits,
        ansatz_type=ansatz_type,
        reps=reps,
        entanglement=entanglement,
        num_electrons=nelecas,
        use_hf_initial_state=use_hf_initial_state,
    )

    ansatz_module.write_ansatz_out(ansatz, out_file)
    ansatz_module.visualize_ansatz(ansatz, save_path="images/ansatz_circuit.png")

    # Run Geometry Optimization
    if mol.natm == 2:
        atom_labels = [mol.atom_symbol(i) for i in range(mol.natm)]
        coords = np.array([mol.atom[i][1] for i in range(mol.natm)])
        initial_distance = np.linalg.norm(coords[1] - coords[0])

        energies, distances, optimal_params, optimal_distance = (
            optimization.geometry_optimization_diatomic(
                ansatz=ansatz,
                backend=backend,
                out_file=out_file,
                atom_labels=atom_labels,
                initial_distance=initial_distance,
                method=vqe_optimizer,
                convergence_threshold=geoopt_convergence_threshold,
                step_method="backtracking",
                max_iterations=max_geoopt_iterations,
                options={"maxiter": max_vqe_iterations},
                # Two stage optimization
                use_two_stage_vqe=False,
                stage1_maxiter=150,
                stage2_maxiter=100,
                # Molecule specific parameters
                basis=basis,
                spin=spin,
                charge=charge,
                symmetry=symmetry,
                ncas=ncas,
                nelecas=nelecas,
                mapping_method=mapping_method,
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
                max_iterations=max_geoopt_iterations,
                convergence_threshold=geoopt_convergence_threshold,
                method=vqe_optimizer,
                step_size=0.05,
                basis=basis,
                ncas=ncas,
                nelecas=nelecas,
                options={"maxiter": max_vqe_iterations},
            )
        )
    else:
        raise ValueError(
            "Geometry optimization currently supports only diatomic or triatomic molecules."
        )
