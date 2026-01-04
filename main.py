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
        reps = 3  # Number of repetitions for the ansatz
        entanglement = "linear"  # Entanglement pattern for the ansatz
        vqe_optimizer = "COBYLA"  # Optimizer choice for VQE
        max_vqe_iterations = 200
        max_geoopt_iterations = 50
        geoopt_convergence_threshold = 1e-4
        hf_initial_state = True

    backend = AerSimulator()

    with open(out_file, "w") as f:
        f.write("VQE Geometry Optimization Results\n")
        f.write("=" * 40 + "\n\n")

    # Build Qubit Hamiltonian
    H_qubit = hamiltonian.build_qubit_hamiltonian_diatomic_from_xyz_qiskit_nature(
        "./test_molecules/h2.xyz", distance=0.74, basis=basis, charge=charge, spin=spin
    )
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

    # Bond scan using Qiskit Nature-based Hamiltonian from 0.6 to 0.9 Å
    scan_method = vqe_optimizer
    scan_options = {"maxiter": max_vqe_iterations}

    optimization.bond_scan_qiskit_nature(
        ansatz=ansatz,
        backend=backend,
        out_file=out_file,
        xyz_path="./test_molecules/h2.xyz",
        distance_range=(0.6, 0.9),
        num_points=10,
        method=scan_method,
        options=scan_options,
        basis=basis,
        charge=charge,
        spin=spin,
    )
