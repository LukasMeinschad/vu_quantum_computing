from qiskit.circuit.library import EfficientSU2, TwoLocal, pauli_two_design 


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