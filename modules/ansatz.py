from qiskit.circuit.library import EfficientSU2, TwoLocal, pauli_two_design


def create_ansatz(
    num_qubits, ansatz_type="efficient_su2", reps=1, entanglement="linear"
):
    """
    Create a variational ansatz circuit for VQE

    Args:
        num_qubits (int): Number of qubits
        ansatz_type (str): Type of ansatz ("efficient_su2", "two_local", "pauli_two_design")
        reps (int): Number of repetitions of the ansatz layers
        entanglement (str): Entanglement pattern ("linear", "full", etc.)
    """
    if ansatz_type == "efficient_su2":
        # EfficientSU2: RY + RZ Rotations + CNOT entanglement
        ansatz = EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=True,
        )
    elif ansatz_type == "two_local":
        # Two Local: with customizable rotation and entanglement gates
        rotation_blocks = ["ry"]
        entanglement_blocks = ["crx"]
        ansatz = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement=entanglement,
            reps=reps,
            insert_barriers=True,
        )
    elif ansatz_type == "pauli_two_design":
        ansatz = pauli_two_design(
            num_qubits=num_qubits, reps=reps, insert_barriers=True
        )
    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    return ansatz


def visualize_ansatz(ansatz, save_path):
    """
    Visualize the ansatz circuit

    Args:
        ansatz: Qiskit QuantumCircuit object
        save_path (str): Saves the circuit diagram to the given path
    """
    fig = ansatz.decompose().draw(output="mpl", fold=-1, style="iqp")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# Logging functions


def write_ansatz_out(ansatz, out_file):
    """
    Write ansatz details to output file

    Args:
        ansatz: Qiskit QuantumCircuit object
        out_file (str): Path to output file
    """
    with open(out_file, "a") as f:
        f.write(f"\n === {type(ansatz).__name__} Ansatz Details ===\n")
        f.write(f"Number of qubits: {ansatz.num_qubits}\n")
        f.write(f"Number of parameters: {ansatz.num_parameters}\n")
        f.write(f"Depth: {ansatz.depth()}\n")
        f.write("\nCircuit:\n")
        f.write(ansatz.decompose().draw(output="text").single_string())
        f.write("\n")
