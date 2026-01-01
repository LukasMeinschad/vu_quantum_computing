from qiskit.circuit.library import EfficientSU2, TwoLocal, pauli_two_design
from qiskit import QuantumCircuit

"""  
Still this ansatz thing is one main concern because VQE just doesnt converge


Another thing we can look at if we still have time is this stuff from the Pennylane guys:
google.com/search?q=VQE+hatree+fock+initial+state&oq=VQE+hatree+fock+initial+state&gs_lcrp=EgZjaHJvbWUyCwgAEEUYChg5GKABMgkIARAhGAoYoAEyCQgCECEYChigAdIBCDU2MDdqMGo0qAIAsAIB&sourceid=chrome&ie=UTF-8
"""

def create_hf_initial_state(num_qubits, num_electrons):
    """  
    Creates a HF-Initial state circuit 

    This works the following way:

    - For a given number of qubits and electrons, we prepare the Hartree-Fock state by applying X gates to the first 'num_electrons' qubits.

    """
    n_alpha, n_beta =  num_electrons
    hf_circuit = QuantumCircuit(num_qubits)

    # For Jordan-Wigner
    # First half qubits = alpha spin, second half = beta spin
    # Occupy the lowest energy orbitals
    for i in range(n_alpha):
        hf_circuit.x(i)  # Apply X gate to occupy alpha orbitals
    for i in range(n_beta):
        hf_circuit.x(num_qubits // 2 + i)  # Apply X gate to occupy beta orbitals

    return hf_circuit

def create_uccsd_ansatz(num_qubits, num_electrons, reps=1):
    """   
    Creates UCCSD (Unitary Coupled Cluster with Singles and Doubles) ansatz
    This one is actually chemically inspired and should perform better than generic ansatzes
    """
    try: 
        from qiskit_nature.second_q.circuit.library import UCCSD
        from qiskit_nature.second_q.mappers import JordanWignerMapper 

        num_spatial_orbitals = num_qubits // 2
        mapper = JordanWignerMapper()

        ansatz = UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_electrons,
            qubit_mapper=mapper,
            reps=reps,
        )
        return ansatz
    except ImportError:
        raise ImportError("qiskit-nature is required for UCCSD ansatz. Please install it via 'pip install qiskit-nature'.")
    
def create_pauli_two_design_ansatz(num_qubits, reps=2, entanglement="full", initial_state=None):
    """
    Create a Pauli Two Design ansatz circuit for VQE

    Args:
        num_qubits (int): Number of qubits
        reps (int): Number of repetitions of the ansatz layers
        entanglement (str): Entanglement pattern ("linear", "full", etc.)
    """
    var_circuit = pauli_two_design(
        num_qubits=num_qubits, reps=reps, entanglement=entanglement, insert_barriers=True
    )
    if initial_state is not None:
        ansatz = initial_state.compose(var_circuit)
    else:
        ansatz = var_circuit


    return ansatz

def creeate_hardware_efficient_ansatz(num_qubits, reps=2, entanglement="full", initial_state=None):
    """
    Create a Hardware Efficient ansatz circuit for VQE

    Args:
        num_qubits (int): Number of qubits
        reps (int): Number of repetitions of the ansatz layers
        entanglement (str): Entanglement pattern ("linear", "full", etc.)
    """
    var_circuit = EfficientSU2(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
        insert_barriers=True,
    )
    if initial_state is not None:
        ansatz = initial_state.compose(var_circuit)
    else:
        ansatz = var_circuit

    return ansatz

def create_ansatz(num_qubits,
                   ansatz_type = "pauli_two_design",
                     reps=2, 
                     entanglement="full",
                     num_electrons = None,
                     use_hf_initial_state = True):
    """
    Factory function to create ansatz circuits based on specified type
    """ 
    # Create HF initial state if required
    initial_state = None
    if use_hf_initial_state and num_electrons is not None:
        initial_state = create_hf_initial_state(num_qubits, num_electrons)

    if ansatz_type == "uccsd":
        if num_electrons is None:
            raise ValueError("num_electrons must be provided for UCCSD ansatz")
        ansatz = create_uccsd_ansatz(num_qubits, num_electrons, reps=reps)
    elif ansatz_type == "pauli_two_design":
        ansatz = create_pauli_two_design_ansatz(num_qubits, reps=reps, entanglement=entanglement, initial_state=initial_state)
    elif ansatz_type == "efficient_su2":
        ansatz = creeate_hardware_efficient_ansatz(num_qubits, reps=reps, entanglement=entanglement, initial_state=initial_state)
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
