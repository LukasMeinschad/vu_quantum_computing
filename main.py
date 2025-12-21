import numpy as np
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers import ParityMapper


def parse_xyz_to_pyscf_driver(filepath, basis_set="sto3g"):
    """  
    Parses the molecule xyz file to the PySCFDriver object of qiskit-nature

    atom="H 0 0 0; H 0 0 0.735" etc
    """
    with open(filepath, "r") as file:
        lines = file.readlines()
        num_atoms = int(lines[0].strip())
        atom_lines = lines[2:2 + num_atoms]
        atom_string = "; ".join(line.strip() for line in atom_lines)
    driver = PySCFDriver(atom=atom_string, basis=basis_set, unit=DistanceUnit.ANGSTROM,charge=0, spin=0)
    return driver


if __name__ == "__main__":
    file_path = "./test_molecules/h2.xyz"
    molecule = parse_xyz_to_pyscf_driver(file_path, basis_set="sto3g")
     
    # Obtain the electronic structure Hamiltonian of the molecule
    es_problem = molecule.run()
    fermionic_operator = es_problem.hamiltonian.second_q_op()

    # Map the fermionic operator to a qubit operator using Jordan-Wigner mapping 
    mapper = JordanWignerMapper()
    qubit_jw_op = mapper.map(fermionic_operator)
 
    # Use Parity Mapping
    mapper = ParityMapper()
    qubit_p_op = mapper.map(fermionic_operator)