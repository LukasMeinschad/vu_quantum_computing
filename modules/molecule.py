import numpy as np 
from pyscf import gto, scf
import os

def build_molecule_from_xyz(xyz_file, basis="sto-6g", spin=0, charge=0, symmetry=True):
    """  
    Helper function to build a PySCF molecule object from a xyz file

    Args:
        xyz_file (str): Path to the xyz file.
        basis (str): Basis set to use.
        spin (int): Spin multiplicity.
        charge (int): Charge of the molecule.
        symmetry (bool or str): Whether to use symmetry. If given a string of point
                                group name, the given point group symmetry will be used.
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    # First line is number of atoms
    n_atoms = int(lines[0].strip())

    # Second line is comment line 
    # Then we have atoms of type Element x y z
    atom_list = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        element = parts[0]
        x,y,z = float(parts[1]), float(parts[2]), float(parts[3])
        atom_list.append([element, (x,y,z)])

    # Build the molecule object 
    mol = gto.Mole()
    mol.build(
        verbose=0, # Print symmetry details
        atom=atom_list,
        basis=basis,
        spin=spin,
        charge=charge,
        symmetry=symmetry
    )

    return mol

def run_scf_calculation(mol, method="RHF", verbose=False):
    """  
    Run SCF calculation on the given molecule object.

    Args:
        mol: PySCF molecule object.
        method (str): SCF method to use ("RHF", "UHF", or "ROHF").
        verbose (bool): Whether to print SCF details.
    
    Returns:
        mf: Converged SCF object.
    """
    print(f"=== Running {method} SCF Calculation ===")
    print(f"Molecule: {mol.atom}") 
    print(f"Basis Set: {mol.basis}")
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Number of basis functions: {mol.nao_nr()}")

    # Select SCF method
    if method == "RHF":
        mf = scf.RHF(mol)
    elif method == "UHF":
        mf = scf.UHF(mol)
    elif method == "ROHF":
        mf = scf.ROHF(mol)
    else:
        raise ValueError(f"Unsupported SCF method: {method}")
    
    # Set verbosity
    mf.verbose = 4 if verbose else 0
    print("Starting SCF iterations...")
    energy = mf.kernel()

    if not mf.converged:
        print("SCF did not converge!")
    
    print(f"Scf converged, final electronic energy: {energy} Ha")

    return mf 

# Logging Functions

def write_molecule_out(mol, filepath):
    """
    Writes basic properties of the molecule to out
    """
    with open(filepath, 'w') as f:
        f.write("=== Molecule Properties ===\n")
        f.write(f"Atoms:\n")
        for atom in mol.atom:
            f.write(f"  {atom}\n")
        f.write(f"Basis Set: {mol.basis}\n")
        f.write(f"Number of electrons: {mol.nelectron}\n")
        f.write(f"Number of basis functions: {mol.nao_nr()}\n")



def write_energy_out(mf, filepath):
    """  
    Writes the final electronic energy to a file

    Args:
        mf: Converged SCF object.
        filepath (str): Path to the output file.
    """
    # Check if outfile exists and append to it
    if os.path.exists(filepath):
        mode = 'a'
    else:
        mode = 'w'
    with open(filepath, mode) as f:
        f.write("=== SCF Energy Results ===\n")
        f.write(f"Nucleic Repulsion Energy: {mf.energy_nuc()}\n")
        f.write(f"Electronic Energy: {mf.energy_elec()[0]}\n")
        f.write(f"Total energy: {mf.energy_tot()}\n")
        f.write(f"Difference (Total - Nuc): {mf.energy_tot() - mf.energy_nuc()}\n")

