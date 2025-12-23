from pyscf import ao2mo, mcscf
import numpy as np

def get_full_space_hamiltonian(mf):
    """  
    Gets the full-space fermionic Hamiltonian from a converged SCF object.

    Args:
        mf: Converged SCF object.

    Returns:
        ecore: Core energy (nuclear repulsion).
        h1e: One-electron integrals.
        h2e: Two-electron integrals.
        E_reference
    """
    # Nuclear repulsion energy
    ecore = mf.energy_nuc()

    # Get molecular orbitals and coefficients
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]

    # Transform one-electron integrals 
    h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff

    # Transform two electron integrals
    eri_mo = ao2mo.kernel(mf.mol, mo_coeff, aosym="s1") #TODO figure out what this symmetry means here
    h2e = eri_mo.reshape((nmo, nmo, nmo, nmo))

    return ecore, h1e, h2e

def get_fermionic_hamiltonian_active_space(mf,ncas,nelecas):
    """ 
    Construct the fermionic Hamiltonian using CASCI for active space calculation
    Args:
        mf: PySCF RHF object after SCF calculation converged
        ncas (int): Number of orbitals in complete active space
        nelecas: Tuple(n_alpha, n_beta) Number of electrons in active space

    Returns:
        h1e: One-electron integrals for active space
        h2e: Two-electron integrals for active space
        ecore: Core energy including frozen orbitals
    """
    mx = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)

    # Run CASCI calculation 
    mx.kernel()

    # Extract effective Hamiltonian 
    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)
    return h1e, h2e, ecore



# Logging functions

def write_hamiltonian_out(ecore, h1e, h2e, filepath):
    """   
    Helper function to write Hamiltonian components to output file.
    """
    with open(filepath, 'a') as f:
        f.write("=== Hamiltonian Components ===\n")
        f.write(f"Core Energy (Nuclear Repulsion): {ecore}\n")
        f.write("One-Electron Integrals (h1e):\n")
        np.savetxt(f, h1e, fmt="%.6f")
        f.write("Two-Electron Integrals (h2e):\n")
        h2e_flat = h2e.reshape(h2e.shape[0], -1)
        np.savetxt(f, h2e_flat, fmt="%.6f")
        f.write("\n")

    
