from pyscf import ao2mo, gto, mcscf, scf
from qiskit.quantum_info import SparsePauliOp
import numpy as np

import modules.mapping as mapping

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

    if method == "RHF":
        mf = scf.RHF(mol)
    elif method == "UHF":
        mf = scf.UHF(mol)
    elif method == "ROHF":
        mf = scf.ROHF(mol)
    else:
        raise ValueError(f"Unsupported SCF method: {method}")
    
    mf.verbose = 4 if verbose else 0
    print("Starting SCF iterations...")
    energy = mf.kernel()

    if not mf.converged:
        print("SCF did not converge!")
    
    print(f"Scf converged, final electronic energy: {energy} Ha")

    return mf

def ham_terms_diatomic(x: float):
    """    
    Build Hamiltonian terms for a diatomic molecule based on bond distance x (in Angstrom)

    Args:
        x (float): Bond distance in Angstrom

    Returns:
        h1e: One-electron integrals
        h2e: Two-electron integrals
        ecore: Core energy
    """
    distance = x 
    a = distance / 2 
    mol = gto.Mole()
    mol.build(
        verbose  = 0,
        atom     = [["H", (0, 0, -a)], ["H", (0, 0, a)]],
        basis    = "sto-6g",
        spin     = 0,
        charge   = 0,
        symmetry = True
    )

    mf = scf.RHF(mol)
    mf.kernel()

    if not mf.converged:
        raise ValueError("SCF calculation did not converge.")
    
    return get_fermionic_hamiltonian_active_space(mf, ncas = 2, nelecas = (1, 1))

def build_hamiltonian_with_geometry(distx: float) -> SparsePauliOp:
    """  
    Build qubit Hamiltonian for H2 molecule at given bond distance

    Args:
        distx (float): Bond distance in Angstrom
    Returns:
        H_qubit: Qubit Hamiltonian as SparsePauliOp
    """
    ecore, h1e, h2e = ham_terms_diatomic(distx)

    ncas, _ = h1e.shape
    C,D = mapping.creators_destructors(ncas * 2, mapping="jordan_wigner")
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]]
        for r in range(p+1, ncas):
            Excp.append(
                C[p] @ D[r]
                + C[ncas + p] @ D[ncas + r]
                + C[r] @ D[p]
                + C[ncas + r] @ D[ncas + p]
            )
        Exc.append(Excp)
    # Low-rank decomposition of h2e
    Lop, ng = mapping.cholesky(h2e, eps=1e-5)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
    H = ecore * mapping.identity(ncas * 2)
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p,r] * Exc[p][r - p]
    # Add two body terms
    for g in range(ng):
        Lg = 0 * mapping.identity(ncas * 2)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p,r,g] * Exc[p][r - p]
        H += 0.5 * (Lg @ Lg)
    return H.chop().simplify()


def build_hamiltonian(ecore: float, h1e: np.ndarray, h2e: np.ndarray, mapping="jordan_wigner") -> SparsePauliOp:
    """ 
    Buils the qubit Hamiltonian from fermionic integrals

    Fermionic Hamiltonian is:
    H = E_core + sum_{pq} h1e_{pq} a_p^† a_q + 0.5 sum_{pqrs} h2e_{pqrs} a_p^† a_q^† a_r a_s
    """

    ncas, _ = h1e.shape

    # Get creation and annihilation operators
    # 2 * ncas to account for spin-orbitals
    C,D = mapping.creators_destructors(ncas * 2, mapping=mapping)

    # Build excitation operators c_p^† c_r for all p,r
    # Exc[p][r-p] represents excitation from orbital r to orbital p
    # Do this for both spins
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]] # Number of operators if p == r
        for r in range(p+1, ncas):
            Excp.append(
                C[p] @ D[r] # \alpha spin p -> r
                + C[ncas + p] @ D[ncas + r] # \beta spin p -> r
                + C[r] @ D[p] # \alpha spin r -> p
                + C[ncas + r] @ D[ncas + p] # \beta spin r -> p
            )
        Exc.append(Excp)

    # Low-rank decomposition of h2e
    Lop, ng = mapping.cholesky(h2e, eps=1e-5)

    # t1e = h1e - 1/2 * Coloumb integral
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)

    # Initialize Hamiltonian with core energy
    H = ecore * mapping.identity(ncas * 2)

    # Add one-body-terms
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p,r] * Exc[p][r - p]

    # Add two body terms
    for g in range(ng):
        Lg = 0 * mapping.identity(ncas * 2)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p,r,g] * Exc[p][r - p]
        # Square operator
        H += 0.5 * (Lg @ Lg)

    # Combine like terms and remove small coefficients
    return H.chop().simplify()

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

