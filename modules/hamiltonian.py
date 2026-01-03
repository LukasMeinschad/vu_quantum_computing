from pyscf import ao2mo, gto, mcscf, scf
from qiskit.quantum_info import SparsePauliOp
import numpy as np

import modules.mapping as mapping


def get_hf_hamiltonian(mf):
    """
    Gets the Hartree-Fock fermionic Hamiltonian from a converged SCF object.

    Args:
        mf: Converged SCF object; mf = mean_field

    Returns:
        ecore: Core energy (nuclear repulsion).
        h1e: One-electron integrals.
        h2e: Two-electron integrals.
        E_reference
    """
    ecore = mf.energy_nuc()

    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]

    # Transform one-electron integrals
    h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff

    # Transform two electron integrals
    eri_mo = ao2mo.kernel(
        mf.mol, mo_coeff, aosym="s1"
    )  # TODO figure out what this symmetry means here
    h2e = eri_mo.reshape((nmo, nmo, nmo, nmo))

    return ecore, h1e, h2e


def get_casci_hamiltonian(mf, ncas, nelecas):
    """
    Construct the fermionic Hamiltonian using CASCI for active space calculation
    Args:
        mf: PySCF RHF object after SCF calculation converged; mf = mean_field
        ncas (int): Number of orbitals in complete active space
        nelecas: Tuple(n_alpha, n_beta) Number of electrons in active space

    Returns:
        h1e: One-electron integrals for active space
        h2e: Two-electron integrals for active space
        ecore: Core energy including frozen orbitals
    """
    mx = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
    mx.kernel()

    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

    return ecore, h1e, h2e


def run_scf_calculation(mol, method="RHF", verbose=False):
    """
    Run SCF calculation on the given molecule object.

    Args:
        mol: PySCF molecule object.
        method (str): SCF method to use ("RHF", "UHF", or "ROHF").
        verbose (bool): Whether to print SCF details.

    Returns:
        mf: Converged SCF object; mf = mean_field
    """
    if method == "RHF":
        mf = scf.RHF(mol)
    elif method == "UHF":
        mf = scf.UHF(mol)
    elif method == "ROHF":
        mf = scf.ROHF(mol)
    else:
        raise ValueError(f"Unsupported SCF method: {method}")

    mf.verbose = 4 if verbose else 0
    mf.kernel()

    if not mf.converged:
        raise RuntimeError("SCF did not converge!")

    return mf


def build_fermionic_hamiltonian_diatomic(
    distance: float,
    atom_labels: list[str],
    basis: str,
    spin: int,
    charge: int,
    symmetry: bool,
    ncas: int,
    nelecas: tuple,
):
    """
    Build Hamiltonian terms for a diatomic molecule based on bond distance x (in Angstrom)

    Args:
        distance (float): Bond distance in Angstrom
        atom_labels: List of two atom symbols, e.g., ["H", "H"]
        basis (str): Basis set for calculation
        spin (int): Spin multiplicity
        charge (int): Molecular charge
        symmetry (bool): Whether to use molecular symmetry
        ncas (int): Number of active space orbitals
        nelecas (tuple): Number of active space electrons (alpha, beta)


    """
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[
            [atom_labels[0], (0.0, 0.0, 0.0)],
            [atom_labels[1], (0.0, 0.0, distance)],
        ],
        basis=basis,
        spin=spin,
        charge=charge,
        symmetry=symmetry,
    )

    mf = scf.RHF(mol)
    mf.kernel()

    if not mf.converged:
        raise ValueError("SCF calculation did not converge.")

    return get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)


def build_qubit_hamiltonian(
    ecore: float, h1e: np.ndarray, h2e: np.ndarray, mapping_method="jordan_wigner"
) -> SparsePauliOp:
    """
    Buils the qubit Hamiltonian from fermionic integrals

    Fermionic Hamiltonian is:
    H = E_core + sum_{pq} h1e_{pq} a_p^† a_q + 0.5 sum_{pqrs} h2e_{pqrs} a_p^† a_q^† a_r a_s
    """
    ncas, _ = h1e.shape

    # Get creation and annihilation operators
    # 2 * ncas to account for spin-orbitals
    C, D = mapping.creators_destructors(ncas * 2, mapping=mapping_method)

    # Build excitation operators c_p^† c_r for all p,r
    # Exc[p][r-p] represents excitation from orbital r to orbital p
    # Do this for both spins
    Exc = []
    for p in range(ncas):
        Excp = [
            C[p] @ D[p] + C[ncas + p] @ D[ncas + p]
        ]  # Number of operators if p == r
        for r in range(p + 1, ncas):
            Excp.append(
                C[p] @ D[r]  # \alpha spin p -> r
                + C[ncas + p] @ D[ncas + r]  # \beta spin p -> r
                + C[r] @ D[p]  # \alpha spin r -> p
                + C[ncas + r] @ D[ncas + p]  # \beta spin r -> p
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
            H += t1e[p, r] * Exc[p][r - p]

    # Add two body terms
    for g in range(ng):
        Lg = 0 * mapping.identity(ncas * 2)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p, r, g] * Exc[p][r - p]
        # Square operator
        H += 0.5 * (Lg @ Lg)

    H = H.chop().simplify()

    # Combine like terms and remove small coefficients
    return H.chop().simplify()


def build_triatomic_hamiltonian(
    geometry_params, atom_labels, basis="sto-3g", ncas=4, nelecas=(2, 2)
):
    """
    Build Hamiltonian for a three-atomic molecule with given geometry parameters

    Assumes atom_labels[0] is the central atom with bonds to atoms 1 and 2.

    Args:
        geometry_params: Dictionary with geometry parameters
            For bent (like H2O): {"R1": bond_length_0_to_1, "R2": bond_length_0_to_2, "theta": angle_in_degrees}
            For linear: {"R1": bond1_length, "R2": bond2_length}
        atom_labels: List of three atom symbols where [0] is central, e.g., ["O", "H", "H"]
        basis: Basis set for calculation
        ncas: Number of active space orbitals
        nelecas: Number of active space electrons (alpha, beta)

    Returns:
        ecore, h1e, h2e: Hamiltonian components
    """

    # Build molecular geometry based on parameters
    R1 = geometry_params["R1"]
    R2 = geometry_params["R2"]

    if "theta" in geometry_params:
        # Bent molecule (e.g., H2O) - central atom at origin
        theta = geometry_params["theta"] * np.pi / 180.0  # Convert to radians

        # Central atom at origin, first atom along z-axis, second in xz-plane
        coords = [
            [atom_labels[0], (0.0, 0.0, 0.0)],  # Central atom (e.g., O) at origin
            [atom_labels[1], (0.0, 0.0, R1)],  # First peripheral atom along z-axis
            [
                atom_labels[2],
                (R2 * np.sin(theta), 0.0, R2 * np.cos(theta)),
            ],  # Second peripheral atom in xz-plane
        ]
    else:
        # Linear molecule - central atom at origin
        coords = [
            [atom_labels[0], (0.0, 0.0, 0.0)],  # Central atom at origin
            [atom_labels[1], (R1, 0.0, 0.0)],  # First atom along x-axis
            [atom_labels[2], (-R2, 0.0, 0.0)],  # Second atom along negative x-axis
        ]

    mol = gto.Mole()
    mol.build(verbose=0, atom=coords, basis=basis, spin=0, charge=0, symmetry=False)

    mf = scf.RHF(mol)
    mf.kernel()

    if not mf.converged:
        raise ValueError("SCF calculation did not converge.")

    return get_casci_hamiltonian(mf, ncas=ncas, nelecas=nelecas)


# Logging functions


def write_hamiltonian_out(ecore, h1e, h2e, filepath, label="Hamiltonian"):
    """Helper function to write Hamiltonian components to output file.

    Args:
        ecore: Core (nuclear repulsion / frozen-core) energy in Hartree.
        h1e: One-electron integrals in Hartree.
        h2e: Two-electron integrals in Hartree.
        filepath: Output file path.
        label: Short description of the Hamiltonian (e.g. "Hartree-Fock" or "CASCI active space").
    """
    with open(filepath, "a") as f:
        f.write(f"=== {label} Hamiltonian Components ===\n")
        f.write(f"Core Energy (Nuclear Repulsion): {ecore} Hartree\n")
        f.write("One-Electron Integrals (h1e) in Hartree:\n")
        np.savetxt(f, h1e, fmt="%.6f")
        f.write("Two-Electron Integrals (h2e) in Hartree:\n")
        h2e_flat = h2e.reshape(h2e.shape[0], -1)
        np.savetxt(f, h2e_flat, fmt="%.6f")
        f.write("\n")


def write_qubit_hamiltonian_out(H, out_file):
    """
    Writes the qubit Hamiltonian to the output file.

    Args:
        H: Qubit Hamiltonian as SparsePauliOp.
        out_file (str): Path to the output file.
    """
    with open(out_file, "a") as f:
        f.write("=== Qubit Hamiltonian ===\n")
        f.write(f"Number of qubits: {H.num_qubits}\n")
        f.write(f"Number of Pauli terms: {len(H.paulis)}\n")
        f.write("Qubit Hamiltonian terms:\n")
        for pauli, coeff in zip(H.paulis, H.coeffs):
            f.write(f"{coeff.real:.6f} * {pauli}\n")
        f.write("\n")
