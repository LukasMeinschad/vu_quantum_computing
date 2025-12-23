import numpy as np
from qiskit.quantum_info import SparsePauliOp

def cholesky(V, eps=1e-5):
    """
    Cholesky decomposition of two-electron integrals
    
    Args:
        V: Two-electron integral tensor (no, no, no, no)
        eps: Threshold for decomposition accuracy
    
    Returns:
        L: Cholesky vectors (no, no, ng)
        ng: Number of Cholesky vectors
    
    References:
        - https://arxiv.org/pdf/1711.02242.pdf section B2
        - https://arxiv.org/abs/1808.02625
        - https://arxiv.org/abs/2104.08957
    """
    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "Accuracy of Cholesky decomposition:",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng


def identity(n):
    """  
    Builds the Identity operator for n qubits

    Returns:
        SparsePauliOp: Identity operator on n qubits I^n
    """
    return SparsePauliOp.from_list([("I" * n, 1.0 )])


def bravyi_kitaev_update_set(n, j):
    """  
    Helper function to calculate the update Set U(j) for Bravyi-Kitaev mapping

    The update set U(j) contains indices of qubits that store occupation information
    that must be updated when orbital j changes

    Args:
        n (int): Number of spin-orbitals
        j (int): Orbital index
    
    Returns:
        list: Indices in the update set
    """
    update_set = [j]
    
    # Find parent nodes in the binary tree
    # Add the bit length that j covers
    k = j
    while k < n - 1:
        # Find the next update position by flipping the rightmost 0 bit
        # This follows the binary tree structure
        k = k + ((k + 1) & (~k))
        if k < n:
            update_set.append(k)
        else:
            break
    
    return update_set

def bravyi_kitaev_parity_set(n, j):
    """
    Calculate the parity set P(j) for Bravyi-Kitaev mapping

    The parity set P(j) contains indices of qubits that must be checked
    to determine the parity of orbitals with indices less than j

    Args:
        n (int): Number of spin-orbitals
        j (int): Orbital index

    Returns:
        list: List of qubit indices in the parity set P(j)
    """
    if j == 0:
        return []
    
    parity_set = []
    
    # Find all positions that store parity information needed for position j
    # We need to traverse back through the binary tree
    k = j - 1
    
    # Keep removing the rightmost set bit to find parent nodes
    while k >= 0:
        parity_set.append(k)
        # Remove rightmost set bit: k & (k-1)
        k_new = k & (k - 1)
        if k_new == k or k == 0:
            break
        k = k_new - 1
        if k < 0:
            break
    
    return parity_set



def creators_destructors(n, mapping="jordan_wigner"):
    """  
    Generate creation and annihilation operators using different mappings
    
    Jordan-Wigner Transformation for fermionic operators:
    - Creation operator a_j^dagger: (X_p - iY_p)/2 x Z_1 x Z_2 ... x Z_(j-1)
    - Annihilation operator a_j: (X_p + iY_p)/2 x Z_1 x Z_2 ... x Z_(j-1)

    Note that the Z chain encodes the fermionic anticommutation relations

    Bravyi-Kitaev Transformation:
    - Stores occupation and parity information in logarithmic number of qubits
    - Creation operator: a_j^dagger = 1/2 (X_U(j) x X_j x Z_P(j) - iX_U(j) x Y_j x Z_P(j))
    - Annihilation operator: a_j = 1/2 (X_U(j) x X_j x Z_P(j) + iX_U(j) x Y_j x Z_P(j))
    
    Args:
        n (int): Number of spin-orbitals
        mapping (str): Mapping method ("jordan_wigner", "parity", "bravyi_kitaev")
    
    Returns:
        creators (list): List of creation operators as SparsePauliOp
        destructors (list): List of annihilation operators as SparsePauliOp
    """
    c_list = []
    if mapping == "jordan_wigner":
        for p in range(n):
            # Construct Pauli String for position p
            # Left side: Identity operators for qubits > p
            # Right side: Z operators for qubits < p
            if p == 0:
                ell, r = "I" *(n-1), ""
            elif p == n-1:
                ell, r = "", "Z" * (n-1)
            else:
                ell, r = "I" * (n-p-1), "Z" * p

            # Creation operator a_p^dagger
            cp = SparsePauliOp.from_list([
                (ell + "X" + r, 0.5),
                (ell + "Y" + r, -0.5j)
            ])
            c_list.append(cp)
    elif mapping == "bravyi_kitaev":
        for j in range(n):
            U_j = bravyi_kitaev_update_set(n,j)
            P_j = bravyi_kitaev_parity_set(n,j)

            # Build pauli string for this orbital
            # Initialize with identity
            pauli_x = ["I"] * n
            pauli_y = ["I"] * n

            # Apply X to update set (all qubits in U(j))
            for idx in U_j:
                pauli_x[n-1 - idx] = "X"
                pauli_y[n-1 - idx] = "X"
            
            # Apply central operator at position j 
            pauli_x[n-1 - j] = "X"
            pauli_y[n-1 - j] = "Y"

            # Apply Z to parity set P(j)
            for idx in P_j:
                pauli_x[n-1 - idx] = "Z"
                pauli_y[n-1 - idx] = "Z"

            pauli_str_x = "".join(pauli_x)
            pauli_str_y = "".join(pauli_y)

            # Creation operator a_j^dagger
            cp = SparsePauliOp.from_list([
                (pauli_str_x, 0.5),
                (pauli_str_y, -0.5j)
            ])
            c_list.append(cp)
    
    else:
        raise ValueError(f"Mapping {mapping} not implemented.")

    # Annhilation operatos are Hermitian adjoints
    d_list = [c.adjoint() for c in c_list]
    return c_list, d_list

# Helper functions

def build_hamiltonian_helper(ecore,h1e,h2e,C,D,ncas):
    """   
    Helper function to test the differences of the Jordan-Wigner and Bravyi-Kitaev mappings
    """
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
    Lop, ng = cholesky(h2e, eps=1e-5)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
    H = ecore * identity(ncas * 2)
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p,r] * Exc[p][r - p]
    # Add two body terms
    for g in range(ng):
        Lg = 0 * identity(ncas * 2)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p,r,g] * Exc[p][r - p]
        H += 0.5 * (Lg @ Lg)
    return H.chop().simplify()

def compare_mappings(ecore,h1e,h2e):
    """  
    Compares the Jordan-Wigner and Bravyi-Kitaev mapping for the same fermionic Hamiltonian
    """
    print("\n === Comparing Jordan-Wigner and Bravyi-Kitaev Mappings ===")

    print("\n-- Jordan-Wigner Mapping --")
    import time
    start_jw = time.time()
    ncas, _ = h1e.shape
    C_jw, D_jw = creators_destructors(ncas * 2, mapping="jordan_wigner")
    H_jw = build_hamiltonian_helper(ecore,h1e,h2e,C_jw,D_jw,ncas)
    end_jw = time.time()
    print(f"Jordan-Wigner Hamiltonian has {len(H_jw.paulis)} terms")
    print(f"Time taken for Jordan-Wigner: {end_jw - start_jw:.4f} seconds")

    print("\n-- Bravyi-Kitaev Mapping --")
    start_bk = time.time()
    C_bk, D_bk = creators_destructors(ncas * 2, mapping="bravyi_kitaev")
    H_bk = build_hamiltonian_helper(ecore,h1e,h2e,C_bk,D_bk,ncas)
    end_bk = time.time()
    print(f"Bravyi-Kitaev Hamiltonian has {len(H_bk.paulis)} terms")
    print(f"Time taken for Bravyi-Kitaev: {end_bk - start_bk:.4f} seconds")

    # Compare Speedup and Term Reduction
    print(f"\nSpeedup (BK vs JW): {(end_jw - start_jw)/(end_bk - start_bk):.2f}x")
    print(f"Term Reduction (BK vs JW): {len(H_jw.paulis)/len(H_bk.paulis):.2f}x")

    return {
    "H_jw": H_jw,
    "H_bk": H_bk,
    "time_jw": end_jw - start_jw,
    "time_bk": end_bk - start_bk
    }