"""
Ansatz utilities.

This module centralizes ansatz construction for VQE runs.

Supported categories
--------------------
1) Hardware-efficient (problem-agnostic)
    - EfficientSU2
    - TwoLocal
    - RealAmplitudes

2) Chemistry-inspired (problem-specific, Qiskit Nature)
    - HartreeFock (initial state)
    - UCCSD (typically used with HartreeFock)
"""

from __future__ import annotations

import inspect
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, TwoLocal, RealAmplitudes
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import QubitMapper


def ctor_accepts_kwarg(cls, kw: str) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    return kw in sig.parameters


def generate_ansatz(
    num_qubits: int,
    method: str,
    reps: int,
    entanglement: str = "linear",
    *,
    insert_barriers: bool = True,
    # TwoLocal-specific knobs (optional)
    two_local_rotation_blocks="ry",
    two_local_entanglement_blocks="cz",
) -> QuantumCircuit:
    """
    Backwards-compatible hardware-efficient ansatz generator.

    This mirrors the previous `generate_ansatz(...)` you had in main.py.

    Parameters
    ----------
    num_qubits:
        Number of qubits of the target qubit Hamiltonian.
    entanglement, reps:
        Standard circuit settings.
    method:
        One of: "EfficientSU2", "TwoLocal", "RealAmplitudes"
    """
    if method == "EfficientSU2":
        return EfficientSU2(
            num_qubits,
            entanglement=entanglement,
            reps=reps,
            insert_barriers=insert_barriers,
        )

    if method == "TwoLocal":
        return TwoLocal(
            num_qubits,
            rotation_blocks=two_local_rotation_blocks,
            entanglement_blocks=two_local_entanglement_blocks,
            entanglement=entanglement,
            reps=reps,
            insert_barriers=insert_barriers,
        )

    if method == "RealAmplitudes":
        return RealAmplitudes(
            num_qubits,
            entanglement=entanglement,
            reps=reps,
            insert_barriers=insert_barriers,
        )

    raise ValueError(f"Unsupported hardware ansatz method: {method}")


def build_hartree_fock_initial_state(
    problem,
    *,
    qubit_mapper: QubitMapper,
) -> QuantumCircuit:
    """
    Build the Hartree-Fock initial state for a given ElectronicStructureProblem.
    """
    return HartreeFock(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=qubit_mapper,
    )


def build_uccsd_ansatz(
    problem,
    *,
    qubit_mapper: QubitMapper,
    initial_state: Optional[QuantumCircuit] = None,
    reps: int = 1,
) -> QuantumCircuit:
    """
    Build a UCCSD ansatz for a given ElectronicStructureProblem.

    Notes
    -----
    - UCCSD is chemistry-inspired and usually used with a Hartree-Fock initial state.
    - Depending on your installed qiskit-nature version, UCCSD may or may not accept `reps`.
      This function detects that safely.
    """
    if initial_state is None:
        initial_state = build_hartree_fock_initial_state(
            problem, qubit_mapper=qubit_mapper
        )

    kwargs = dict(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=qubit_mapper,
        initial_state=initial_state,
    )

    if ctor_accepts_kwarg(UCCSD, "reps"):
        kwargs["reps"] = reps

    ansatz = UCCSD(**kwargs)

    # fig = ansatz.decompose().draw(output="mpl", fold=-1, style="iqp")
    # fig.savefig("ansatz-uccsd.png", dpi=300, bbox_inches="tight")

    return ansatz


def build_ansatz(
    ansatz_type,
    *,
    num_qubits: int | None = None,
    entanglement: str = "linear",
    reps: int = 1,
    problem=None,
    qubit_mapper: QubitMapper | None = None,
    initial_state: Optional[QuantumCircuit] = None,
) -> QuantumCircuit:
    """
    Unified ansatz builder.

    Examples
    --------
    - Hardware-efficient:
        build_ansatz("EfficientSU2", num_qubits=4, entanglement="linear", reps=2)

    - Chemistry-inspired:
        mapper = JordanWignerMapper()
        build_ansatz("UCCSD", problem=problem, qubit_mapper=mapper)
    """
    if ansatz_type in ("EfficientSU2", "TwoLocal", "RealAmplitudes"):
        if num_qubits is None:
            raise ValueError("num_qubits is required for hardware-efficient ansatz.")
        return generate_ansatz(
            num_qubits=num_qubits,
            method=ansatz_type,  # type: ignore[arg-type]
            reps=reps,
            entanglement=entanglement,
        )

    if ansatz_type == "HartreeFock":
        if problem is None or qubit_mapper is None:
            raise ValueError("Problem and qubit_mapper are required for HartreeFock.")
        return build_hartree_fock_initial_state(problem, qubit_mapper=qubit_mapper)

    if ansatz_type == "UCCSD":
        if problem is None or qubit_mapper is None:
            raise ValueError("Problem and qubit_mapper are required for UCCSD.")
        return build_uccsd_ansatz(
            problem,
            qubit_mapper=qubit_mapper,
            initial_state=initial_state,
            reps=reps,
        )

    raise ValueError(f"Unsupported ansatz type: {ansatz_type}")
