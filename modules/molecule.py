"""
Molecule utilities.

This module provides helpers to:
- read an .xyz file and convert it into a PySCF-compatible atom string
- build a Qiskit Nature ElectronicStructureProblem via PySCFDriver
- optionally apply Freeze-Core and/or Active-Space reductions
- extract the second-quantized (fermionic) Hamiltonian
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import (
    ActiveSpaceTransformer,
    FreezeCoreTransformer,
)
from qiskit_nature.units import DistanceUnit


@dataclass(frozen=True)
class MoleculeSpec:
    """
    Configuration for the PySCF driver input.

    Parameters
    ----------
    atom:
        Either a PySCF atom string (recommended format: ';'-separated),
        e.g. "H 0 0 -0.37; H 0 0 0.37",
        or a Qiskit Nature MoleculeInfo.
    basis, charge, spin:
        Standard PySCFDriver settings.
    driver_kwargs:
        Extra keyword arguments passed to PySCFDriver.
    """

    atom: str | MoleculeInfo
    basis: str
    charge: int
    spin: int
    driver_kwargs: dict[str, Any] | None = None


def xyz_file_to_pyscf_atom_string(xyz_path: str | Path) -> str:
    """
    Read an .xyz file and convert it into a PySCF-compatible atom string.

    The XYZ format is expected as:
        line 1: number of atoms
        line 2: comment
        next N lines: "Element x y z [ignored extras...]"

    Returns
    -------
    str
        PySCF atom string: "H 0 0 0; H 0 0 0.74"
    """
    path = Path(xyz_path)
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"XYZ file too short: {path}")

    try:
        n = int(lines[0])
    except ValueError as exc:
        raise ValueError(
            f"First line must be the atom count, got: {lines[0]!r}"
        ) from exc

    body = lines[2 : 2 + n]
    if len(body) != n:
        raise ValueError(
            f"XYZ declares {n} atoms but contains {len(body)} coordinate lines."
        )

    atoms: list[str] = []
    for ln in body:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ line (need 'El x y z'): {ln!r}")
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(f"{sym} {x} {y} {z}")

    return "; ".join(atoms)


def to_pyscf_atom_string(molecule: str | MoleculeInfo) -> str:
    """Convert MoleculeInfo -> PySCF atom string; keep str unchanged."""
    if isinstance(molecule, MoleculeInfo):
        parts: list[str] = []
        for sym, (x, y, z) in molecule.geometry:
            parts.append(f"{sym} {float(x)} {float(y)} {float(z)}")
        return "; ".join(parts)
    return molecule


def sanitize_active_space(problem, active_space: tuple[int, int]) -> tuple[int, int]:
    """
    Make (num_electrons, num_spatial_orbitals) feasible for ActiveSpaceTransformer.

    Enforced conditions:
    - 1 <= active_orbitals <= total_orbitals
    - 0 <= active_electrons <= total_electrons
    - active_electrons <= 2 * active_orbitals
    - inactive_electrons <= 2 * inactive_orbitals
    """
    req_e, req_o = map(int, active_space)

    total_orb = int(problem.num_spatial_orbitals)
    total_e = int(sum(problem.num_particles))  # (n_alpha, n_beta) -> total

    req_o = max(1, min(req_o, total_orb))
    req_e = max(0, min(req_e, total_e))

    req_e = min(req_e, 2 * req_o)

    inactive_e = total_e - req_e
    inactive_orb = total_orb - req_o

    if inactive_e > 2 * inactive_orb:
        needed_inactive_orb = int(np.ceil(inactive_e / 2))
        req_o = max(1, total_orb - needed_inactive_orb)
        req_e = min(req_e, 2 * req_o)

    return req_e, req_o


def build_molecule_problem(
    spec: MoleculeSpec,
    *,
    freeze_core: bool = False,
    active_space: (
        tuple[int, int] | None
    ) = None,  # (num_electrons, num_spatial_orbitals)
    sanitize_active_space: bool = True,
):
    """
    Build a Qiskit Nature ElectronicStructureProblem using PySCFDriver.

    Optionally applies:
    - FreezeCoreTransformer
    - ActiveSpaceTransformer
    """
    atom_str = to_pyscf_atom_string(spec.atom)
    driver = PySCFDriver(
        atom=atom_str,
        basis=spec.basis,
        charge=spec.charge,
        spin=spec.spin,
        unit=DistanceUnit.ANGSTROM,
        **(spec.driver_kwargs or {}),
    )
    problem = driver.run()

    if freeze_core:
        problem = FreezeCoreTransformer().transform(problem)

    if active_space is not None:
        nelec, norb = active_space
        if sanitize_active_space:
            nelec, norb = sanitize_active_space(problem, (nelec, norb))
        problem = ActiveSpaceTransformer(
            num_electrons=nelec, num_spatial_orbitals=norb
        ).transform(problem)

    return problem


def build_fermionic_hamiltonian(problem):
    """
    Extract the second-quantized (fermionic) Hamiltonian from an ElectronicStructureProblem.

    Returns
    -------
    FermionicOp (typically)
    """
    return problem.second_q_ops()[0]
