"""Qubit-Hamiltonian utilities.

This module contains helpers to map second-quantized (fermionic) operators coming
from Qiskit Nature into qubit operators suitable for VQE.

Currently supported mappers:
- Jordan-Wigner
- Bravyi-Kitaev
- Parity (optionally with 2-qubit tapering when num_particles is available)
"""

from __future__ import annotations

from typing import Literal

from modules.molecule import build_fermionic_hamiltonian


QubitMapperName = Literal["JordanWigner", "BravyiKitaev", "Parity"]


def map_to_qubit_hamiltonian(problem, mapper: QubitMapperName = "JordanWigner"):
	"""Map an ElectronicStructureProblem Hamiltonian to a qubit operator.

	Parameters
	----------
	problem:
		Qiskit Nature ElectronicStructureProblem (optionally already transformed,
		e.g. via FreezeCore/ActiveSpace).
	mapper:
		Mapping method name.

	Returns
	-------
	SparsePauliOp (typically)
		Qubit operator corresponding to the electronic Hamiltonian.
	"""

	from qiskit_nature.second_q.mappers import (
		BravyiKitaevMapper,
		JordanWignerMapper,
		ParityMapper,
	)

	second_q_hamiltonian = build_fermionic_hamiltonian(problem)

	if mapper == "JordanWigner":
		qubit_mapper = JordanWignerMapper()
	elif mapper == "BravyiKitaev":
		qubit_mapper = BravyiKitaevMapper()
	elif mapper == "Parity":
		qubit_mapper = ParityMapper(num_particles=problem.num_particles)
	else:
		raise ValueError(f"Unsupported mapper: {mapper}")

	return qubit_mapper.map(second_q_hamiltonian)

