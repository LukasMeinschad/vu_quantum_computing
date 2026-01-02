"""
Debug script to understand why VQE doesn't converge
"""
import numpy as np
from qiskit_aer import AerSimulator
import modules.molecule as molecule
import modules.hamiltonian as hamiltonian
import modules.ansatz as ansatz_module
import modules.optimization as optimization
import matplotlib.pyplot as plt

# Build H2 at equilibrium distance
mol = molecule.build_molecule_from_xyz("./test_molecules/h2.xyz", charge=0, spin=0, basis="sto-3g", symmetry=True)
mf = hamiltonian.run_scf_calculation(mol)
ecore, h1e, h2e = hamiltonian.get_casci_hamiltonian(mf, ncas=2, nelecas=(1,1))
H = hamiltonian.build_hamiltonian(ecore, h1e, h2e, "jordan_wigner")

# Get exact ground state energy
H_matrix = H.to_matrix()
eigenvalues = np.linalg.eigvalsh(H_matrix)
exact_energy = np.min(eigenvalues)
print(f"Exact ground state energy: {exact_energy:.8f} Ha\n")

backend = AerSimulator()

# Test 1: Different initial parameters
print("="*60)
print("TEST 1: Impact of Initial Parameters")
print("="*60)

initial_params_sets = {
    "zeros": lambda n: np.zeros(n),
    "small_random": lambda n: 0.01 * np.random.randn(n),
    "random": lambda n: np.random.randn(n),
    "pi_random": lambda n: 2 * np.pi * np.random.rand(n),
}

for name, param_gen in initial_params_sets.items():
    print(f"\n{name}:")
    
    ansatz = ansatz_module.create_ansatz(
        num_qubits=4,
        ansatz_type="efficient_su2",
        reps=2,
        entanglement="full",
        num_electrons=(1, 1),
        use_hf_initial_state=True,
    )
    
    initial_params = param_gen(ansatz.num_parameters)
    
    # Evaluate initial energy
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    estimator = BackendEstimatorV2(backend=backend)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)
    
    initial_energy = optimization.cost_func(initial_params, ansatz_isa, H, estimator)
    print(f"  Initial energy: {initial_energy:.8f} Ha")
    print(f"  Distance to exact: {abs(initial_energy - exact_energy):.8f} Ha")

# Test 2: Different ansatz depths
print("\n" + "="*60)
print("TEST 2: Impact of Ansatz Depth (with HF init)")
print("="*60)

for reps in [1, 2, 3, 4]:
    print(f"\nreps={reps}:")
    
    ansatz = ansatz_module.create_ansatz(
        num_qubits=4,
        ansatz_type="efficient_su2",
        reps=reps,
        entanglement="full",
        num_electrons=(1, 1),
        use_hf_initial_state=True,
    )
    
    print(f"  Parameters: {ansatz.num_parameters}")
    print(f"  Depth: {ansatz.depth()}")
    
    # Run VQE
    from scipy.optimize import minimize
    
    estimator = BackendEstimatorV2(backend=backend)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)
    
    initial_params = 0.01 * np.random.randn(ansatz.num_parameters)
    
    history = []
    def callback(xk):
        e = optimization.cost_func(xk, ansatz_isa, H, estimator)
        history.append(e)
    
    result = minimize(
        fun=optimization.cost_func,
        x0=initial_params,
        args=(ansatz_isa, H, estimator),
        method="COBYLA",
        options={"maxiter": 300},
        callback=callback,
    )
    
    print(f"  Final energy: {result.fun:.8f} Ha")
    print(f"  Error: {abs(result.fun - exact_energy):.8f} Ha")
    print(f"  Iterations: {len(history)}")
    print(f"  Converged: {result.success}")
    
    # Check if stuck in local minimum
    if len(history) > 10:
        last_10_std = np.std(history[-10:])
        print(f"  Last 10 iterations std: {last_10_std:.2e}")

# Test 3: Check if ansatz can represent ground state
print("\n" + "="*60)
print("TEST 3: Can ansatz represent ground state?")
print("="*60)

ansatz = ansatz_module.create_ansatz(
    num_qubits=4,
    ansatz_type="efficient_su2",
    reps=3,
    entanglement="full",
    num_electrons=(1, 1),
    use_hf_initial_state=True,
)

# Try many random initializations
min_energy_found = float('inf')
best_params = None

print(f"\nTrying 20 random initializations...")
for trial in range(20):
    initial_params = 0.01 * np.random.randn(ansatz.num_parameters)
    
    estimator = BackendEstimatorV2(backend=backend)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)
    
    result = minimize(
        fun=optimization.cost_func,
        x0=initial_params,
        args=(ansatz_isa, H, estimator),
        method="COBYLA",
        options={"maxiter": 200},
    )
    
    if result.fun < min_energy_found:
        min_energy_found = result.fun
        best_params = result.x
        print(f"  Trial {trial}: New best = {result.fun:.8f} Ha")

print(f"\nBest energy found: {min_energy_found:.8f} Ha")
print(f"Exact energy: {exact_energy:.8f} Ha")
print(f"Error: {abs(min_energy_found - exact_energy):.8f} Ha")

# Test 4: Try different optimizers
print("\n" + "="*60)
print("TEST 4: Different Optimizers")
print("="*60)

ansatz = ansatz_module.create_ansatz(
    num_qubits=4,
    ansatz_type="efficient_su2",
    reps=3,
    entanglement="full",
    num_electrons=(1, 1),
    use_hf_initial_state=True,
)

optimizers = ["COBYLA", "Powell", "BFGS", "L-BFGS-B", "SLSQP"]

for opt_name in optimizers:
    print(f"\n{opt_name}:")
    
    initial_params = 0.01 * np.random.randn(ansatz.num_parameters)
    
    estimator = BackendEstimatorV2(backend=backend)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)
    
    history = []
    def callback(xk):
        e = optimization.cost_func(xk, ansatz_isa, H, estimator)
        history.append(e)
    
    try:
        result = minimize(
            fun=optimization.cost_func,
            x0=initial_params,
            args=(ansatz_isa, H, estimator),
            method=opt_name,
            options={"maxiter": 300},
            callback=callback,
        )
        
        print(f"  Final energy: {result.fun:.8f} Ha")
        print(f"  Error: {abs(result.fun - exact_energy):.8f} Ha")
        print(f"  Success: {result.success}")
        
        if len(history) > 0:
            improvement = history[0] - result.fun
            print(f"  Improvement: {improvement:.8f} Ha")
    except Exception as e:
        print(f"  Failed: {e}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)