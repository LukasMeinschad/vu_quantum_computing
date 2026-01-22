from __future__ import annotations

from modules.jobs import (
    run_h2_noise_benchmark,
    run_h2_bond_scan,
    run_convergence_benchmark,
    run_lih_bond_scan,
    run_hf_bond_scan,
    run_h2_joint_comparison,
    run_h2_joint_optimization,
    run_lih_joint_optimization,
    run_hf_joint_optimization,
    run_h2o_joint_optimization,
)


if __name__ == "__main__":
    
    # Comparisons
    # run_h2_noise_benchmark()
    # run_h2_joint_comparison()
    # run_convergence_benchmark()
    
    # Bond scans
    run_h2_bond_scan()
    # run_lih_bond_scan()
    # run_hf_bond_scan()
    
    # Joint optimizations
    # run_h2_joint_optimization()
    # run_lih_joint_optimization()
    # run_hf_joint_optimization()
    # run_h2o_joint_optimization()
