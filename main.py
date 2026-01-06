from __future__ import annotations

from modules.jobs import (
    run_h2_noise_benchmark,
    run_h2_bond_scan,
    run_lih_bond_scan,
    run_hf_bond_scan,
    run_h2_joint_comparison,
    run_h2o_joint_optimization,
)


if __name__ == "__main__":
    # run_h2_noise_benchmark()
    # run_h2_bond_scan()
    run_lih_bond_scan()
    run_hf_bond_scan()
    # run_h2_joint_comparison()
    # run_h2o_joint_optimization()
