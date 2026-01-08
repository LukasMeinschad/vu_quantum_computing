from __future__ import annotations

from modules.jobs import (
    run_h2_bond_scan,
    run_h2_joint_optimization,
)
from modules.results_io import redirect_stdout_to_results

if __name__ == "__main__":

    with redirect_stdout_to_results():
        run_h2_bond_scan()
        run_h2_joint_optimization()
