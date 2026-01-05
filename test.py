"""Project test runner.

Runs a few lightweight benchmark-style comparisons and saves plots into
images/.

Run:
    python test.py
"""

from modules.tests import (
    compare_ansatz_types_h2,
    compare_entanglement_and_reps_h2,
    compare_optimizers_h2_uccsd,
)


def main() -> None:
    out1 = compare_ansatz_types_h2()
    print(f"Saved plot: {out1['plot_path']}")

    out2 = compare_optimizers_h2_uccsd()
    print(f"Saved plot: {out2['plot_path']}")

    out3 = compare_entanglement_and_reps_h2()
    print(f"Saved plot: {out3['plot_path']}")


if __name__ == "__main__":
    main()
