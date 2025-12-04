"""Batch launcher for expert-iteration sweeps."""

import subprocess
from typing import Dict, List

EXPERIMENTS: List[Dict[str, int]] = [
    # A: Sweep G (6 runs)
    {"G": 4, "n_epochs": 3, "D_B": 1024},
    {"G": 8, "n_epochs": 3, "D_B": 1024},
    {"G": 16, "n_epochs": 3, "D_B": 1024},
    {"G": 24, "n_epochs": 3, "D_B": 1024},
    {"G": 32, "n_epochs": 3, "D_B": 1024},
    {"G": 48, "n_epochs": 3, "D_B": 1024},
    # B: Sweep n_epochs (5 runs)
    {"G": 16, "n_epochs": 1, "D_B": 1024},
    {"G": 16, "n_epochs": 2, "D_B": 1024},
    {"G": 16, "n_epochs": 3, "D_B": 1024},
    {"G": 16, "n_epochs": 6, "D_B": 1024},
    {"G": 16, "n_epochs": 10, "D_B": 1024},
    # C: Sweep D_B (4 runs)
    {"G": 16, "n_epochs": 3, "D_B": 512},
    {"G": 16, "n_epochs": 3, "D_B": 1024},
    {"G": 16, "n_epochs": 3, "D_B": 1536},
    {"G": 16, "n_epochs": 3, "D_B": 2048},
]


def run_experiment(index: int, exp: Dict[str, int]) -> int:
    """Execute a single expert_iteration.py run with provided hyperparams."""
    print(
        f"\n=== Running exp {index + 1} / {len(EXPERIMENTS)}: "
        f"G={exp['G']}, n_epochs={exp['n_epochs']}, D_B={exp['D_B']} ==="
    )

    cmd = [
        "python",
        "expert_iteration.py",
        f"--G={exp['G']}",
        f"--n-epochs={exp['n_epochs']}",
        f"--D-B={exp['D_B']}",
    ]

    print("Command:", " ".join(cmd))
    proc = subprocess.run(cmd)

    if proc.returncode != 0:
        print(f"[WARN] Experiment {index + 1} FAILED (code={proc.returncode})")
    else:
        print(f"[OK] Experiment {index + 1} PASSED")

    return proc.returncode


def main() -> int:
    failed = 0
    for i, exp in enumerate(EXPERIMENTS):
        if run_experiment(i, exp) != 0:
            failed += 1

    print(f"\nAll experiments finished. Failed {failed}/{len(EXPERIMENTS)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
