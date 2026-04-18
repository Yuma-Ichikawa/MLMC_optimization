"""Bit-level equivalence test for the optimised kernels.

For each target kernel we:

1. Seed ``torch`` (and CUDA) deterministically.
2. Run a short annealing pass using the **original** implementation.
3. Seed ``torch`` again to the same value.
4. Run the same pass using the **optimised** implementation via ``install()``.
5. Assert every returned tensor matches exactly (``torch.equal``).

If any assertion fails the script exits non-zero so it can be used as a
release gate. All tests run on GPU (CUDA required).

Usage::

    python Reproduction/speedups/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Code" / "Legacy" / "packages"))
sys.path.insert(0, str(REPO_ROOT / "Code" / "Modern" / "optimization"))
sys.path.insert(0, str(REPO_ROOT / "Reproduction" / "speedups"))

from monte_carlo import Observables, get_indices, read_couplings  # noqa: E402
from simulated_annealing import simulated_annealing  # noqa: E402
from population_annealing import population_annealing  # noqa: E402
from global_annealing import global_annealing  # noqa: E402

import kernels  # noqa: E402


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _run_sa(L, J, pop_size, num_temps, Tstart, Tend, seed):
    _seed_all(seed)
    t, observ, _ = simulated_annealing(
        L, J, pop_size, num_steps_MC=15, N=L ** 3,
        Tstart=Tstart, Tend=Tend, Observables=Observables,
        schedule="logT", num_temps_determiner=num_temps,
    )
    return (observ.get_observable_history("min_energy"),
            observ.get_observable_history("mean_energy"))


def _run_pa(L, J, pop_size, num_temps, Tstart, Tend, seed):
    _seed_all(seed)
    t, observ, _ = population_annealing(
        L, J, pop_size, num_steps_MC=10,
        Tstart=Tstart, Tend=Tend, Observables=Observables,
        schedule="logT", num_temps_determiner=num_temps,
    )
    return (observ.get_observable_history("min_energy"),
            observ.get_observable_history("mean_energy"))


def _run_ga(L, J, pop_size, num_temps, Tstart, Tend, seed):
    _seed_all(seed)
    t, observ, _ = global_annealing(
        L, J, pop_size, num_steps_MC=5, swap_step=15, N=L ** 3,
        Tstart=Tstart, Tend=Tend, Observables=Observables,
        schedule="logT", num_temps_determiner=num_temps,
    )
    return (observ.get_observable_history("min_energy"),
            observ.get_observable_history("mean_energy"))


def _compare(name, orig, opt):
    a1, a2 = np.asarray(orig[0]), np.asarray(opt[0])
    b1, b2 = np.asarray(orig[1]), np.asarray(opt[1])
    if not (np.array_equal(a1, a2) and np.array_equal(b1, b2)):
        max_a = float(np.max(np.abs(a1 - a2))) if a1.shape == a2.shape else float("nan")
        max_b = float(np.max(np.abs(b1 - b2))) if b1.shape == b2.shape else float("nan")
        print(f"[FAIL] {name}: min_energy max|Δ|={max_a:.3e}, "
              f"mean_energy max|Δ|={max_b:.3e}")
        return False
    print(f"[ OK ] {name}: min_energy and mean_energy trajectories identical")
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("verify.py requires a GPU.", file=sys.stderr)
        return 2

    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    L = 6
    N = L ** 3
    pop_size = 1024
    num_temps = 12
    Tstart, Tend = 1.92, 0.1
    seed = 1736329224

    couplings = REPO_ROOT / f"Data/Alpha/Couplings/couplings_L{L}_R1_seed{seed}.txt"
    if not couplings.exists():
        print(f"generating couplings at L={L}, seed={seed}...")
        from subprocess import check_call
        check_call([
            sys.executable,
            str(REPO_ROOT / "Reproduction/code/generate_coupling.py"),
            "--L", str(L), "--seed", str(seed),
        ], cwd=REPO_ROOT)
    J = read_couplings(str(couplings), N).cuda()

    ok = True
    for label, fn in (("SA", _run_sa), ("PA", _run_pa), ("GA", _run_ga)):
        print(f"--- {label} ---")
        kernels.uninstall()
        orig = fn(L, J, pop_size, num_temps, Tstart, Tend, seed=7)
        kernels.install()
        opt = fn(L, J, pop_size, num_temps, Tstart, Tend, seed=7)
        kernels.uninstall()
        ok &= _compare(label, orig, opt)

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
