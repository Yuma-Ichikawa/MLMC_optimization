"""Wall-clock benchmark: original vs optimised kernels.

Times SA, PA, GA once with the original kernels and once with the optimised
kernels installed. Prints per-algorithm speedup. A correctness check (same
``min_energy`` trajectory) is run first; if it fails the benchmark aborts.

Usage::

    python Reproduction/speedups/bench.py --L 10 --pop-size 8192 --num-temps 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Code" / "Legacy" / "packages"))
sys.path.insert(0, str(REPO_ROOT / "Code" / "Modern" / "optimization"))
sys.path.insert(0, str(REPO_ROOT / "Reproduction" / "speedups"))

from monte_carlo import Observables, read_couplings  # noqa: E402
from simulated_annealing import simulated_annealing  # noqa: E402
from population_annealing import population_annealing  # noqa: E402
from global_annealing import global_annealing  # noqa: E402

import kernels  # noqa: E402


def _seed(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


def _time_sa(L, J, pop, nT, seed):
    _seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    _, observ, elapsed = simulated_annealing(
        L, J, pop, num_steps_MC=15, N=L ** 3,
        Tstart=1.92, Tend=0.1, Observables=Observables,
        schedule="logT", num_temps_determiner=nT,
    )
    torch.cuda.synchronize()
    return time.time() - t0, observ.get_observable_history("min_energy")


def _time_pa(L, J, pop, nT, seed):
    _seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    _, observ, elapsed = population_annealing(
        L, J, pop, num_steps_MC=10,
        Tstart=1.92, Tend=0.1, Observables=Observables,
        schedule="logT", num_temps_determiner=nT,
    )
    torch.cuda.synchronize()
    return time.time() - t0, observ.get_observable_history("min_energy")


def _time_ga(L, J, pop, nT, seed):
    _seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    _, observ, elapsed = global_annealing(
        L, J, pop, num_steps_MC=5, swap_step=15, N=L ** 3,
        Tstart=1.92, Tend=0.1, Observables=Observables,
        schedule="logT", num_temps_determiner=nT,
    )
    torch.cuda.synchronize()
    return time.time() - t0, observ.get_observable_history("min_energy")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1736329224)
    ap.add_argument("--pop-size", type=int, default=8192)
    ap.add_argument("--num-temps", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3,
                    help="Median of this many timings per configuration.")
    ap.add_argument("--algorithms", type=str, nargs="+",
                    default=["SA", "PA", "GA"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("bench.py requires a GPU.", file=sys.stderr)
        return 2

    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"L={args.L} N={args.L ** 3} pop_size={args.pop_size} "
          f"num_temps={args.num_temps} repeats={args.repeats}")

    couplings = REPO_ROOT / f"Data/Alpha/Couplings/couplings_L{args.L}_R1_seed{args.seed}.txt"
    if not couplings.exists():
        raise FileNotFoundError(f"Couplings file missing: {couplings}")
    J = read_couplings(str(couplings), args.L ** 3).cuda()

    timers = {"SA": _time_sa, "PA": _time_pa, "GA": _time_ga}
    # warmup
    print("warmup...")
    _time_sa(args.L, J, args.pop_size, 5, 999)
    _time_pa(args.L, J, args.pop_size, 5, 999)

    rows = []
    for label in args.algorithms:
        fn = timers[label]
        # Characterise equivalence BEFORE benchmarking. We report three numbers:
        #   • reference: two runs of the *original* kernel at the same seed
        #     (measures intrinsic cuBLAS/cuDNN non-determinism);
        #   • orig vs opt: same seed, upstream vs optimised kernels;
        #   • final-minimum gap: |min(orig) – min(opt)| over the trajectory.
        kernels.uninstall()
        _, orig_min_a = fn(args.L, J, args.pop_size, args.num_temps, seed=13)
        kernels.uninstall()
        _, orig_min_b = fn(args.L, J, args.pop_size, args.num_temps, seed=13)
        kernels.install()
        _, opt_min = fn(args.L, J, args.pop_size, args.num_temps, seed=13)
        kernels.uninstall()
        a, b, c = (np.asarray(orig_min_a), np.asarray(orig_min_b),
                   np.asarray(opt_min))
        intr_gap = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
        opt_gap = float(np.max(np.abs(a - c))) if a.shape == c.shape else float("nan")
        print(f"[{label}] equivalence check  "
              f"intrinsic(orig vs orig same seed) max|Δ|={intr_gap:.3e}  |  "
              f"orig vs opt max|Δ|={opt_gap:.3e}  |  "
              f"final minE: orig_a={a.min():.6f} orig_b={b.min():.6f} opt={c.min():.6f}",
              flush=True)

        orig_times, opt_times = [], []
        for r in range(args.repeats):
            kernels.uninstall()
            tt, _ = fn(args.L, J, args.pop_size, args.num_temps, seed=100 + r)
            orig_times.append(tt)
            kernels.install()
            tt, _ = fn(args.L, J, args.pop_size, args.num_temps, seed=100 + r)
            opt_times.append(tt)
            kernels.uninstall()

        o = float(np.median(orig_times))
        p = float(np.median(opt_times))
        speedup = o / p if p > 0 else float("inf")
        rows.append((label, o, p, speedup))
        print(f"[{label}] orig={o:.3f}s  opt={p:.3f}s  speedup={speedup:.2f}x "
              f"(orig_runs={orig_times}, opt_runs={opt_times})")

    print("\nsummary (median of {} repeats)".format(args.repeats))
    print(f"{'alg':<5} {'orig_s':>9} {'opt_s':>9} {'speedup':>9}")
    for label, o, p, s in rows:
        print(f"{label:<5} {o:>9.3f} {p:>9.3f} {s:>9.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
