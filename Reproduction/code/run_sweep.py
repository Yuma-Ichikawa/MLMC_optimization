"""Small-scale reproduction of Fig. 2 of Del Bono et al. (arXiv:2510.19544).

Runs Simulated Annealing (SA), Population Annealing (PA), and Global Annealing
with 15 local MCS per global move (GA_15) on a single 3D Edwards-Anderson
instance, for several annealing lengths (``num_temps``) and several independent
runs per configuration. Emits a tidy CSV with columns

    algorithm, num_temps, schedule, run, min_energy, mean_energy, runtime_s

The schedule follows the paper's main comparison: logarithmic spacing in ``T``
between ``Tstart=1.92`` and ``Tend=0.1``. Hyperparameters follow the paper
(Methods Sec. IV), except the lattice size and population size, which are
reduced for tractability.

Run from the repository root::

    python Reproduction/code/run_sweep.py --pop-size 8192 --runs 10 --out Reproduction/fresh_runs/sweep_L6.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Code" / "Legacy" / "packages"))
sys.path.insert(0, str(REPO_ROOT / "Code" / "Modern" / "optimization"))

from monte_carlo import Observables, read_couplings  # noqa: E402

from simulated_annealing import simulated_annealing  # noqa: E402
from population_annealing import population_annealing  # noqa: E402
from global_annealing import global_annealing  # noqa: E402


def run_single(algorithm, L, J, pop_size, num_temps, schedule, Tstart, Tend,
               ga_local=15, pa_mcs=10, sa_mcs_per_T=15):
    """Run one annealing pass and return (min_energy_intensive, mean_energy_final, elapsed_s)."""
    N = L * L * L
    if algorithm == "SA":
        temperatures, observ, dt = simulated_annealing(
            L, J, pop_size, sa_mcs_per_T, N, Tstart, Tend, Observables,
            num_temps_determiner=num_temps, schedule=schedule,
        )
    elif algorithm == "PA":
        temperatures, observ, dt = population_annealing(
            L, J, pop_size, pa_mcs, Tstart, Tend, Observables,
            num_temps_determiner=num_temps, schedule=schedule,
        )
    elif algorithm == "GA":
        # Paper: 5 global moves per temperature, 15 local MCS per global move.
        temperatures, observ, dt = global_annealing(
            L, J, pop_size, num_steps_MC=5, swap_step=ga_local, N=N,
            Tstart=Tstart, Tend=Tend, Observables=Observables,
            num_temps_determiner=num_temps, schedule=schedule,
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    minE = float(torch.tensor(observ.get_observable_history("min_energy")).min())
    meanE = float(observ.get_observable_history("mean_energy")[-1])
    return minE, meanE, dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1736329224)
    parser.add_argument("--pop-size", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--num-temps", type=int, nargs="+",
                        default=[10, 20, 30, 50])
    parser.add_argument("--schedule", type=str, default="logT",
                        choices=["logT", "linearT", "linearBeta"])
    parser.add_argument("--Tstart", type=float, default=1.92)
    parser.add_argument("--Tend", type=float, default=0.1)
    parser.add_argument("--algorithms", type=str, nargs="+",
                        default=["SA", "PA", "GA"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", action="store_true",
                        help="Run one discarded SA pass to warm up the GPU JIT caches.")
    parser.add_argument("--optimized", action="store_true",
                        help="Install the Reproduction/speedups/ kernels before sweeping. "
                             "Equivalent output (verified by speedups/verify.py); faster.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no GPU visible to torch.")

    torch.cuda.set_device(0)
    print(f"[sweep] device={torch.cuda.get_device_name(0)}", flush=True)

    if args.optimized:
        sys.path.insert(0, str(REPO_ROOT / "Reproduction" / "speedups"))
        import kernels
        kernels.install()
        print("[sweep] optimised kernels installed.", flush=True)

    N = args.L ** 3
    coupling_path = REPO_ROOT / f"Data/Alpha/Couplings/couplings_L{args.L}_R1_seed{args.seed}.txt"
    if not coupling_path.exists():
        raise FileNotFoundError(f"Couplings not found: {coupling_path}")
    J = read_couplings(str(coupling_path), N).cuda()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] L={args.L} N={N} pop_size={args.pop_size} seed={args.seed}"
          f" runs={args.runs} num_temps={args.num_temps} schedule={args.schedule}",
          flush=True)

    if args.warmup:
        print("[sweep] warmup SA pass…", flush=True)
        run_single("SA", args.L, J, args.pop_size, num_temps=5,
                   schedule=args.schedule, Tstart=args.Tstart, Tend=args.Tend)

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "num_temps", "schedule", "run",
                         "min_energy", "mean_energy", "runtime_s"])
        total = len(args.algorithms) * len(args.num_temps) * args.runs
        k = 0
        t0 = time.time()
        for algorithm in args.algorithms:
            for num_temps in args.num_temps:
                for run in range(args.runs):
                    k += 1
                    torch.manual_seed(args.seed + 1000 * run
                                      + 7 * num_temps
                                      + {"SA": 1, "PA": 2, "GA": 3}[algorithm])
                    minE, meanE, dt = run_single(
                        algorithm, args.L, J, args.pop_size, num_temps,
                        args.schedule, args.Tstart, args.Tend,
                    )
                    writer.writerow([algorithm, num_temps, args.schedule, run,
                                     f"{minE:.10f}", f"{meanE:.10f}", f"{dt:.4f}"])
                    f.flush()
                    elapsed = time.time() - t0
                    eta = elapsed / k * (total - k)
                    print(f"[{k:>4d}/{total}] {algorithm} nT={num_temps:<3d} run={run:<2d}"
                          f" minE={minE:.6f} dt={dt:.3f}s (ETA {eta/60:.1f} min)",
                          flush=True)
    print(f"[sweep] done. wrote {args.out} (elapsed {(time.time()-t0)/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
