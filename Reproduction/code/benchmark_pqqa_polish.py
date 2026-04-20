"""PQQA + GPU Monte-Carlo polish runner.

Pipeline per run (all stages on a single B200 GPU, on the same K=8192
parallel replicas):

    1. PQQA on the full sol_size population
    2. Replace ``init_random_frac`` of the chains with random ±1 (diversity)
    3. Long parallel cooling anneal (checkerboard MC, geometric T_high->T_low)
    4. 1-flip + 2-flip greedy descent on every replica
    5. Kicked anneal: ``kick_cycles`` cycles of (cooled-MC kick + greedy)
    6. Short ILS safety net (k-flip perturb + greedy)
    7. Report the minimum (intensive) energy across all replicas

CSV schema matches benchmark_3d_ea.py so ``plot_success_vs_time.py``
can pick the rows up unmodified. Used by qqa_winner_run.sbatch to
reproduce the headline PQQA result against GA on the hard L=10
instance (seed 310411727, MEC = -1.6930031776).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Reproduction" / "third_party"))
sys.path.insert(0, str(REPO_ROOT / "Reproduction" / "code"))
sys.path.insert(0, str(REPO_ROOT / "Code" / "Legacy" / "packages"))
sys.path.insert(0, str(REPO_ROOT / "Code" / "Modern" / "optimization"))

import qqa  # noqa: E402  vendored
from qqa.callbacks import Callback, CallbackState  # noqa: E402

from monte_carlo import read_couplings  # noqa: E402
from benchmark_3d_ea import (  # noqa: E402
    _bipartite_coloring,
    _build_bond_tensors,
    _batched_single_flip,
    _batched_pair_flip,
    _batched_mc_polish,
    _batched_kicked_anneal,
    _batched_ils,
)


def _load_couplings_numpy(path: pathlib.Path, N: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    rows = data[:, 0].astype(np.int64)
    cols = data[:, 1].astype(np.int64)
    vals = data[:, 2].astype(np.float64)
    J = np.zeros((N, N), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        J[r, c] = v
        J[c, r] = v
    return J


def _intensive_energy(s: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    return -0.5 * (s @ J * s).sum(dim=-1) / s.shape[1]


class _PopulationSnapshot(Callback):
    def __init__(self) -> None:
        self.x_disc: torch.Tensor | None = None

    def on_train_end(self, state: CallbackState) -> None:
        with torch.no_grad():
            self.x_disc = state.relaxation.project(state.x).detach()


def run_single(
    *,
    seed: int,
    run_idx: int,
    sol_size: int,
    num_epochs: int,
    lr: float,
    pqqa_temp: float,
    min_bg: float,
    max_bg: float,
    curve_rate: int,
    div_param: float,
    cool_sweeps: int,
    cool_t_high: float,
    cool_t_low: float,
    init_random_frac: float,
    kick_cycles: int,
    kick_sweeps: int,
    kick_t_high: float,
    kick_t_low: float,
    ils_iters: int,
    ils_k: int,
    descent_sweeps: int,
    mc_matmul_dtype: torch.dtype | None,
    J_cuda: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    J_bonds: torch.Tensor,
    color_idx: tuple[torch.Tensor, torch.Tensor],
    problem,
    verbose: bool,
) -> dict:
    qqa.fix_seed(seed + run_idx)
    snap = _PopulationSnapshot()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    schedule = qqa.LinearBGSchedule(min_bg=min_bg, max_bg=max_bg)
    qqa.anneal(
        problem,
        sol_size=sol_size,
        learning_rate=lr,
        temp=pqqa_temp,
        schedule=schedule,
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        device="cuda",
        callbacks=[snap],
        record_history=False,
        verbose=False,
    )
    if snap.x_disc is None:
        raise RuntimeError("PQQA snapshot failed to capture final population.")
    S = snap.x_disc.to(dtype=torch.float32, device="cuda").contiguous()

    # ---- Stage 1: long parallel cooling anneal on K chains ----
    # init_random_frac > 0 replaces that fraction of the PQQA-trained
    # population with random ±1 chains (high-T diversity that the
    # gradient-based PQQA cannot inject).
    if init_random_frac > 0.0:
        K, N = S.shape
        n_rand = int(round(K * init_random_frac))
        if n_rand > 0:
            g = torch.Generator(device=S.device).manual_seed(seed + run_idx + 91)
            rand_chains = (torch.randint(
                0, 2, (n_rand, N), device=S.device, generator=g,
            ).to(S.dtype) * 2 - 1)
            S[-n_rand:] = rand_chains
    if cool_sweeps > 0:
        _batched_mc_polish(
            S, J_cuda.float(), color_idx,
            n_sweeps=cool_sweeps,
            temperature=cool_t_high, temp_end=cool_t_low,
            seed=seed + run_idx + 13,
            matmul_dtype=mc_matmul_dtype,
        )

    # ---- Stage 2: greedy 1-flip + 2-flip descent ----
    _batched_single_flip(S, J_cuda.float(), max_sweeps=descent_sweeps)
    _batched_pair_flip(
        S, J_cuda.float(), rows, cols, J_bonds,
        max_sweeps=descent_sweeps, inner_sweeps=descent_sweeps,
    )

    # ---- Stage 3: parallel basin-hopping with cooled-MC kicks ----
    if kick_cycles > 0:
        _batched_kicked_anneal(
            S, J_cuda.float(), rows, cols, J_bonds, color_idx,
            n_cycles=kick_cycles, kick_temp_high=kick_t_high,
            kick_temp_low=kick_t_low, kick_sweeps=kick_sweeps,
            descent_sweeps=descent_sweeps,
            seed=seed + run_idx + 17,
            matmul_dtype=mc_matmul_dtype,
        )

    # ---- Stage 4: short ILS safety net ----
    if ils_iters > 0:
        _batched_ils(
            S, J_cuda.float(), rows, cols, J_bonds,
            n_iter=ils_iters, k_perturb=ils_k,
            descent_sweeps=descent_sweeps,
            seed=seed + run_idx + 31415,
        )

    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    energies = _intensive_energy(S, J_cuda.float())
    minE = float(energies.min().item())
    meanE = float(energies.mean().item())

    if verbose:
        print(f"  run={run_idx}  final={minE:.6f}  total={t_total:.2f}s", flush=True)
    return {
        "run": run_idx,
        "min_energy": minE,
        "mean_energy": meanE,
        "runtime_s": t_total,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coupling-path", required=True, type=pathlib.Path)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--out-csv", required=True, type=pathlib.Path)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=310411727)
    ap.add_argument("--algorithm-label", default="QQA")
    ap.add_argument("--schedule-label", default="pqqa+polish")
    # PQQA
    ap.add_argument("--sol-size", type=int, default=8192)
    ap.add_argument("--num-epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--pqqa-temp", type=float, default=1e-3)
    ap.add_argument("--min-bg", type=float, default=-3.0)
    ap.add_argument("--max-bg", type=float, default=0.1)
    ap.add_argument("--curve-rate", type=int, default=6)
    ap.add_argument("--div-param", type=float, default=0.2)
    # Greedy descent
    ap.add_argument("--descent-sweeps", type=int, default=200)
    # Long parallel cooling anneal
    ap.add_argument("--cool-sweeps", type=int, default=0,
                    help="MC sweeps for the cooling anneal. 0 = off.")
    ap.add_argument("--cool-t-high", type=float, default=2.0)
    ap.add_argument("--cool-t-low", type=float, default=0.02)
    ap.add_argument("--init-random-frac", type=float, default=0.0,
                    help="Replace this fraction of the PQQA population with "
                         "random plus/minus 1 chains before cooling.")
    # Kicked anneal
    ap.add_argument("--kick-cycles", type=int, default=20)
    ap.add_argument("--kick-sweeps", type=int, default=50)
    ap.add_argument("--kick-t-high", type=float, default=1.0)
    ap.add_argument("--kick-t-low", type=float, default=0.05)
    # ILS tail
    ap.add_argument("--ils-iters", type=int, default=50)
    ap.add_argument("--ils-k", type=int, default=5)
    # MC matmul dtype: bf16 ~ 2-3x faster cool/kick on B200.
    ap.add_argument("--mc-matmul-dtype",
                    choices=("fp32", "bf16", "fp16"), default="fp32")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    torch.cuda.set_device(0)

    N = args.L ** 3
    print(f"[pqqapol] loading {args.coupling_path} (N={N})", flush=True)
    J_np = _load_couplings_numpy(args.coupling_path, N)
    J_cuda = read_couplings(str(args.coupling_path), N).cuda()
    problem = qqa.EdwardsAnderson.from_couplings_txt(
        str(args.coupling_path), N=N, device="cuda")

    rows, cols, J_bonds = _build_bond_tensors(J_np, torch.device("cuda"))
    color = _bipartite_coloring(J_np)
    color_idx = (
        torch.as_tensor(np.flatnonzero(color == 0), dtype=torch.long, device="cuda"),
        torch.as_tensor(np.flatnonzero(color == 1), dtype=torch.long, device="cuda"),
    )

    mc_dtype_map = {
        "fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16,
    }
    mc_matmul_dtype = mc_dtype_map[args.mc_matmul_dtype]

    print(
        f"[pqqapol] sol_size={args.sol_size}  PQQA(nE={args.num_epochs}, "
        f"curve={args.curve_rate}, div={args.div_param})  "
        f"kick(c={args.kick_cycles}, sw={args.kick_sweeps}, "
        f"T={args.kick_t_high}->{args.kick_t_low})  "
        f"ILS(n={args.ils_iters}, k={args.ils_k})  "
        f"mc_mm={args.mc_matmul_dtype}  runs={args.runs}",
        flush=True,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "algorithm", "num_temps", "schedule", "run",
            "min_energy", "mean_energy", "runtime_s",
        ])
        tic = time.perf_counter()
        for r in range(args.runs):
            print(f"[{r+1:>2d}/{args.runs}] {args.algorithm_label} ...", flush=True)
            row = run_single(
                seed=args.seed,
                run_idx=r,
                sol_size=args.sol_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                pqqa_temp=args.pqqa_temp,
                min_bg=args.min_bg,
                max_bg=args.max_bg,
                curve_rate=args.curve_rate,
                div_param=args.div_param,
                cool_sweeps=args.cool_sweeps,
                cool_t_high=args.cool_t_high,
                cool_t_low=args.cool_t_low,
                init_random_frac=args.init_random_frac,
                kick_cycles=args.kick_cycles,
                kick_sweeps=args.kick_sweeps,
                kick_t_high=args.kick_t_high,
                kick_t_low=args.kick_t_low,
                ils_iters=args.ils_iters,
                ils_k=args.ils_k,
                descent_sweeps=args.descent_sweeps,
                mc_matmul_dtype=mc_matmul_dtype,
                J_cuda=J_cuda,
                rows=rows,
                cols=cols,
                J_bonds=J_bonds,
                color_idx=color_idx,
                problem=problem,
                verbose=args.verbose,
            )
            writer.writerow([
                args.algorithm_label, args.kick_cycles, args.schedule_label,
                row["run"], f"{row['min_energy']:.10f}",
                f"{row['mean_energy']:.10f}", f"{row['runtime_s']:.6f}",
            ])
            fh.flush()
        elapsed = (time.perf_counter() - tic) / 60.0
    print(f"[pqqapol] wrote {args.out_csv} (total {elapsed:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
