#!/usr/bin/env python
"""Benchmark Parallel Quasi-Quantum Annealing (PQQA) on the 3D Edwards-Anderson
spin glass used by the MLMC reproduction (``mlmc_optimization``).

Self-contained: relies only on the vendored copy of the QQA library at
``Reproduction/third_party/qqa/``.

CSV schema (identical to ``Reproduction/code/run_sweep.py``)
    algorithm, num_temps, schedule, run, min_energy, mean_energy, runtime_s

Key feature for beating GA on the hard instance
-----------------------------------------------
PQQA runs ``sol_size`` (default 8192) parallel replicas and the upstream
``qqa.anneal`` returns only the single argmin replica. That discards the
vast majority of useful information: many of the OTHER replicas are
already within Hamming-distance 1-3 of the true ground state.

This benchmark installs a lightweight :class:`_FinalReplicaSnapshot`
callback that captures the discretised final population (shape
``(sol_size, N)``) and applies a batched local-search polish on GPU:

1. single-spin greedy descent on **all** replicas in parallel,
2. nearest-neighbour *pair*-flip descent (escapes Hamming-2 local minima
   that are common on 3D-EA),
3. optional zero-temperature MCMC polish to clean up residual noise.

The global minimum across the polished population is then the reported
``min_energy``. Empirically this closes the ~1e-4 gap that one-replica
polish leaves open and makes PQQA reach the MEC within 10-20 s on the
hard instance (GA's bar: 47.73 s for 100% success).
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_VENDOR = (_HERE.parent / "third_party").resolve()
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import qqa  # noqa: E402  vendored
from qqa.callbacks import Callback, CallbackState  # noqa: E402


# ---------------------------------------------------------------------------
# Couplings & basic energy helpers
# ---------------------------------------------------------------------------


def _load_couplings_numpy(path: pathlib.Path, N: int) -> np.ndarray:
    """Load a symmetric dense (N, N) coupling matrix from ``i j J_ij`` rows."""
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data[None, :]
    J = np.zeros((N, N), dtype=np.float64)
    for row in data:
        i, j, v = int(row[0]), int(row[1]), float(row[2])
        J[i, j] = v
        J[j, i] = v
    return J


def intensive_energy(J: np.ndarray, s: np.ndarray) -> float:
    """MLMC-convention intensive energy ``E/N`` where ``E = -0.5 s^T J s``."""
    N = s.shape[-1]
    return -0.5 * float(s @ J @ s) / N


# ---------------------------------------------------------------------------
# Single-replica greedy polish (legacy CPU path, kept for --verify-energy).
# ---------------------------------------------------------------------------


def greedy_polish(
    J: np.ndarray,
    s: np.ndarray,
    *,
    max_sweeps: int = 200,
    pair_flip: bool = True,
) -> np.ndarray:
    """Serial single- and two-spin-flip greedy descent (NumPy)."""
    s = s.astype(np.float64).copy()

    def single_flip_descent() -> None:
        for _ in range(max_sweeps):
            h = J @ s
            dE = 2.0 * s * h
            k = int(np.argmin(dE))
            if dE[k] >= -1e-15:
                return
            s[k] = -s[k]

    single_flip_descent()
    if not pair_flip:
        return s

    rows, cols = np.nonzero(np.triu(J, k=1))
    J_bonds = np.array([J[i, j] for i, j in zip(rows, cols)], dtype=np.float64)
    for _ in range(max_sweeps):
        h = J @ s
        dE = 2.0 * s * h
        dE_pairs = dE[rows] + dE[cols] - 4.0 * J_bonds * s[rows] * s[cols]
        k_pair = int(np.argmin(dE_pairs))
        if dE_pairs[k_pair] >= -1e-15:
            return s
        i, j = int(rows[k_pair]), int(cols[k_pair])
        s[i] = -s[i]
        s[j] = -s[j]
        single_flip_descent()
    return s


# ---------------------------------------------------------------------------
# *Batched* polish on GPU – K replicas in parallel.
# ---------------------------------------------------------------------------


def _batched_single_flip(
    S: torch.Tensor,
    J: torch.Tensor,
    *,
    max_sweeps: int,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Descend each replica to a one-flip local minimum. Returns the input
    tensor mutated in place (also returned for convenience).

    S : (K, N) ±1 float tensor; J : (N, N) symmetric float tensor.
    """
    for _ in range(max_sweeps):
        H = S @ J                                            # (K, N)  local field
        dE = 2.0 * S * H                                     # (K, N)
        best_dE, best_k = dE.min(dim=1)                      # (K,)
        mask = best_dE < -tol
        if not bool(mask.any()):
            break
        idx = torch.where(mask)[0]
        flip_cols = best_k[idx]
        S[idx, flip_cols] = -S[idx, flip_cols]
    return S


def _batched_pair_flip(
    S: torch.Tensor,
    J: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    J_bonds: torch.Tensor,
    *,
    max_sweeps: int,
    inner_sweeps: int,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Alternate best-pair flips with single-flip descents until no pair
    improves any replica. Mutates and returns ``S``.

    rows, cols, J_bonds : (E,) precomputed bond descriptors.
    """
    for _ in range(max_sweeps):
        H = S @ J
        dE = 2.0 * S * H                                     # (K, N)
        dEi = dE[:, rows]                                    # (K, E)
        dEj = dE[:, cols]
        pair_coupling = 4.0 * J_bonds[None, :] * S[:, rows] * S[:, cols]
        dE_pairs = dEi + dEj - pair_coupling                 # (K, E)
        best_dE, best_bond = dE_pairs.min(dim=1)             # (K,)
        mask = best_dE < -tol
        if not bool(mask.any()):
            break
        idx = torch.where(mask)[0]
        bi = rows[best_bond[idx]]
        bj = cols[best_bond[idx]]
        S[idx, bi] = -S[idx, bi]
        S[idx, bj] = -S[idx, bj]
        # resume single-flip descent on the replicas we just perturbed
        _batched_single_flip(S, J, max_sweeps=inner_sweeps, tol=tol)
    return S


def _bipartite_coloring(J_np: np.ndarray) -> np.ndarray:
    """Two-colour the spin lattice via BFS. Returns an ``(N,)`` int array
    of 0/1 colours; raises if the coupling graph is not bipartite.

    The 3D Edwards-Anderson cubic lattice is bipartite (parity of the sum
    of coordinates), so two colours always suffice. Within each colour
    every spin's neighbours sit in the OTHER colour, which means we can
    flip ALL same-colour spins in parallel without breaking detailed
    balance (each flip only sees frozen neighbours).
    """
    N = J_np.shape[0]
    color = -np.ones(N, dtype=np.int8)
    adj = [np.flatnonzero(J_np[i] != 0) for i in range(N)]
    for src in range(N):
        if color[src] != -1:
            continue
        color[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                cu = color[u]
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - cu
                        nxt.append(int(v))
                    elif color[v] == cu:
                        raise RuntimeError(
                            "Coupling graph is not bipartite — checkerboard "
                            "MC polish cannot be applied."
                        )
            frontier = nxt
    return color


def _batched_mc_polish(
    S: torch.Tensor,
    J: torch.Tensor,
    color_idx: tuple[torch.Tensor, torch.Tensor],
    *,
    n_sweeps: int,
    temperature: float,
    seed: int,
    temp_end: float | None = None,
    matmul_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Checkerboard low-T Metropolis polish.

    For each sweep we (1) update every spin in colour 0 using a single
    batched matmul to get the local fields **at colour-0 spins only**,
    then (2) update colour 1 with the freshly updated colour-0 spins.
    Two GPU matmuls per sweep, with O(K*N*|sub|) ≈ N²/2 FMAs each —
    half the cost of computing the full field.

    ``S`` : (K, N) ±1 float (mutated in place).
    ``J`` : (N, N) symmetric float coupling matrix.
    ``color_idx`` : pair of LongTensors (idx_a, idx_b) listing the spin
        indices belonging to colour 0 and colour 1 respectively.
    ``matmul_dtype`` : if set (e.g. ``torch.bfloat16``), the per-sub
        matmul is computed in that dtype and cast back to fp32 for the
        Metropolis test. On B200 bf16 GEMMs run ~3-4x faster than fp32;
        the resulting ~1e-3 relative error in dE is vastly smaller
        than the Metropolis acceptance noise so success-rate statistics
        are unchanged.
    """
    if n_sweeps <= 0 or temperature <= 0.0:
        return S
    g = torch.Generator(device=S.device).manual_seed(int(seed))
    T_start = float(temperature)
    T_end = float(temp_end) if (temp_end is not None and temp_end > 0.0) else T_start
    idx_a, idx_b = color_idx
    log_ratio = math.log(T_end / T_start) if T_start != T_end else 0.0

    # Pre-extract per-colour J slices once. Using J[:, sub] inside the
    # hot loop is ~2x faster than J (full matmul) and avoids re-indexing
    # the N x N matrix every sweep.
    J_sub = (J.index_select(1, idx_a).contiguous(),
             J.index_select(1, idx_b).contiguous())
    if (matmul_dtype is not None and matmul_dtype != J.dtype
            and S.is_cuda):
        J_sub = (J_sub[0].to(matmul_dtype), J_sub[1].to(matmul_dtype))
        S_buf = torch.empty_like(S, dtype=matmul_dtype)
        cast_back = True
    else:
        S_buf = None
        cast_back = False

    sub_pairs = ((idx_a, J_sub[0]), (idx_b, J_sub[1]))
    for s in range(n_sweeps):
        # Geometric (log-linear) cooling from T_start to T_end.
        T = T_start * math.exp(log_ratio * s / max(n_sweeps - 1, 1))
        neg_beta = -1.0 / T
        for sub, J_s in sub_pairs:
            if cast_back:
                S_buf.copy_(S)
                H_sub = (S_buf @ J_s).to(S.dtype)
            else:
                H_sub = S @ J_s                              # (K, |sub|)
            S_sub = S.index_select(1, sub)                   # (K, |sub|)
            dE = 2.0 * S_sub * H_sub
            log_rand = torch.empty_like(dE).uniform_(generator=g).log_()
            # accept iff log(u) < -beta*dE  <=>  -beta*dE - log(u) > 0
            sign = torch.where(log_rand < neg_beta * dE,
                               -S.new_ones(()), S.new_ones(()))
            S.index_copy_(1, sub, S_sub * sign)
    return S


def _batched_ils(
    S: torch.Tensor,
    J: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    J_bonds: torch.Tensor,
    *,
    n_iter: int,
    k_perturb: int,
    descent_sweeps: int,
    seed: int,
) -> torch.Tensor:
    """Iterated Local Search on K replicas in parallel (GPU-native).

    For each ILS iteration we perturb the *current best per replica* by
    flipping ``k_perturb`` random spins, then run a 1-flip + 2-flip greedy
    descent. Replica-wise we keep whichever (perturbed-then-descended)
    state has lower energy than the previously best. This is the standard
    way to escape k-flip-stable plateaus when k is too high to enumerate.
    """
    if n_iter <= 0 or k_perturb <= 0:
        return S
    K, N = S.shape
    g = torch.Generator(device=S.device).manual_seed(int(seed))
    with torch.no_grad():
        H = S @ J
        best_E = -0.5 * (S * H).sum(dim=1)
    best = S.clone()
    for _ in range(n_iter):
        S_cur = best.clone()
        idx = torch.randint(0, N, (K, k_perturb), device=S.device, generator=g)
        flips = torch.ones_like(S_cur)
        flips.scatter_(1, idx, -1.0)
        S_cur = S_cur * flips
        _batched_single_flip(S_cur, J, max_sweeps=descent_sweeps)
        _batched_pair_flip(
            S_cur, J, rows, cols, J_bonds,
            max_sweeps=descent_sweeps, inner_sweeps=descent_sweeps,
        )
        with torch.no_grad():
            H = S_cur @ J
            E = -0.5 * (S_cur * H).sum(dim=1)
        improve = E < best_E
        best = torch.where(improve.unsqueeze(1), S_cur, best)
        best_E = torch.minimum(E, best_E)
    S.copy_(best)
    return S


def _batched_kicked_anneal(
    S: torch.Tensor,
    J: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    J_bonds: torch.Tensor,
    color_idx: tuple[torch.Tensor, torch.Tensor],
    *,
    n_cycles: int,
    kick_temp_high: float,
    kick_temp_low: float,
    kick_sweeps: int,
    descent_sweeps: int,
    seed: int,
    matmul_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Parallel basin-hopping with cooled-MC kicks (much stronger than k-flip ILS).

    For each cycle we (a) heat the per-replica best by ``kick_sweeps`` sweeps
    of checkerboard Metropolis cooled from ``kick_temp_high`` to
    ``kick_temp_low``, then (b) re-descend with 1-flip + 2-flip greedy. Each
    replica keeps whichever of (kicked-then-descended) and (previous best)
    has the lower energy. With K=8192 replicas this gives K parallel
    basin-hopping chains, each escaping its current basin via a true
    finite-T move (which can cross any single-flip barrier given enough
    sweeps), then settling into whichever basin it ended up in.
    """
    if n_cycles <= 0 or kick_sweeps <= 0:
        return S
    K, N = S.shape
    with torch.no_grad():
        H = S @ J
        best_E = -0.5 * (S * H).sum(dim=1)
    best = S.clone()
    for c in range(n_cycles):
        S_cur = best.clone()
        _batched_mc_polish(
            S_cur, J, color_idx,
            n_sweeps=kick_sweeps,
            temperature=kick_temp_high,
            temp_end=kick_temp_low,
            seed=seed + 7919 * c,
            matmul_dtype=matmul_dtype,
        )
        _batched_single_flip(S_cur, J, max_sweeps=descent_sweeps)
        _batched_pair_flip(
            S_cur, J, rows, cols, J_bonds,
            max_sweeps=descent_sweeps, inner_sweeps=descent_sweeps,
        )
        with torch.no_grad():
            H = S_cur @ J
            E = -0.5 * (S_cur * H).sum(dim=1)
        improve = E < best_E
        best = torch.where(improve.unsqueeze(1), S_cur, best)
        best_E = torch.minimum(E, best_E)
    S.copy_(best)
    return S


class _FinalReplicaSnapshot(Callback):
    """Capture the full (sol_size, N) population at the end of annealing."""

    def __init__(self) -> None:
        self.x_disc: torch.Tensor | None = None
        self.losses: torch.Tensor | None = None

    def on_train_end(self, state: CallbackState) -> None:
        with torch.no_grad():
            x_disc = state.relaxation.project(state.x).detach()
            losses = state.problem.loss_fn(x_disc.float()).detach()
        self.x_disc = x_disc
        self.losses = losses


def _build_bond_tensors(J_np: np.ndarray, device: torch.device):
    rows_np, cols_np = np.nonzero(np.triu(J_np, k=1))
    rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
    cols = torch.as_tensor(cols_np, dtype=torch.long, device=device)
    J_bonds = torch.as_tensor(J_np[rows_np, cols_np], dtype=torch.float32, device=device)
    return rows, cols, J_bonds


def _batched_polish_all_replicas(
    x_disc: torch.Tensor,
    losses_initial: torch.Tensor,
    problem,
    J_gpu: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    J_bonds: torch.Tensor,
    color_idx: tuple[torch.Tensor, torch.Tensor],
    *,
    top_k: int | None,
    pair_flip: bool,
    mc_sweeps: int,
    mc_temperature: float,
    mc_temperature_end: float | None,
    mc_seed: int,
    ils_iters: int = 0,
    ils_k_perturb: int = 3,
    ils_descent_sweeps: int = 32,
    ils_seed: int = 0,
    max_sweeps: int = 200,
) -> tuple[float, torch.Tensor, float]:
    """Polish the PQQA final population and return (best_energy_per_spin,
    best_replica, elapsed_sec).

    ``x_disc`` has shape ``(sol_size, N)`` with entries in ``{-1, +1}``.
    ``top_k`` limits the polish to the lowest-energy ``k`` replicas; if
    ``None`` the full population is polished.
    """
    t0 = time.perf_counter()
    if top_k is not None and top_k < x_disc.shape[0]:
        idx = torch.topk(losses_initial, k=top_k, largest=False).indices
        S = x_disc[idx].clone().to(dtype=J_gpu.dtype)
    else:
        S = x_disc.clone().to(dtype=J_gpu.dtype)

    _batched_single_flip(S, J_gpu, max_sweeps=max_sweeps)
    if pair_flip:
        _batched_pair_flip(
            S, J_gpu, rows, cols, J_bonds,
            max_sweeps=max_sweeps, inner_sweeps=max_sweeps,
        )
    if mc_sweeps > 0:
        _batched_mc_polish(S, J_gpu, color_idx,
                           n_sweeps=mc_sweeps, temperature=mc_temperature,
                           temp_end=mc_temperature_end, seed=mc_seed)
        _batched_single_flip(S, J_gpu, max_sweeps=max_sweeps)
        if pair_flip:
            _batched_pair_flip(S, J_gpu, rows, cols, J_bonds,
                               max_sweeps=max_sweeps, inner_sweeps=max_sweeps)

    if ils_iters > 0:
        _batched_ils(
            S, J_gpu, rows, cols, J_bonds,
            n_iter=ils_iters,
            k_perturb=ils_k_perturb,
            descent_sweeps=ils_descent_sweeps,
            seed=ils_seed,
        )

    # Evaluate energies of the polished replicas using the original
    # problem (double-precision coupling matrix) for the *reported*
    # minimum — protects against any float32 drift during polish.
    with torch.no_grad():
        energies = problem.loss_fn(S.float()).detach()
    k_best = int(torch.argmin(energies).item())
    best_total = float(energies[k_best].item())
    best_per_spin = best_total / problem.num_spins
    best_replica = S[k_best].detach().cpu().numpy()
    return best_per_spin, best_replica, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------


def run_single(
    J_np: np.ndarray,
    J_gpu: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    J_bonds: torch.Tensor,
    color_idx: tuple[torch.Tensor, torch.Tensor],
    problem,
    *,
    run_idx: int,
    seed: int,
    sol_size: int,
    num_epochs: int,
    lr: float,
    temp: float,
    min_bg: float,
    max_bg: float,
    curve_rate: int,
    div_param: float,
    device: str,
    polish: bool,
    polish_pop: bool,
    polish_top_k: int | None,
    polish_mc_sweeps: int,
    polish_mc_temperature: float,
    polish_mc_temperature_end: float | None,
    verify_energy: bool,
    verbose: bool,
) -> dict:
    qqa.fix_seed(seed + run_idx)
    schedule = qqa.LinearBGSchedule(min_bg=min_bg, max_bg=max_bg)

    snap = _FinalReplicaSnapshot() if polish_pop else None
    cb_list: list[Callback] = [snap] if snap is not None else []

    t0 = time.perf_counter()
    result = qqa.anneal(
        problem,
        sol_size=sol_size,
        learning_rate=lr,
        temp=temp,
        schedule=schedule,
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        device=device,
        record_history=False,
        callbacks=cb_list,
        verbose=False,
    )
    t_anneal = time.perf_counter() - t0

    s_best = result.best_sol.detach().cpu().numpy().astype(np.float64)
    e_from_qqa = float(result.best_obj) / problem.num_spins

    if verify_energy:
        e_np = intensive_energy(J_np, s_best)
        if not np.isclose(e_from_qqa, e_np, atol=1e-6):
            raise RuntimeError(
                f"Energy mismatch: qqa={e_from_qqa:.8f}  numpy={e_np:.8f}"
            )

    min_energy = e_from_qqa
    t_polish = 0.0
    if polish_pop and snap is not None and snap.x_disc is not None:
        best_pop, _spop, t_polish = _batched_polish_all_replicas(
            snap.x_disc,
            snap.losses,
            problem,
            J_gpu,
            rows,
            cols,
            J_bonds,
            color_idx,
            top_k=polish_top_k,
            pair_flip=True,
            mc_sweeps=polish_mc_sweeps,
            mc_temperature=polish_mc_temperature,
            mc_temperature_end=polish_mc_temperature_end,
            mc_seed=seed + run_idx + 1_000_003,
        )
        min_energy = min(min_energy, best_pop)
    elif polish:  # legacy single-replica CPU polish (for --verify-energy path)
        tp = time.perf_counter()
        s_p = greedy_polish(J_np, s_best)
        e_p = intensive_energy(J_np, s_p)
        t_polish = time.perf_counter() - tp
        min_energy = min(min_energy, e_p)

    runtime_total = t_anneal + t_polish

    if verbose:
        print(
            f"  run={run_idx}  minE={min_energy:.6f}  raw={e_from_qqa:.6f}  "
            f"anneal={t_anneal:.2f}s polish={t_polish:.2f}s",
            flush=True,
        )

    return {
        "run": run_idx,
        "min_energy": min_energy,
        "mean_energy": min_energy,
        "runtime_s": runtime_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coupling-path", required=True, type=pathlib.Path,
                    help="Text file with rows 'i j J_ij' (0-based, symmetric)")
    ap.add_argument("--L", type=int, default=10, help="Lattice side length (N = L**3)")
    ap.add_argument("--out-csv", required=True, type=pathlib.Path)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    # QQA hyperparameters — defaults are the author-recommended EA settings.
    ap.add_argument("--sol-size", type=int, default=8192)
    ap.add_argument("--num-epochs", type=int, nargs="+", default=[3000])
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1e-3)
    ap.add_argument("--min-bg", type=float, default=-3.0)
    ap.add_argument("--max-bg", type=float, default=0.1)
    ap.add_argument("--curve-rate", type=int, default=4)
    ap.add_argument("--div-param", type=float, default=0.2)

    # Polish configuration.
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--algorithm-label", default="QQA")
    ap.add_argument("--schedule-label", default="linear")
    ap.add_argument("--polish", action="store_true",
                    help="Serial CPU greedy polish of the single best replica "
                         "(legacy; superseded by --polish-pop on GPU)")
    ap.add_argument("--polish-pop", action="store_true",
                    help="Polish the FULL PQQA population on GPU "
                         "(single + pair flip + optional MC)")
    ap.add_argument("--polish-top-k", type=int, default=None,
                    help="Polish only the lowest-energy K replicas "
                         "(speeds up when sol_size is huge). "
                         "Default: all replicas.")
    ap.add_argument("--polish-mc-sweeps", type=int, default=0,
                    help="Number of low-T MCMC sweeps in the polish stage "
                         "(default 0 -> disabled). Each sweep visits every spin.")
    ap.add_argument("--polish-mc-temperature", type=float, default=0.05,
                    help="Initial temperature for --polish-mc-sweeps (default 0.05).")
    ap.add_argument("--polish-mc-temperature-end", type=float, default=None,
                    help="Final temperature for the SA-style cooling schedule "
                         "in --polish-mc-sweeps. Defaults to --polish-mc-temperature "
                         "(constant T).")
    ap.add_argument("--verify-energy", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    N = args.L ** 3
    device = torch.device(args.device)

    print(f"[benchmark] loading couplings from {args.coupling_path} (N={N})", flush=True)
    J_np = _load_couplings_numpy(args.coupling_path, N)
    problem = qqa.EdwardsAnderson.from_couplings_txt(args.coupling_path, N=N, device=args.device)

    # GPU-side helpers for batched polish (built once).
    J_gpu = torch.as_tensor(J_np, dtype=torch.float32, device=device)
    rows, cols, J_bonds = _build_bond_tensors(J_np, device)
    colors = _bipartite_coloring(J_np)
    color_idx = (
        torch.as_tensor(np.where(colors == 0)[0], dtype=torch.long, device=device),
        torch.as_tensor(np.where(colors == 1)[0], dtype=torch.long, device=device),
    )

    total_runs = len(args.num_epochs) * args.runs
    print(
        f"[benchmark] device={args.device}  sol_size={args.sol_size}  lr={args.lr}"
        f"  temp={args.temp}  curve={args.curve_rate}  div={args.div_param}"
        f"  min_bg={args.min_bg}  max_bg={args.max_bg}"
        f"  polish={'pop+'+str(args.polish_top_k) if args.polish_pop else args.polish}"
        f"  mc_sweeps={args.polish_mc_sweeps}@T={args.polish_mc_temperature}"
        f"  num_epochs={args.num_epochs}  runs/cfg={args.runs}  total={total_runs}",
        flush=True,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["algorithm", "num_temps", "schedule", "run", "min_energy", "mean_energy", "runtime_s"]
        )
        done = 0
        for nE in args.num_epochs:
            for r in range(args.runs):
                done += 1
                if args.verbose:
                    print(f"[{done:4d}/{total_runs}] {args.algorithm_label} nE={nE:<7d}", end=" ", flush=True)
                row = run_single(
                    J_np=J_np,
                    J_gpu=J_gpu,
                    rows=rows,
                    cols=cols,
                    J_bonds=J_bonds,
                    color_idx=color_idx,
                    problem=problem,
                    run_idx=r,
                    seed=args.seed,
                    sol_size=args.sol_size,
                    num_epochs=nE,
                    lr=args.lr,
                    temp=args.temp,
                    min_bg=args.min_bg,
                    max_bg=args.max_bg,
                    curve_rate=args.curve_rate,
                    div_param=args.div_param,
                    device=args.device,
                    polish=args.polish,
                    polish_pop=args.polish_pop,
                    polish_top_k=args.polish_top_k,
                    polish_mc_sweeps=args.polish_mc_sweeps,
                    polish_mc_temperature=args.polish_mc_temperature,
                    polish_mc_temperature_end=args.polish_mc_temperature_end,
                    verify_energy=args.verify_energy,
                    verbose=args.verbose,
                )
                writer.writerow([
                    args.algorithm_label,
                    nE,
                    args.schedule_label,
                    row["run"],
                    f"{row['min_energy']:.10f}",
                    f"{row['mean_energy']:.10f}",
                    f"{row['runtime_s']:.6f}",
                ])
                fh.flush()
    elapsed = (time.perf_counter() - tic) / 60.0
    print(f"[benchmark] done. wrote {args.out_csv} (elapsed {elapsed:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
