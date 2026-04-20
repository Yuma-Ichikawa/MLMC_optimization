r"""GPU Monte-Carlo polish kernels used by the PQQA pipeline.

All kernels operate on a batch of ``K`` parallel ±1 spin configurations
``S \in {-1,+1}^{K x N}`` and a dense symmetric coupling matrix
``J \in R^{N x N}``. The 3-D Edwards-Anderson cubic lattice is bipartite,
so a single sweep of checkerboard Metropolis costs only two GPU GEMMs.

The kernels are imported by

* ``Reproduction/code/benchmark_pqqa_polish.py`` (the winner runner), and
* ``Reproduction/code/test_mc_polish_correctness.py`` (CPU bit-equivalence).

Nothing here knows about PQQA itself; the file is a self-contained
PyTorch helper library.
"""

from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------


def _bipartite_coloring(J_np: np.ndarray) -> np.ndarray:
    """Two-colour the spin lattice via BFS.

    Returns an ``(N,)`` int array of 0/1 colours; raises if the coupling
    graph is not bipartite. Within each colour every spin's neighbours
    sit in the other colour, so we can flip every same-colour spin in
    parallel without breaking detailed balance (each flip only sees
    frozen neighbours).
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


def _build_bond_tensors(J_np: np.ndarray, device: torch.device):
    """Pre-compute the (rows, cols, J_bonds) descriptors used by pair-flip."""
    rows_np, cols_np = np.nonzero(np.triu(J_np, k=1))
    rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
    cols = torch.as_tensor(cols_np, dtype=torch.long, device=device)
    J_bonds = torch.as_tensor(
        J_np[rows_np, cols_np], dtype=torch.float32, device=device,
    )
    return rows, cols, J_bonds


# ---------------------------------------------------------------------------
# Greedy descent (1-flip and 2-flip)
# ---------------------------------------------------------------------------


def _batched_single_flip(
    S: torch.Tensor,
    J: torch.Tensor,
    *,
    max_sweeps: int,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Descend each replica to a 1-flip local minimum (in-place)."""
    for _ in range(max_sweeps):
        H = S @ J
        dE = 2.0 * S * H
        best_dE, best_k = dE.min(dim=1)
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
    """Alternate best-pair flips with 1-flip descents until no pair improves."""
    for _ in range(max_sweeps):
        H = S @ J
        dE = 2.0 * S * H
        dEi = dE[:, rows]
        dEj = dE[:, cols]
        pair_coupling = 4.0 * J_bonds[None, :] * S[:, rows] * S[:, cols]
        dE_pairs = dEi + dEj - pair_coupling
        best_dE, best_bond = dE_pairs.min(dim=1)
        mask = best_dE < -tol
        if not bool(mask.any()):
            break
        idx = torch.where(mask)[0]
        bi = rows[best_bond[idx]]
        bj = cols[best_bond[idx]]
        S[idx, bi] = -S[idx, bi]
        S[idx, bj] = -S[idx, bj]
        _batched_single_flip(S, J, max_sweeps=inner_sweeps, tol=tol)
    return S


# ---------------------------------------------------------------------------
# Checkerboard Metropolis polish
# ---------------------------------------------------------------------------


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
    """Checkerboard Metropolis with optional geometric cooling.

    For each sweep we (a) update every spin in colour 0 using a single
    batched matmul that produces the local field at colour-0 spins only,
    then (b) update colour 1 with the freshly updated colour-0 spins.
    Two GPU matmuls per sweep, each with ``O(K * N * |sub|) ~ N^2/2``
    FMAs — half the cost of computing the full local field.

    Parameters
    ----------
    S : (K, N) ±1 float tensor (mutated in place).
    J : (N, N) symmetric float coupling matrix.
    color_idx : pair of LongTensors listing the colour-0 / colour-1 indices.
    n_sweeps : number of full sweeps to perform.
    temperature : initial temperature ``T_start``.
    temp_end : optional final temperature ``T_end`` (geometric schedule).
    matmul_dtype : if set (e.g. ``torch.bfloat16``) the per-sub matmul is
        computed in that dtype and cast back to fp32 for the Metropolis
        test. On B200, bf16 GEMMs run ~3-4x faster than fp32; the
        induced ~1e-3 relative error in dE is much smaller than the
        Metropolis acceptance noise.
    """
    if n_sweeps <= 0 or temperature <= 0.0:
        return S
    g = torch.Generator(device=S.device).manual_seed(int(seed))
    T_start = float(temperature)
    T_end = float(temp_end) if (temp_end is not None and temp_end > 0.0) else T_start
    idx_a, idx_b = color_idx
    log_ratio = math.log(T_end / T_start) if T_start != T_end else 0.0

    J_sub = (
        J.index_select(1, idx_a).contiguous(),
        J.index_select(1, idx_b).contiguous(),
    )
    if matmul_dtype is not None and matmul_dtype != J.dtype and S.is_cuda:
        J_sub = (J_sub[0].to(matmul_dtype), J_sub[1].to(matmul_dtype))
        S_buf: torch.Tensor | None = torch.empty_like(S, dtype=matmul_dtype)
        cast_back = True
    else:
        S_buf = None
        cast_back = False

    sub_pairs = ((idx_a, J_sub[0]), (idx_b, J_sub[1]))
    for s in range(n_sweeps):
        T = T_start * math.exp(log_ratio * s / max(n_sweeps - 1, 1))
        neg_beta = -1.0 / T
        for sub, J_s in sub_pairs:
            if cast_back:
                S_buf.copy_(S)
                H_sub = (S_buf @ J_s).to(S.dtype)
            else:
                H_sub = S @ J_s
            S_sub = S.index_select(1, sub)
            dE = 2.0 * S_sub * H_sub
            log_rand = torch.empty_like(dE).uniform_(generator=g).log_()
            sign = torch.where(
                log_rand < neg_beta * dE, -S.new_ones(()), S.new_ones(()),
            )
            S.index_copy_(1, sub, S_sub * sign)
    return S


# ---------------------------------------------------------------------------
# Higher-level escape moves
# ---------------------------------------------------------------------------


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
    """Parallel basin-hopping with cooled-MC kicks.

    Per cycle we (a) heat the per-replica best by ``kick_sweeps`` checkerboard
    Metropolis sweeps cooled from ``kick_temp_high`` to ``kick_temp_low``, then
    (b) re-descend with 1-flip + 2-flip greedy. Each replica keeps whichever
    of (kicked-then-descended) and (previous best) has lower energy.
    """
    if n_cycles <= 0 or kick_sweeps <= 0:
        return S
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
    """Iterated Local Search safety net (k-flip perturb + greedy descent)."""
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


__all__ = [
    "_bipartite_coloring",
    "_build_bond_tensors",
    "_batched_single_flip",
    "_batched_pair_flip",
    "_batched_mc_polish",
    "_batched_kicked_anneal",
    "_batched_ils",
]
