"""Offline (CPU) correctness check for the refactored _batched_mc_polish.

We can't easily run bf16 on CPU but we CAN verify that the new fp32 code
path (partial matmul + index_copy_) produces bit-identical state to the
old fp32 code path (full matmul + slice assignment).
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# import the new implementation
from polish_kernels import _batched_mc_polish, _bipartite_coloring


def _reference_mc_polish(S, J, color_idx, *, n_sweeps, temperature,
                         seed, temp_end=None):
    """Pre-refactor implementation, copied verbatim for the gold standard."""
    if n_sweeps <= 0 or temperature <= 0.0:
        return S
    g = torch.Generator(device=S.device).manual_seed(int(seed))
    T_start = float(temperature)
    T_end = float(temp_end) if (temp_end is not None and temp_end > 0.0) else T_start
    idx_a, idx_b = color_idx
    log_ratio = math.log(T_end / T_start) if T_start != T_end else 0.0
    for s in range(n_sweeps):
        T = T_start * math.exp(log_ratio * s / max(n_sweeps - 1, 1))
        beta = 1.0 / T
        for sub in (idx_a, idx_b):
            H = S @ J
            dE = 2.0 * S[:, sub] * H[:, sub]
            log_rand = torch.empty_like(dE).uniform_(generator=g).log()
            accept = log_rand < (-beta * dE)
            sign = torch.where(accept, -1.0, 1.0).to(S.dtype)
            S[:, sub] = S[:, sub] * sign
    return S


def main() -> None:
    rng = np.random.default_rng(0)
    L = 4   # tiny lattice -> N=64
    N = L ** 3
    # Build a 3D EA-like coupling: cubic lattice nearest neighbours, ±1.
    J = np.zeros((N, N), dtype=np.float32)
    def site(x, y, z):
        return ((x % L) * L + (y % L)) * L + (z % L)
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = site(x, y, z)
                for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    j = site(x + dx, y + dy, z + dz)
                    Jij = float(rng.choice([-1.0, 1.0]))
                    J[i, j] = Jij
                    J[j, i] = Jij
    Jt = torch.from_numpy(J)
    color = _bipartite_coloring(J)
    color_idx = (
        torch.from_numpy(np.flatnonzero(color == 0)).long(),
        torch.from_numpy(np.flatnonzero(color == 1)).long(),
    )
    K = 64
    S0 = (torch.randint(0, 2, (K, N)).float() * 2 - 1)

    out_old = _reference_mc_polish(
        S0.clone(), Jt, color_idx,
        n_sweeps=200, temperature=0.5, temp_end=0.05, seed=42)
    out_new = _batched_mc_polish(
        S0.clone(), Jt, color_idx,
        n_sweeps=200, temperature=0.5, temp_end=0.05, seed=42)

    same = bool(torch.equal(out_old, out_new))
    diff = (out_old != out_new).float().mean().item()
    print(f"old vs new state bit-equal: {same}")
    print(f"hamming fraction differing: {diff:.4f}")
    if not same:
        print("WARNING: state diverges; refactor changed semantics.")
        return

    # Energy distributions
    def energy(s):
        return -0.5 * (s @ Jt * s).sum(dim=1)
    e_old = energy(out_old)
    e_new = energy(out_new)
    print(f"energy old (mean/min): {e_old.mean():.6f} / {e_old.min():.6f}")
    print(f"energy new (mean/min): {e_new.mean():.6f} / {e_new.min():.6f}")
    print("OK: refactor preserves fp32 semantics.")


if __name__ == "__main__":
    main()
