"""Generate a 3D Edwards-Anderson coupling file in the repo's standard format.

The repository's simulations expect couplings at
``Data/Alpha/Couplings/couplings_L{L}_R1_seed{seed}.txt`` with one edge per
line formatted as ``i j J_ij`` (site indices use the same raster convention as
``monte_carlo.get_indices``: ``index = x*L*L + y*L + z``).

We generate the three nearest-neighbor bonds per site (x+1, y+1, z+1 with
periodic boundaries) with Gaussian couplings J ~ N(0, 1), matching the paper's
model definition (Sec. II.A).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def site_index(x: int, y: int, z: int, L: int) -> int:
    return x * L * L + y * L + z


def generate_couplings(L: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = site_index(x, y, z, L)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = site_index((x + dx) % L, (y + dy) % L, (z + dz) % L, L)
                    J = float(rng.standard_normal())
                    edges.append((i, j, J))
    edges.sort(key=lambda t: (t[0], t[1]))
    return np.array(edges, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Data/Alpha/Couplings"),
        help="Destination directory (repo-relative by default).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"couplings_L{args.L}_R1_seed{args.seed}.txt"
    edges = generate_couplings(args.L, args.seed)

    with out_path.open("w") as f:
        for i, j, J in edges:
            f.write(f"{int(i)} {int(j)} {float(J):.6f}\n")

    print(f"Wrote {len(edges)} edges to {out_path}")


if __name__ == "__main__":
    main()
