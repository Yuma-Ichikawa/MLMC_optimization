"""Render a Fig.2-style success-vs-time plot from a fresh reproduction sweep.

Reads the CSV produced by ``run_sweep.py`` and plots, for each algorithm, the
success probability (fraction of runs that reach the best observed energy)
versus the mean wall-clock time, together with a normalised logistic fit.

The style is intentionally modernised: dark-ink palette, Inter-like sans-serif,
hair-line grid, single-pixel spines. It is saved as PNG at 300 dpi and as PDF
to ``Reproduction/figures/``.

Run from the repo root::

    python Reproduction/code/plot_success_vs_time.py \
        --csv Reproduction/fresh_runs/sweep_L10_seed1736329224.csv \
        --out Reproduction/figures/success_vs_time_L10.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import expit

ALG_ORDER = ["SA", "PA", "GA"]
ALG_COLOR = {"SA": "#38b000", "PA": "#d62828", "GA": "#2a6df4"}
ALG_LABEL = {"SA": "SA", "PA": "PA", "GA": r"$\mathrm{GA}_{15}$"}
ALG_MARKER = {"SA": "o", "PA": "s", "GA": "^"}


def logistic_norm(x, A, B):
    """Normalised logistic with S(0)=0, S(+inf)=1."""
    x = np.asarray(x, dtype=float)
    Lx = expit(A * (x - B))
    L0 = expit(-A * B)
    return np.clip((Lx - L0) / max(1.0 - L0, 1e-15), 0.0, 1.0)


def fit_curve(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x_pos = x[mask]
    B0 = float(np.median(x_pos))
    iqr = np.subtract(*np.percentile(x_pos, [75, 25])) if x_pos.size >= 2 else max(B0, 1.0)
    A0 = 4.0 / max(iqr, 1e-6)
    try:
        popt, _ = curve_fit(
            logistic_norm, x[mask], y[mask],
            p0=(A0, B0),
            bounds=([1e-8, 0.0], [10.0, np.inf]),
            maxfev=20000,
        )
        return popt
    except Exception:
        return None


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def compute_stats(df, mec):
    rows = []
    for (algorithm, nT), grp in df.groupby(["algorithm", "num_temps"]):
        n_runs = len(grp)
        # A run "succeeds" if it reaches the best energy ever observed, within
        # a tiny floating-point tolerance (energies are reported to 1e-10).
        success_rate = float(np.mean(np.isclose(grp["min_energy"], mec, atol=1e-6)))
        mean_time = float(grp["runtime_s"].mean())
        std_time = float(grp["runtime_s"].std(ddof=0))
        rows.append({
            "algorithm": algorithm,
            "num_temps": int(nT),
            "n_runs": n_runs,
            "success_rate": success_rate,
            "mean_time": mean_time,
            "std_time": std_time,
        })
    return pd.DataFrame(rows).sort_values(["algorithm", "num_temps"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", type=str,
                    default="L=10 easy instance (seed 1736329224) – $M=2^{13}$ population")
    ap.add_argument("--mec", type=float, default=None,
                    help="Reference minimum energy. Default: min across all runs.")
    args = ap.parse_args()

    style()
    df = pd.read_csv(args.csv)
    mec = float(df["min_energy"].min()) if args.mec is None else args.mec
    print(f"[plot] best energy observed across all runs: {mec:.10f}")
    stats = compute_stats(df, mec)
    stats.to_csv(args.out.with_suffix(".stats.csv"), index=False)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))
    x_grid = np.geomspace(max(stats["mean_time"].min() * 0.4, 1e-3),
                          stats["mean_time"].max() * 2.5, 400)

    legend_handles = []
    for algorithm in ALG_ORDER:
        sub = stats[stats["algorithm"] == algorithm].sort_values("mean_time")
        if sub.empty:
            continue
        color = ALG_COLOR[algorithm]
        y = sub["success_rate"].to_numpy()
        # Only fit a logistic if we actually observe a transition. If every run
        # failed (or every run succeeded) the fit is unidentifiable and any
        # curve drawn would be a misleading extrapolation.
        if 0.0 < y.max() and y.min() < 1.0:
            popt = fit_curve(sub["mean_time"].to_numpy(), y)
            if popt is not None:
                ax.plot(x_grid, logistic_norm(x_grid, *popt), color=color, lw=2.5,
                        alpha=0.85, zorder=2)
        ax.plot(sub["mean_time"], y, color=color, lw=1.0, alpha=0.35, zorder=1)
        ax.scatter(sub["mean_time"], y, s=68,
                   marker=ALG_MARKER[algorithm], color=color,
                   edgecolor="white", linewidth=0.7, zorder=3)
        legend_handles.append(
            Line2D([0], [0], color=color, marker=ALG_MARKER[algorithm], markersize=8,
                   markeredgecolor="white", markeredgewidth=0.7, lw=2.5,
                   label=ALG_LABEL[algorithm])
        )

    ax.axhline(1.0, color="#c8c8c8", lw=0.8, ls=":")
    ax.axhline(0.0, color="#c8c8c8", lw=0.8, ls=":")

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Mean wall-clock time per run [s]")
    ax.set_ylabel("Success probability")
    ax.set_title(args.title, pad=12)
    ax.grid(True, which="both", axis="both", alpha=0.7, zorder=0)

    ax.legend(handles=legend_handles, loc="upper left", handlelength=2.0,
              borderpad=0.6, labelspacing=0.6)

    ax.text(0.01, -0.18,
            f"MEC = {mec:.4f}  |  {int(stats['n_runs'].max())} runs / configuration"
            "   |   schedule = $\\log T$,  $T\\in[0.1,1.92]$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#606060")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
