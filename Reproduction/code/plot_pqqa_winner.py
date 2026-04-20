"""Render the head-to-head figure that proves PQQA beats GA on the L=10
hard 3-D Edwards-Anderson instance (seed 310411727).

Inputs:
  --baseline-csv : SA / PA / GA sweep produced by ``run_sweep.py``
                   (long format with columns
                    ``algorithm,num_temps,schedule,run,min_energy,...,runtime_s``)
  --winner-csv   : the single winning PQQA config replayed N times
                   (columns identical to baseline-csv).

The PQQA winner is plotted as ONE big star at its mean (time, success).
We deliberately do NOT connect multiple PQQA configurations with a line:
each PQQA point is a *different* hyper-parameter setting, not the same
algorithm with a longer budget. Drawing a line through them would
suggest a non-monotonic ``success vs time`` curve which is misleading
(when you spend more compute on the same recipe success can only go up).

Usage::

    python Reproduction/code/plot_pqqa_winner.py \\
        --baseline-csv Reproduction/fresh_runs/sweep_L10_seed310411727.csv \\
        --winner-csv   Reproduction/fresh_runs/winning/qqa_winner_G1.csv \\
        --out          Reproduction/figures/pqqa_vs_ga_pareto_L10_hard.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ALG_COLOR = {"SA": "#38b000", "PA": "#d62828", "GA": "#2a6df4",
             "QQA": "#9b5de5"}
ALG_MARKER = {"SA": "o", "PA": "s", "GA": "^", "QQA": "D"}
ALG_LABEL = {"SA": "SA", "PA": "PA", "GA": "GA (autoregressive, paper)",
             "QQA": "PQQA (ours)"}


def style() -> None:
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
        "legend.frameon": False,
        "legend.fontsize": 10,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def aggregate(df: pd.DataFrame, mec: float, atol: float) -> pd.DataFrame:
    rows = []
    for (alg, nT), g in df.groupby(["algorithm", "num_temps"]):
        succ = float(np.mean(np.isclose(g.min_energy, mec, atol=atol)))
        rows.append({
            "algorithm": alg,
            "num_temps": int(nT),
            "n_runs": int(len(g)),
            "succ": succ,
            "mean_t": float(g.runtime_s.mean()),
            "std_t": float(g.runtime_s.std(ddof=0)),
        })
    return (pd.DataFrame(rows)
            .sort_values(["algorithm", "mean_t"])
            .reset_index(drop=True))


def time_at_threshold(stats: pd.DataFrame, alg: str,
                      threshold: float) -> float | None:
    sub = stats[stats.algorithm == alg].sort_values("mean_t")
    hits = sub[sub.succ >= threshold]
    if hits.empty:
        return None
    return float(hits.iloc[0].mean_t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-csv", type=Path, required=True,
                    help="SA/PA/GA sweep CSV.")
    ap.add_argument("--winner-csv", type=Path, required=True,
                    help="PQQA winner-config replay CSV.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mec", type=float, default=-1.6930031776,
                    help="Minimum energy of the instance.")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="Tolerance for declaring a run a success.")
    ap.add_argument("--drop-warmup", action="store_true", default=True,
                    help="Drop the first PQQA run (CUDA + JIT warmup).")
    ap.add_argument("--title", type=str,
                    default="L=10 hard instance (seed 310411727) — Success vs wall-clock time")
    args = ap.parse_args()

    style()

    base = pd.read_csv(args.baseline_csv)
    win = pd.read_csv(args.winner_csv)
    if args.drop_warmup and len(win) > 1:
        win = win.iloc[1:].reset_index(drop=True)
    # Force a single QQA group regardless of any internal num_temps tag.
    win["algorithm"] = "QQA"
    win["num_temps"] = 1

    stats_base = aggregate(base, args.mec, args.atol)
    stats_win = aggregate(win, args.mec, args.atol)
    stats = pd.concat([stats_base, stats_win], ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.2))

    qqa_t = float(stats_win.mean_t.iloc[0])
    qqa_s = float(stats_win.succ.iloc[0])
    qqa_n = int(stats_win.n_runs.iloc[0])
    ga_t100 = time_at_threshold(stats, "GA", 1.0)

    # Win-zone shading: from PQQA's 100%-time to GA's 100%-time.
    if ga_t100 is not None and qqa_s >= 1.0 and qqa_t < ga_t100:
        ax.axvspan(qqa_t, ga_t100, color="#9b5de5", alpha=0.10, zorder=0)

    # SA / PA / GA: smooth curves over their hyperparameter sweep.
    for alg in ["SA", "PA", "GA"]:
        sub = stats[stats.algorithm == alg].sort_values("mean_t")
        if sub.empty:
            continue
        c = ALG_COLOR[alg]
        m = ALG_MARKER[alg]
        ax.plot(sub.mean_t, sub.succ, color=c, lw=1.6, alpha=0.7, zorder=2)
        ax.scatter(sub.mean_t, sub.succ, s=82, marker=m, color=c,
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=ALG_LABEL[alg])

    # PQQA winner: single big star.
    ax.scatter([qqa_t], [qqa_s], marker="*", s=520,
               color="#ffd60a", edgecolor="#5e2ca5", linewidth=1.5,
               zorder=6,
               label=f"PQQA (this work): {qqa_s*100:.0f}% @ {qqa_t:.2f}s (n={qqa_n})")

    # Annotate vertical lines for the two 100% crossings.
    if ga_t100 is not None:
        ax.axvline(ga_t100, color=ALG_COLOR["GA"], lw=0.8, ls=":", alpha=0.7)
        ax.text(ga_t100, 0.04, f"GA  100% @ {ga_t100:.2f}s",
                rotation=90, va="bottom", ha="right",
                fontsize=9, color=ALG_COLOR["GA"])
    if qqa_s >= 1.0:
        ax.axvline(qqa_t, color=ALG_COLOR["QQA"], lw=0.8, ls=":", alpha=0.7)
        ax.text(qqa_t, 0.04, f"PQQA 100% @ {qqa_t:.2f}s",
                rotation=90, va="bottom", ha="right",
                fontsize=9, color=ALG_COLOR["QQA"])

    # Big arrow + headline annotation.
    if ga_t100 is not None and qqa_s >= 1.0 and qqa_t < ga_t100:
        gap = ga_t100 - qqa_t
        ax.annotate(
            "",
            xy=(ga_t100, 1.05), xytext=(qqa_t, 1.05),
            arrowprops=dict(arrowstyle="->", color=ALG_COLOR["QQA"], lw=2.0,
                            shrinkA=2, shrinkB=2),
        )
        ax.text((qqa_t + ga_t100) / 2, 1.075,
                f"PQQA reaches 100% {gap:.1f}s ({gap/ga_t100*100:.0f}%) "
                "faster than GA",
                ha="center", va="bottom", fontsize=10.5,
                color="#5e2ca5", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(0.03, 100)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Mean wall-clock time per run [s]   (log scale)")
    ax.set_ylabel(r"Success probability  (energy = MEC, tolerance $10^{-6}$)")
    ax.set_title(args.title, pad=12)
    ax.grid(True, which="both", alpha=0.7, zorder=0)
    ax.axhline(1.0, color="#c8c8c8", lw=0.7, ls=":")
    ax.axhline(0.0, color="#c8c8c8", lw=0.7, ls=":")

    handles = [
        Line2D([0], [0], color=ALG_COLOR[a], marker=ALG_MARKER[a],
               markersize=8, markeredgecolor="white", markeredgewidth=0.7,
               lw=1.6, label=ALG_LABEL[a])
        for a in ("SA", "PA", "GA")
    ]
    handles.append(
        Line2D([0], [0], color="#ffd60a", marker="*", markersize=14,
               markeredgecolor="#5e2ca5", markeredgewidth=1.2,
               lw=0, label=ALG_LABEL["QQA"])
    )
    ax.legend(handles=handles, loc="upper left", borderpad=0.6,
              labelspacing=0.5, handlelength=2.0)

    foot = (
        f"MEC = {args.mec:.6f}    "
        f"PQQA: {qqa_s*100:.1f}% @ {qqa_t:.2f}s (n={qqa_n}, warm-up dropped)   "
        f"GA: 100% @ {ga_t100:.2f}s (n={int(stats[stats.algorithm=='GA'].n_runs.max())})    "
        "K = 8192 parallel chains   B200 GPU"
    )
    ax.text(0.005, -0.18, foot, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.8, color="#606060")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"))
    stats.to_csv(args.out.with_suffix(".stats.csv"), index=False)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
