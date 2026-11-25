"""Plot learning curves: mean over seeds with min–max band.

    python plot_runs.py runs/q1_baseline runs/q2_no_target runs/q2_no_replay -o figures/Q2_ablation.png
    python plot_runs.py runs/q1_baseline:"baseline" runs/q3_faulty:"retrained on faulty" -o figures/Q3.png

Reads runs/<group>/seed*/monitor.csv (written during training). Use --metric length to plot
episode length instead of return (useful in Q4, where the return depends on your reward design).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_group(group: Path, metric: str, grid: np.ndarray, window: int):
    curves = []
    for mon in sorted(group.glob("seed*/monitor.csv")):
        df = pd.read_csv(mon, skiprows=1)  # first line is a JSON header
        if df.empty:
            continue
        steps = df["l"].cumsum().to_numpy()
        y = (df["r"] if metric == "return" else df["l"]).rolling(window, min_periods=1).mean().to_numpy()
        curves.append(np.interp(grid, steps, y, left=np.nan, right=np.nan))
    return np.array(curves)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="run group directories, optionally dir:label")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--metric", choices=["return", "length"], default="return")
    ap.add_argument("--window", type=int, default=10, help="moving-average window (episodes)")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for spec in a.groups:
        path, _, label = spec.partition(":")
        path = Path(path)
        mons = list(path.glob("seed*/monitor.csv"))
        if not mons:
            print(f"warning: no seed*/monitor.csv in {path}")
            continue
        # stop where the shortest seed's log ends, so the mean never jumps when a seed drops out
        tmax = min(pd.read_csv(m, skiprows=1)["l"].sum() for m in mons)
        grid = np.linspace(0, tmax, 300)
        c = load_group(path, a.metric, grid, a.window)
        mean = np.nanmean(c, axis=0)
        line, = ax.plot(grid, mean, lw=2, label=f"{label or path.name} (n={len(c)} seeds)")
        ax.fill_between(grid, np.nanmin(c, axis=0), np.nanmax(c, axis=0), color=line.get_color(), alpha=0.18)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode {a.metric} (moving avg, {a.window} episodes)")
    ax.set_title(a.title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
