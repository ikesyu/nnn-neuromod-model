"""Figure for the noise-intensity experiment. Reads CSV/JSON only."""
import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (7, 11, 23)


def verify_keys(rows, keys, expected):
    found = [tuple(row[key] for key in keys) for row in rows]
    if len(found) != len(expected) or set(found) != set(expected):
        raise ValueError("Missing, duplicate, or unexpected experiment conditions")


def values(rows, col):
    a = np.array([float(r[col]) for r in rows])
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite values: {col}")
    return a


def line(ax, x, y, color="tab:blue"):
    mean, sd = y.mean(1), y.std(1, ddof=0)
    ax.plot(x, mean, marker="o", markersize=4, color=color)
    ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.2)


def matrix(rows, axis, levels, col):
    indexed = {(int(r["seed"]), float(r[axis])): r for r in rows}
    return np.stack([values([indexed[(s,x)] for s in SEEDS], col) for x in levels])


def intensity(rows, settings):
    c = settings["c"]
    verify_keys(rows, ("seed", "c"), [(str(s), str(x)) for s,x in itertools.product(SEEDS,c)])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, col, label in zip(axes, ("signal", "task_err"), ("Output variation (signal)", "Task error (MSE)")):
        line(ax, c, matrix(rows, "c", c, col))
        ax.axvline(1, color=".5", ls="--", lw=1)
        ax.set(xscale="log", xlabel="Noise multiplier c", ylabel=label)
        ax.grid(alpha=.25)
    return fig, "fig_noise_intensity"


def main():
    p = argparse.ArgumentParser(description="Plot saved CSV only; no learning/evaluation")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    a = p.parse_args()
    meta = json.loads((a.run_dir / "evaluation.json").read_text(encoding="utf-8"))
    if not meta["complete"] or meta["kind"] != "intensity":
        raise ValueError("A completed intensity evaluation is required")
    source = a.run_dir / "results.csv"
    with source.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, name = intensity(rows, meta["settings"])
    out = a.out_dir or a.run_dir / "figures"
    out.mkdir(parents=True)  # refuse an existing figure directory
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    with (out / "figure_source.json").open("x", encoding="utf-8") as f:
        json.dump(dict(csv=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       plot_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       invocation=sys.argv, matplotlib=matplotlib.__version__, numpy=np.__version__), f, indent=2)
    print(f"Saved {out}; no evaluation executed")
