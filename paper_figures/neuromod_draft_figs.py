"""Draw the paper's interpolation and lesion figures from saved result CSVs.

    python -B paper_figures/neuromod_draft_figs.py --figure interpolation \\
        --interpolation-csv <03_interpolation/.../results.csv> [--out-dir DIR]
    python -B paper_figures/neuromod_draft_figs.py --figure lesion \\
        --lesion-csv <04_lesion/.../results.csv> [--out-dir DIR]

No training or network evaluation is performed; only NumPy and Matplotlib are
needed. Without --out-dir, figures are written to figures/ in the repository
root.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 11, 23]


def fig_interpolation(csv_path, out_dir):
    """Plot saved interpolation results without network evaluation."""
    pairs = (("food", "threat"), ("threat", "shelter"), ("shelter", "food"))
    labels = ("Foraging → Avoidance", "Avoidance → Sheltering",
              "Sheltering → Foraging")
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(seed, a, b, i / 10) for seed in SEEDS
                for a, b in pairs for i in range(11)}
    actual = {(int(r["seed"]), r["a"], r["b"], float(r["w"])) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Expected one result per seed, behavior pair, and weight")
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    for (a, b), label, color in zip(pairs, labels,
                                   ("tab:blue", "tab:orange", "tab:green")):
        selected = {(int(r["seed"]), float(r["w"])): float(r["implied_lambda"])
                    for r in rows if (r["a"], r["b"]) == (a, b)}
        weights = np.arange(11) / 10
        values = np.asarray([[selected[seed, w] for w in weights] for seed in SEEDS])
        if not np.isfinite(values).all():
            raise ValueError("Interpolation results contain non-finite values")
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=0)
        ax.plot(weights, mean, color=color, lw=2, marker="o", ms=5, label=label)
        ax.fill_between(weights, mean - sd, mean + sd, color=color, alpha=0.2)
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1.5,
            label=r"$\hat{\lambda}=w$")
    ax.set_xlabel(r"Noise-field mixing weight $w$", fontsize=12)
    ax.set_ylabel(r"Implied output mixing $\hat{\lambda}$", fontsize=12)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05),
           xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig_noise_interpolation.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved fig_noise_interpolation.png and .pdf")


def fig_lesion(csv_path, out_dir):
    """Plot saved lesion results; preserve the CSV's group names."""
    category_codes = {"food": 0, "threat": 1, "shelter": 2}
    group_codes = {"shared": 0, "a-only": 1, "b-only": 2, "random": 3}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(seed, a, b, group) for seed in SEEDS
                for a, b in (("food", "threat"), ("food", "shelter"),
                             ("threat", "shelter"))
                for group in group_codes}
    actual = {(int(row["seed"]), row["a"], row["b"], row["group"])
              for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Expected one result per seed, behavior pair, and lesion group")
    r = np.asarray([
        [0, int(row["seed"]), category_codes[row["a"]],
         category_codes[row["b"]], group_codes[row["group"]],
         int(row["n_units"]), float(row["delta_err_a"]),
         float(row["delta_err_b"])] for row in rows
    ], dtype=float)
    if not np.isfinite(r).all():
        raise ValueError("Lesion results contain non-finite values")
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = ["Foraging / Avoidance", "Foraging / Sheltering",
                  "Avoidance / Sheltering"]
    groups = ["Shared", r"$b_1$ only", r"$b_2$ only", "Random"]
    font_size = 13
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    w = 0.36
    for ax, (ca, cb), pname in zip(axes, pairs, pair_names):
        sel = r[(r[:, 2] == ca) & (r[:, 3] == cb)]
        ma = [sel[sel[:, 4] == g][:, 6].mean() for g in range(4)]
        mb = [sel[sel[:, 4] == g][:, 7].mean() for g in range(4)]
        # Each CSV row already averages the lesion draws within one network.
        # Match the intensity/interpolation bands: SD across 3 networks, ddof=0.
        sa = [sel[sel[:, 4] == g][:, 6].std(ddof=0) for g in range(4)]
        sb = [sel[sel[:, 4] == g][:, 7].std(ddof=0) for g in range(4)]
        x = np.arange(4)
        error_style = dict(ecolor=".15", elinewidth=1.1, capthick=1.1)
        ax.bar(x - w / 2, ma, w, color="tab:blue", label=r"Behavior $b_1$")
        ax.bar(x + w / 2, mb, w, color="tab:red", label=r"Behavior $b_2$")
        # Omit the non-target behavior's error bar in each Only condition.
        # Select by condition, never by a numerical threshold; all means stay.
        show_a, show_b = np.array([0, 1, 3]), np.array([0, 2, 3])
        ax.errorbar((x - w / 2)[show_a], np.asarray(ma)[show_a],
                    yerr=np.asarray(sa)[show_a], fmt="none", capsize=3,
                    **error_style)
        ax.errorbar((x + w / 2)[show_b], np.asarray(mb)[show_b],
                    yerr=np.asarray(sb)[show_b], fmt="none", capsize=3,
                    **error_style)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=font_size)
        ax.set_xlabel(pname, fontsize=font_size)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.grid(alpha=0.2, axis="y")
        ax.tick_params(labelsize=font_size)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Change in task error (MSE)", fontsize=font_size)
    axes[0].set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=font_size,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "fig_lesion.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "fig_lesion.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_lesion.png and .pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved interpolation/lesion CSVs; no learning or evaluation")
    parser.add_argument("--figure", choices=("lesion", "interpolation"), required=True)
    parser.add_argument("--lesion-csv", type=Path)
    parser.add_argument("--interpolation-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    output = args.out_dir or ROOT / "figures"
    if args.figure == "interpolation":
        if args.lesion_csv is not None:
            parser.error("--lesion-csv requires --figure lesion")
        if args.interpolation_csv is None:
            parser.error("--figure interpolation requires --interpolation-csv")
        fig_interpolation(args.interpolation_csv, output)
    else:
        if args.interpolation_csv is not None:
            parser.error("--interpolation-csv requires --figure interpolation")
        if args.lesion_csv is None:
            parser.error("--figure lesion requires --lesion-csv")
        fig_lesion(args.lesion_csv, output)


if __name__ == "__main__":
    main()
