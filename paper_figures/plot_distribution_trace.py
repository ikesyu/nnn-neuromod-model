"""Plot saved trace CSVs only. No torch/model/simulation imports.

Draws the representative life (nearest distances, mixture weights, events,
commanded speed) and the pooled food/shelter usage counts of a region run.
The distance row shows food during the day, and threat/shelter at all times.
Without --out-dir, figures are written to figures/ in the repository root.
Distances use the recorded post-update agent positions. Food availability is
reconstructed using the original depletion/regrowth rule, and every eating
event must match the recorded event. A missing available food is NaN, not 0.
"""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_common"))  # trace_records
if "MPLCONFIGDIR" not in os.environ or "XDG_CACHE_HOME" not in os.environ:
    _plot_cache = tempfile.TemporaryDirectory(prefix="nnn-matplotlib-")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(_plot_cache.name) / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(_plot_cache.name) / "cache"))
    atexit.register(_plot_cache.cleanup)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import MaxNLocator
from trace_records import read_csv, validate_episode


def load_evaluation(run):
    run = Path(run)
    meta = json.loads((run / "metadata/evaluation.json").read_text(encoding="utf-8"))
    if not meta["complete"] or meta["c"] != 1.0:
        raise ValueError("Incomplete evaluation or unexpected concentration")
    rows = read_csv(run / "data/episodes.csv")
    expected = {(seed, episode) for seed in meta["seeds"] for episode in range(meta["episodes"])}
    identities = [(int(r["training_seed"]), int(r["episode"])) for r in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Missing or duplicate episodes")
    if len(list((run / "data/traces").glob("*.csv"))) != len(expected):
        raise ValueError("Unexpected number of trace files")
    for row in rows:
        seed, episode = int(row["training_seed"]), int(row["episode"])
        if int(row["frames"]) != meta["frames"]:
            raise ValueError("Unexpected frame count")
        params = meta["loop_params"][str(seed)]
        trace = read_csv(run / "data/traces" / f"seed{seed}_episode{episode}.csv")
        validate_episode(trace, row, params["circadian_period"], params["day_fraction"])
    if meta.get("method") != "region":
        raise ValueError("A region-method evaluation is required")
    expected_rep = dict(training_seed=7, kind="original_one_life",
                        torch_seed=1007, environment_seed=7)
    if meta["representative"] != expected_rep or meta.get("representative_in_pooled_counts") is not False:
        raise ValueError("Original region life must be separate from pooled episodes")
    summary = json.loads((run / "data/representative_summary.json").read_text(encoding="utf-8"))
    if (int(summary["training_seed"]), int(summary["episode"]), int(summary["frames"])) != (7, -1, meta["frames"]):
        raise ValueError("Wrong separate representative identity/frame count")
    trace = read_csv(run / "data/representative_trace.csv")
    params = meta["representative_loop_params"]
    validate_episode(trace, summary, params["circadian_period"], params["day_fraction"])
    return meta, rows, trace


def night_bands(axes, t, night):
    edges = np.diff(np.r_[False, night, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        for ax in axes:
            ax.axvspan(t[start] - .5, t[end - 1] + .5, color=".80", alpha=.45, zorder=0)


def distance_series(run, meta, trace):
    """Recover nearest-object distances at the same time as each saved row.

The original world.apply_food_depletion starts with all foods available.
An eaten food receives its regrowth timer; subsequent frames first decrement
that timer, and only a later frame with timer=0 and distance>2.5*eat_radius
can restore it. This replays bookkeeping along SAVED positions, not agent
motion, predictions, random noise, or a new behavioral evaluation.
"""
    objects = json.loads((Path(run) / "metadata/initial_objects.json").read_text(encoding="utf-8"))
    food = np.asarray(objects["food"], dtype=np.float32)
    shelter = np.asarray(objects["shelter"], dtype=np.float32)
    seed, episode = int(trace[0]["training_seed"]), int(trace[0]["episode"])
    params = meta["representative_loop_params"]
    if not params["food_respawn"]:
        raise ValueError("Review food respawn rules before plotting this evaluation")
    radius = float(params["eat_radius"])
    regrow = int(params["food_regrow_frames"])
    available = np.ones(len(food), dtype=bool)
    timers = np.zeros(len(food), dtype=np.int32)
    food_distance = np.full(len(trace), np.nan)
    shelter_distance = np.empty(len(trace))
    threat_distance = np.asarray([float(r["d_threat"]) for r in trace])
    food_state = np.empty((len(trace), len(food)), dtype=bool)
    for i, row in enumerate(trace):
        position = np.asarray([float(row["pos_x"]), float(row["pos_y"])], dtype=np.float32)
        eaten = -1
        for j, point in enumerate(food):
            d = float(np.linalg.norm(point - position))
            if d < radius and available[j]:
                available[j] = False
                timers[j] = regrow
                eaten = j
            elif not available[j]:
                if timers[j] > 0:
                    timers[j] -= 1
                elif d > 2.5 * radius:
                    available[j] = True
        if eaten != int(row["ate_index"]):
            raise ValueError(f"Food reconstruction disagrees at seed={seed}, episode={episode}, frame={i}")
        food_state[i] = available
        if np.any(available):
            food_distance[i] = float(np.linalg.norm(food[available] - position[None, :], axis=1).min())
        shelter_distance[i] = float(np.linalg.norm(shelter - position[None, :], axis=1).min())
        if int(row["home_event"]) and shelter_distance[i] >= float(params["shelter_radius"]):
            raise ValueError("A recorded home arrival is outside every shelter")
    if not np.isfinite(threat_distance).all() or np.any(threat_distance < 0):
        raise ValueError("Invalid saved threat distance")
    return {"food": food_distance, "threat": threat_distance,
            "shelter": shelter_distance, "food_available": food_state}


def row_legend(ax, columns, handlelength=2.1):
    """One horizontal legend per data type; no column-major wraparound."""
    return ax.legend(ncol=columns, loc="lower center", bbox_to_anchor=(.5, 1.015),
                     borderaxespad=0, frameon=False, fontsize=12.5,
                     handlelength=handlelength, columnspacing=1.8)


def save_figure(fig, target):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    for legend in legends:
        if any(legend.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer))
               for ax in fig.axes):
            raise ValueError("Legend overlaps data axes")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.12)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def draw(run, output=None, distance_max=1.2):
    meta, rows, trace = load_evaluation(run)
    output = Path(output) if output else ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    values = lambda key: np.array([float(r[key]) for r in trace])
    t, night = values("frame"), values("is_night").astype(bool)
    distances = distance_series(run, meta, trace)
    fig = plt.figure(figsize=(10, 8.5))
    grid = fig.add_gridspec(4, 1, left=.11, right=.98, bottom=.075, top=.90,
                           height_ratios=[2.8, 2.2, .85, 1.4], hspace=.50)
    ax_distance = fig.add_subplot(grid[0])
    for key, color, label, linestyle in [
            ("food", "tab:green", "Available food", "-"),
            ("threat", "tab:red", "Threat", (0, (6, 3))),
            ("shelter", "tab:blue", "Shelter", (0, (1, 2)))]:
        # Hide food only at night. Keep both other distances throughout.
        # Never replace missing food distances with zero or change raw values.
        data = np.where(night, np.nan, distances[key]) if key == "food" else distances[key]
        if np.any(data[np.isfinite(data)] > distance_max):
            raise ValueError("A displayed distance would be clipped; increase --distance-max")
        # Solid / long dashed / dotted lines remain distinct in monochrome.
        # A narrow white edge separates crossings, without smoothing values.
        ax_distance.plot(t, data, color=color, lw=1.9,
                         linestyle=linestyle, label=label,
                         dash_capstyle="round", solid_capstyle="round",
                         path_effects=[path_effects.Stroke(linewidth=2.7,
                                                          foreground="white"),
                                       path_effects.Normal()])
    ticks = np.arange(0, distance_max + 1e-9, .4)
    ax_distance.set(ylabel="Nearest distance", ylim=(0, distance_max), yticks=ticks)
    row_legend(ax_distance, 3, handlelength=3.4)
    ax_weights = fig.add_subplot(grid[1], sharex=ax_distance)
    ax_events = fig.add_subplot(grid[2], sharex=ax_distance)
    ax_speed = fig.add_subplot(grid[3], sharex=ax_distance)
    for ax in (ax_distance, ax_weights, ax_events):
        ax.tick_params(axis="x", labelbottom=False)
    axes = [ax_distance, ax_weights, ax_events, ax_speed]
    ax_weights.stackplot(t, values("w_food"), values("w_threat"), values("w_shelter"),
                      colors=["tab:green", "tab:red", "tab:blue"],
                      labels=["Foraging", "Avoidance", "Sheltering"], alpha=.85)
    ax_weights.set(ylabel="Mixture weight", ylim=(0, 1), yticks=[0, .5, 1])
    row_legend(ax_weights, 3)
    for key, y, marker, color, label in [("ate", 1, "v", "tab:green", "Eating"),
                                        ("home_event", 0, "s", "tab:blue", "Shelter arrival")]:
        frames = t[values(key).astype(bool)]
        ax_events.scatter(frames, np.full(len(frames), y), s=42, marker=marker,
                        color=color, label=label, zorder=3)
    ax_events.set(ylim=(-.6, 1.6), yticks=[0, 1], yticklabels=["Shelter", "Food"])
    row_legend(ax_events, 2)
    ax_speed.plot(t, values("speed"), color=".25", lw=1)
    ax_speed.set(xlabel="Time (steps)", ylabel="Commanded speed")
    ax_speed.set_ylim(bottom=0)
    night_bands(axes, t, night)
    for ax in axes:
        ax.set_xlim(-.5, len(trace) - .5)
        ax.grid(alpha=.2, zorder=0)
    ax_distance.grid(alpha=.12, zorder=0)
    if meta.get("test_only"):
        fig.text(.5, -.01, "TEST DATA - NOT RESEARCH RESULTS", color="red", ha="center")
    save_figure(fig, output / "fig_region_ethogram_distances")

    foods = np.array([sum(int(r[f"food_{i}_hits"]) for r in rows) for i in range(3)])
    dens = np.array([sum(int(r[f"den_{i}_hits"]) for r in rows) for i in range(2)])
    diet = float(np.mean([float(r["diet_evenness"]) for r in rows]))
    den = float(np.mean([float(r["den_evenness"]) for r in rows]))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    fig.subplots_adjust(left=.10, right=.97, top=.84, bottom=.19, wspace=.42)
    for ax, counts, color, prefix, ylabel in [
            (axes[0], foods, "tab:green", "Food", "Eating events"),
            (axes[1], dens, "tab:blue", "Shelter", "Shelter arrivals")]:
        ax.bar(np.arange(len(counts)), counts, color=color, edgecolor="black", linewidth=.7)
        ax.set_xticks(np.arange(len(counts)), [f"{prefix} {i + 1}" for i in range(len(counts))])
        ax.set_ylabel(ylabel)
        ticks = MaxNLocator(nbins=3, integer=True, steps=[1, 2, 5, 10]).tick_values(
            0, max(1, float(counts.max())) * 1.08)
        ax.set_yticks(ticks)
        ax.grid(alpha=.2, axis="y")
        ax.set_axisbelow(True)
        ax.set_ylim(0, ticks[-1])
    if meta.get("test_only"):
        fig.text(.5, -.01, "TEST DATA - NOT RESEARCH RESULTS", color="red", ha="center")
    save_figure(fig, output / "fig_region_diversity")
    return {"food_total_counts": foods.tolist(), "den_total_counts": dens.tolist(),
            "mean_episode_diet_evenness": diet, "mean_episode_den_evenness": den}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot existing CSVs only; no learning or simulation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--distance-max", type=float, default=1.2)
    args = parser.parse_args()
    stats = draw(args.run_dir, args.out_dir, distance_max=args.distance_max)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Saved figures from recorded CSVs. No training or behavioral simulation executed.")
