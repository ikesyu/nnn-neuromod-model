"""Record the original rollout without changing dynamics or consuming RNG."""
from contextlib import contextmanager
import csv
import math
from pathlib import Path
import numpy as np

TRACE_COLUMNS = ["training_seed", "episode", "frame", "clock", "phase", "is_night",
                 "w_food", "w_threat", "w_shelter", "speed", "ate", "ate_index",
                 "home_event", "den_index", "pos_x", "pos_y", "d_threat", "contact"]
SUMMARY_COLUMNS = ["training_seed", "episode", "c", "frames", "foods", "foods_per_1k",
                   "nights", "night_home_rate", "contact_frac", "wall_frac", "stall_frac",
                   "food_0_hits", "food_1_hits", "food_2_hits", "den_0_hits", "den_1_hits",
                   "diet_evenness", "den_evenness", "mean_speed", "speed_ref"]
METRICS = ["foods_per_1k", "night_home_rate", "contact_frac", "wall_frac", "stall_frac",
           "diet_evenness", "den_evenness", "mean_speed"]


@contextmanager
def record_frames(world, seed, episode, records):
    """Temporarily observe advance_frame; world.rollout still owns evaluation.

    Synchronous use only. No additional prediction or random draw is made.
    All recorded scalars are copied after the original frame update.
    """
    original = world.advance_frame
    prev_goal, got_home = None, False

    def observed(state, predict, fields, params, **kwargs):
        nonlocal prev_goal, got_home
        if prev_goal is None:
            prev_goal = state["goal"]
        rec = original(state, predict, fields, params, **kwargs)
        goal = state["goal"]
        if goal == "shelter" and prev_goal == "food":
            got_home = False
        home_event = goal == "shelter" and rec["inside_shelter"] and not got_home
        den = -1
        if home_event:
            got_home = True
            den = int(np.argmin(np.linalg.norm(
                state["objects"]["shelter"] - state["pos"][None, :], axis=1)))
        prev_goal = goal
        w = np.asarray(rec["weights"]).copy()
        position = state["pos"].copy()
        phase = world.circadian_phase(state["clock"], params.circadian_period)
        records.append(dict(zip(TRACE_COLUMNS, [
            int(seed), int(episode), int(kwargs["frame"]), int(state["clock"]),
            float(phase), int(goal == "shelter"), float(w[0]), float(w[1]), float(w[2]),
            float(rec["speed"]), int(rec["ate"]), int(rec["ate_index"]),
            int(home_event), den, float(position[0]), float(position[1]),
            float(rec["d_threat"]), int(rec["d_threat"] < world.THREAT_CONTACT_RADIUS)])))
        return rec

    world.advance_frame = observed
    try:
        yield
    finally:
        world.advance_frame = original


def rollout_with_trace(world, predict, fields, params, objects, seed, episode, frames):
    records = []
    with record_frames(world, seed, episode, records):
        result = world.rollout(predict, fields, params, n_frames=frames,
                               concentration=1.0, objects=objects, seed=seed + episode)
    summary = dict(training_seed=int(seed), episode=int(episode), c=1.0,
                   frames=int(frames), speed_ref=float(params.speed_ref))
    for key in ["foods", "foods_per_1k", "nights", "night_home_rate", "contact_frac",
                "wall_frac", "stall_frac", "diet_evenness", "den_evenness", "mean_speed"]:
        summary[key] = result[key]
    for i, count in enumerate(result["food_hits"]):
        summary[f"food_{i}_hits"] = int(count)
    for i, count in enumerate(result["den_hits"]):
        summary[f"den_{i}_hits"] = int(count)
    validate_episode(records, summary, params.circadian_period, params.day_fraction)
    return records, summary, result


def evenness(counts):
    """NumPy-only copy of world.evenness, used for independent checks/plotting."""
    c = np.asarray(counts, dtype=float)
    if c.sum() <= 0 or c.size < 2:
        return 0.0
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(c.size))


def validate_episode(records, summary, period, day_fraction):
    if len(records) != int(summary["frames"]) or not records:
        raise ValueError("Trace row count differs from episode frame count")
    a = {k: np.asarray([float(r[k]) for r in records]) for k in TRACE_COLUMNS}
    if not np.array_equal(a["frame"], np.arange(len(records))):
        raise ValueError("Missing or duplicate frame")
    for key in ["training_seed", "episode"]:
        if not np.all(a[key] == int(summary[key])):
            raise ValueError("Trace identity differs from summary")
    if float(summary["c"]) != 1.0:
        raise ValueError("Only c=1 is supported")
    if not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError("Non-finite trace value")
    w = np.column_stack([a["w_food"], a["w_threat"], a["w_shelter"]])
    if np.any(w < -1e-7) or np.any(w > 1 + 1e-7) or not np.allclose(w.sum(1), 1, atol=1e-6):
        raise ValueError("Mixture weights are invalid")
    for key in ["ate", "home_event", "is_night", "contact"]:
        if not np.isin(a[key], [0, 1]).all():
            raise ValueError("Invalid event flag")
    if not np.array_equal(a["clock"], np.arange(1, len(records) + 1)):
        raise ValueError("Clock is not the post-update clock")
    phase = (a["clock"].astype(int) % int(period)) / int(period)
    if not np.allclose(a["phase"], phase, atol=1e-12):
        raise ValueError("Phase and clock disagree")
    night = phase >= day_fraction
    if not np.array_equal(a["is_night"], night.astype(int)):
        raise ValueError("Night band is shifted relative to clock")
    food_idx, den_idx = a["ate_index"], a["den_index"]
    if not np.isin(food_idx, [-1, 0, 1, 2]).all() or not np.isin(den_idx, [-1, 0, 1]).all():
        raise ValueError("Invalid food/den index")
    if not np.array_equal(a["ate"].astype(bool), food_idx >= 0):
        raise ValueError("Eating flag and object index disagree")
    if not np.array_equal(a["home_event"].astype(bool), den_idx >= 0):
        raise ValueError("Home flag and object index disagree")
    night_id = np.cumsum(night & ~np.r_[False, night[:-1]])
    if np.any((a["home_event"] > 0) & ~night):
        raise ValueError("Home event outside the night")
    for ident in np.unique(night_id[night]):
        if a["home_event"][(night_id == ident) & night].sum() > 1:
            raise ValueError("More than one arrival in one night")
    foods = [int(np.sum(food_idx == i)) for i in range(3)]
    dens = [int(np.sum(den_idx == i)) for i in range(2)]
    for i, value in enumerate(foods):
        if value != int(summary[f"food_{i}_hits"]):
            raise ValueError("Per-food count mismatch")
    for i, value in enumerate(dens):
        if value != int(summary[f"den_{i}_hits"]):
            raise ValueError("Per-den count mismatch")
    nights = int(np.sum(night & ~np.r_[False, night[:-1]]))
    if sum(foods) != int(summary["foods"]) or nights != int(summary["nights"]):
        raise ValueError("Total event count mismatch")
    expected = dict(foods_per_1k=1000 * sum(foods) / len(records),
                    night_home_rate=sum(dens) / nights if nights else 0.,
                    contact_frac=float(a["contact"].mean()),
                    diet_evenness=evenness(foods), den_evenness=evenness(dens),
                    mean_speed=float(a["speed"].mean()))
    for key, value in expected.items():
        if not math.isclose(value, float(summary[key]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Recomputed {key} disagrees with summary")


def write_csv(path, columns, rows):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_seeds(rows):
    seeds = sorted({int(row["training_seed"]) for row in rows})
    per_seed = []
    for seed in seeds:
        selected = [row for row in rows if int(row["training_seed"]) == seed]
        per_seed.append(dict(training_seed=seed, episodes=len(selected),
                             **{key: float(np.mean([float(r[key]) for r in selected]))
                                for key in METRICS}))
    group = dict(n_seeds=len(seeds), episodes_per_seed=len(rows) // len(seeds))
    for key in METRICS:
        values = [row[key] for row in per_seed]
        group[key + "_mean"] = float(np.mean(values))
        group[key + "_sd"] = float(np.std(values, ddof=0))
    return per_seed, group
