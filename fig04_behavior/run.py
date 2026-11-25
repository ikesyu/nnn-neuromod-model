"""Behavior experiment (region method): explicit train and evaluate commands.

Training and the closed loop come from _common/core (tolerance.train_base,
ethogram.one_life, world.rollout); this script adds only checkpoint and
trace storage. The representative one_life precedes the pooled episodes.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import storage
sys.path[:0] = [str(storage.SOURCE), str(storage.SOURCE / "tmp")]
from neuromod import fields as F, protocol as P, world
import neuromod_tolerance as T
from trace_records import (record_frames, rollout_with_trace, validate_episode,
                           write_csv, aggregate_seeds, TRACE_COLUMNS, SUMMARY_COLUMNS, METRICS)

ROOT, SOURCE = storage.ROOT, storage.SOURCE
_spec = importlib.util.spec_from_file_location(
    "original_region_ethogram", SOURCE / "tmp/neuromod_ethogram.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
SEEDS = [7, 11, 23]


def fields(device):
    return {key: value.to(device) for key, value in
            F.build_fields(world.CATEGORIES, 144, .8, .22, .15).items()}


def new_model(device):
    return P.build_network(144, n_hidden=2, base_std=.8, kind="sample", t=64,
                           crossing_h=.2, in_dim=world.obs_dim()).to(device)


def fingerprint():
    files = sorted(SOURCE.rglob("*.py")) + [Path(__file__),
             ROOT / "_common/storage.py", ROOT / "_common/trace_records.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def configuration(steps):
    return dict(schema=1, method="region", source_commit="51fbdce03c2ff680b4230071f826caed26d021ba",
        seeds=SEEDS, train_steps=int(steps), batch_size=512, learning_rate=3e-4,
        position_pool=16384, training="original neuromod_tolerance.train_base",
        architecture=[32, 144, 144, 2], samples=64, crossing_h=.2,
        noise_distribution="Gaussian with neuron-specific spatial std",
        field_parameters=dict(base_std=.8, width=.22, theta_cut=.15, ring_radius=.28),
        c=1., episodes=6, frames=1260, representative="original one_life, seed 7, torch seed 1007",
        optimizer="Adam", teacher_alphas={k: v.tolist() for k,v in world.alpha_states(None).items()},
        source_hashes=fingerprint(), standard_training=steps == 40000, test_only=steps != 40000)


def train(args):
    if args.train_steps <= 0:
        raise ValueError("Training steps must be positive")
    device = storage.get_device(args.device)
    out = storage.run_path(args.run_name)
    config = configuration(args.train_steps)
    path = out / "metadata/run_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Run belongs to a different method/settings/code; use a new name")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Nonempty run directory has no recognized region config")
        (out / "metadata").mkdir(parents=True)
        (out / "checkpoints").mkdir()
        storage.write_json(path, config)
        storage.write_json(out / "metadata/training_environment.json", storage.runtime(device))
    objects, nf = world.make_scripted_objects(), fields(device)
    original_args = SimpleNamespace(hidden_dim=144, samples=64, crossing_h=.2,
                                   train_steps=args.train_steps)
    for seed in SEEDS:
        checkpoint = out / "checkpoints" / f"seed{seed}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["config"] != config or saved["training_seed"] != seed:
                raise ValueError("Wrong checkpoint; no overwrite permitted")
            print(f"seed {seed}: saved region model exists; skipping training", flush=True)
            continue
        start = time.monotonic()
        print(f"seed {seed}: region training {args.train_steps} updates; original helper suppresses step logs", flush=True)
        net = T.train_base(seed, original_args, objects, nf, device)
        storage.save_checkpoint(checkpoint, dict(schema=1, method="region", training_seed=seed,
            config=config, environment=storage.runtime(device), test_only=config["test_only"],
            model_state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            rng_after_training=storage.capture_rng()))
        storage.write_json(out / "metadata" / f"training_seed{seed}.json",
                           dict(training_seed=seed, seconds=time.monotonic() - start))
        print(f"saved {checkpoint}", flush=True)
    print("Region models saved. Evaluation and plotting have NOT been run.")


def representative(net, nf, objects, frames, device):
    """Observe the original one_life, without adding predictions/random draws."""
    records, context = [], {}
    original_speed_ref = E.T.speed_ref_of

    def observed_speed_ref(*args, **kwargs):
        value = original_speed_ref(*args, **kwargs)
        context["speed_ref"] = value
        context["rng_after_speed_ref"] = storage.capture_rng()
        return value

    E.T.speed_ref_of = observed_speed_ref
    try:
        # -1 explicitly identifies the separate life, NOT pooled episode zero.
        with record_frames(world, 7, -1, records):
            W, speed, ate, home, phase, day_fraction = E.one_life(
                net, nf, objects, 7, frames, device)
    finally:
        E.T.speed_ref_of = original_speed_ref
    if not (np.array_equal(W, [[r[k] for k in ["w_food", "w_threat", "w_shelter"]] for r in records])
            and np.array_equal(speed, [r["speed"] for r in records])
            and np.array_equal(ate, [r["ate"] for r in records])
            and np.array_equal(home, [r["home_event"] for r in records])
            and np.array_equal(phase, [r["phase"] for r in records])):
        raise ValueError("Recorder differs from the original representative life")
    food = [sum(r["ate_index"] == i for r in records) for i in range(3)]
    den = [sum(r["den_index"] == i for r in records) for i in range(2)]
    night = np.array([bool(r["is_night"]) for r in records])
    nights = int(np.sum(night & ~np.r_[False, night[:-1]]))
    summary = dict(training_seed=7, episode=-1, c=1., frames=frames, foods=sum(food),
        foods_per_1k=1000 * sum(food) / frames, nights=nights,
        night_home_rate=sum(den) / nights if nights else 0.,
        contact_frac=float(np.mean([r["contact"] for r in records])),
        diet_evenness=world.evenness(food), den_evenness=world.evenness(den),
        mean_speed=float(np.mean(speed)), speed_ref=float(context["speed_ref"]))
    summary.update({f"food_{i}_hits": v for i, v in enumerate(food)})
    summary.update({f"den_{i}_hits": v for i, v in enumerate(den)})
    params = world.LoopParams(speed_ref=context["speed_ref"], threat_seed=7)
    validate_episode(records, summary, params.circadian_period, day_fraction)
    return records, summary, context, dataclasses.asdict(params)


def evaluate(args):
    training_run = storage.run_path(args.run_name)
    config = json.loads((training_run / "metadata/run_config.json").read_text(encoding="utf-8"))
    if config != configuration(config["train_steps"]):
        raise ValueError("Not a matching region run; no experiment started")
    checkpoints = [training_run / "checkpoints" / f"seed{s}.pt" for s in SEEDS]
    if not all(p.is_file() for p in checkpoints):
        raise FileNotFoundError("Three saved region models are required; no training started")
    storage.checked_name(args.evaluation_name)
    out = training_run / "evaluations" / args.evaluation_name
    if out.exists():
        raise FileExistsError("Evaluation exists; choose a new evaluation name")
    device = storage.get_device(args.device)
    # Check all models before creating an evaluation directory.
    for seed, path in zip(SEEDS, checkpoints):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["training_seed"] != seed or saved["method"] != "region":
            raise ValueError("Wrong checkpoint method/seed/config")
        if saved["environment"]["device"] != str(device):
            raise ValueError("Use the same device as training")
    (out / "data/traces").mkdir(parents=True)
    (out / "metadata").mkdir()
    storage.write_json(out / "metadata/run_config.json", config)
    storage.write_json(out / "metadata/evaluation_environment.json", storage.runtime(device))
    objects, nf = world.make_scripted_objects(), fields(device)
    storage.write_json(out / "metadata/initial_objects.json", {k: v.tolist() for k, v in objects.items()})
    rows, params_by_seed, params_by_episode = [], {}, {}
    representative_params = None
    for seed, path in zip(SEEDS, checkpoints):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        net = new_model(device)
        net.load_state_dict(saved["model_state"], strict=True)
        net.eval()
        storage.restore_rng(saved["rng_after_training"], device)
        if seed == 7:
            records, summary, context, representative_params = representative(
                net, nf, objects, config["frames"], device)
            write_csv(out / "data/representative_trace.csv", TRACE_COLUMNS, records)
            storage.write_json(out / "data/representative_summary.json", summary)
            storage.save_checkpoint(out / "metadata/representative_context.pt", context)
        # For seed 7 this is deliberately AFTER the separate representative life,
        # matching the original ethogram/diversity script's order.
        predict = T.make_predict(net, device)
        speed_ref = T.speed_ref_of(predict, objects, nf)
        if not math.isfinite(speed_ref) or speed_ref <= 0:
            raise ValueError("Invalid speed_ref")
        storage.save_checkpoint(out / "metadata" / f"seed{seed}_evaluation_context.pt",
            dict(speed_ref=speed_ref, rng_after_speed_ref=storage.capture_rng()))
        params_by_seed[str(seed)] = dataclasses.asdict(world.LoopParams(speed_ref=speed_ref, threat_seed=seed))
        for episode in range(config["episodes"]):
            torch.manual_seed(1000 + episode)
            params = world.LoopParams(speed_ref=speed_ref, threat_seed=seed + episode)
            params_by_episode[f"{seed}:{episode}"] = dataclasses.asdict(params)
            records, summary, _ = rollout_with_trace(world, predict, nf, params,
                objects, seed, episode, config["frames"])
            write_csv(out / "data/traces" / f"seed{seed}_episode{episode}.csv", TRACE_COLUMNS, records)
            rows.append(summary)
            print(f"region seed {seed}, episode {episode}: {len(records)} frames recorded", flush=True)
    write_csv(out / "data/episodes.csv", SUMMARY_COLUMNS, rows)
    per_seed, group = aggregate_seeds(rows)
    write_csv(out / "data/seed_summary.csv", ["training_seed", "episodes"] + METRICS, per_seed)
    write_csv(out / "data/group_summary.csv", list(group), [group])
    storage.write_json(out / "metadata/evaluation.json", dict(complete=True, method="region", c=1.,
        seeds=SEEDS, episodes=config["episodes"], frames=config["frames"], test_only=config["test_only"],
        representative=dict(training_seed=7, kind="original_one_life", torch_seed=1007, environment_seed=7),
        representative_loop_params=representative_params, loop_params=params_by_seed,
        loop_params_by_episode=params_by_episode, source_hashes=fingerprint(),
        torch_episode_seed="1000 + episode", environment_seed="training_seed + episode",
        representative_in_pooled_counts=False,
        checkpoint_sha256={str(s): storage.sha256(p) for s,p in zip(SEEDS,checkpoints)},
        clock_position_phase="after advance_frame; frame starts at 0, clock at 1",
        speed="smoothed velocity command magnitude, not displacement speed"))
    print("Region evaluation complete. Plotting has NOT been run.")


def main():
    parser = argparse.ArgumentParser(description="Behavior experiment: train or evaluate the region models")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and save the region models only")
    evaluation = commands.add_parser("evaluate", help="Record saved region models only; never train")
    for command in [training, evaluation]:
        command.add_argument("--run-name", required=True)
        command.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    training.add_argument("--train-steps", type=int, default=40000)
    evaluation.add_argument("--evaluation-name", default="c1_run01")
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
