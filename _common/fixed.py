"""One binary-field model cohort; three separate evaluation entry points."""
import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import storage as S
from trace_records import write_csv

sys.path[:0] = [str(S.SOURCE), str(S.SOURCE / "tmp")]
from neuromod import fields as F, protocol as P, world

SEEDS = S.SEEDS
C = (.05, .10, .20, .35, .50, .70, .85, 1., 1.2, 1.5, 2., 3., 4.)
W = tuple(i / 10 for i in range(11))
PAIRS = (("food", "threat"), ("threat", "shelter"), ("shelter", "food"))
DIR = {"intensity": "02_intensity", "interpolation": "03_interpolation", "lesion": "04_lesion"}
GROUPS = ("shared", "a-only", "b-only", "random")
EPOCHS, BATCH, LR = 4000, 1024, 3e-4
GRID, HIDDEN, SAMPLES, H = 61, 144, 64, .2


def model_root(name):
    return S.ROOT / DIR["intensity"] / "runs" / S.checked_name(name)


def evaluation_root(kind, name, evaluation):
    return S.ROOT / DIR[kind] / "runs" / S.checked_name(name) / "evaluations" / S.checked_name(evaluation)


def model():
    return P.build_network(HIDDEN, n_hidden=2, base_std=.8, kind="sample", t=SAMPLES,
                           crossing_h=H, in_dim=world.obs_dim())


def scene(device):
    objects = world.make_scripted_objects()
    positions = world.make_training_grid(GRID)
    tensor = lambda array: torch.as_tensor(array, dtype=torch.float32, device=device)
    obs = tensor(world.encode_observations(positions, objects))
    alphas = world.alpha_states(None)
    targets = {s: tensor(world.make_behavior_targets(positions, objects, alphas[s]))
               for s in world.STATES}
    graded = F.build_fields(world.CATEGORIES, HIDDEN, .8, .22, .15)
    nf = {k: (.8 * (v > 0).float()).to(device) for k, v in graded.items()}
    return objects, obs, targets, nf


def config():
    files = sorted(S.SOURCE.rglob("*.py")) + [Path(__file__), Path(S.__file__)]
    return dict(schema=1, source_commit=S.SOURCE_COMMIT, method="spatial_binary_pure",
                seeds=list(SEEDS), epochs=EPOCHS, batch_size=BATCH, optimizer="Adam",
                learning_rate=LR, grid_side=GRID, updates_per_epoch=3 * ((GRID**2 + BATCH-1)//BATCH),
                architecture=[32, HIDDEN, HIDDEN, 2], samples=SAMPLES, crossing_h=H,
                teacher="world.alpha_states(None)", alphas={k: v.tolist() for k,v in world.alpha_states(None).items()},
                field=dict(std=.8, radius=.28, width=.22, relative_cut=.15, binary=True),
                source_hashes={p.relative_to(S.ROOT).as_posix(): S.sha256(p) for p in files})


def load_saved(path, expected, device, seed):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["config"] != expected or record["seed"] != seed:
        raise ValueError(f"Model configuration or seed mismatch: {path}")
    verify_environment(record["environment"], device)
    return record


def verify_environment(previous, device):
    current = S.runtime(device)
    if any(previous[key] != current[key] for key in ("python", "numpy", "torch", "cuda", "device", "gpu")):
        raise ValueError("Use the same PyTorch/CUDA/device/GPU as the saved models")


def train(args):
    device = S.get_device(args.device)
    root = model_root(args.run_name)
    expected = config()
    metadata = root / "training.json"
    if not root.exists():
        (root / "checkpoints").mkdir(parents=True)
        S.write_json(metadata, dict(config=expected, environment=S.runtime(device)))
    else:
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if previous["config"] != expected:
            raise ValueError("Existing run has different training conditions; choose a new run name")
        verify_environment(previous["environment"], device)
    objects, obs, targets, nf = scene(device)
    state_fields = {s: nf[world.STATE_TO_FIELD[s]] for s in world.STATES}
    for seed in SEEDS:
        path = root / "checkpoints" / f"seed{seed}.pt"
        if path.exists():
            load_saved(path, expected, device, seed)
            print(f"seed {seed}: existing model validated; skipped", flush=True)
            continue
        start = time.monotonic()
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = model().to(device)
        print(f"seed {seed}: {EPOCHS} epochs, pure binary fields", flush=True)
        P.train(net, obs, targets, state_fields, world.STATES, 2, EPOCHS, LR, chunk=BATCH)
        record = dict(seed=seed, config=expected, environment=S.runtime(device),
                      model_state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                      rng=S.capture_rng(), seconds=time.monotonic() - start)
        S.save_checkpoint(path, record)
        S.write_json(root / f"training_seed{seed}.json", dict(seed=seed, seconds=record["seconds"],
                     updates=EPOCHS * expected["updates_per_epoch"], checkpoint_sha256=S.sha256(path)))
        print(f"Saved {path}", flush=True)


def intensity(net, obs, targets, nf, seed):
    # Original pure script evaluates final losses before the sweep; keep RNG order.
    state_fields = {s: nf[world.STATE_TO_FIELD[s]] for s in world.STATES}
    P.final_losses(net, obs, targets, state_fields, world.STATES, 2)
    rows = []
    for c in C:
        sep, signal, mse = P.capability(net, obs, targets, {k: c * v for k,v in nf.items()},
                                        world.CATEGORIES, world.STATES, world.STATE_TO_FIELD, 2)
        rows.append(dict(seed=seed, c=c, separation=sep, signal=signal, task_err=mse))
    return rows


@torch.no_grad()
def interpolation(net, obs, targets, nf, seed):
    by_field = {field: targets[state] for state, field in world.STATE_TO_FIELD.items()}
    rows = []
    for a, b in PAIRS:
        ta, tb = by_field[a], by_field[b]
        d = (tb - ta).flatten()
        denominator = d @ d
        if float(denominator) <= 0:
            raise ValueError("Targets coincide; lambda is undefined")
        for w in W:
            y = P.evaluate_vector_field(net, obs, (1-w)*nf[a] + w*nf[b], 2)
            lam = float(((y-ta).flatten() @ d) / denominator)
            err = float(((y - ((1-w)*ta + w*tb))**2).mean())
            rows.append(dict(seed=seed, a=a, b=b, w=w, implied_lambda=lam, err_vs_mixed_target=err))
    return rows


@torch.no_grad()
def per_error(net, nf, obs, targets):
    return {f: float(((P.evaluate_vector_field(net, obs, nf[f], 2)-targets[s])**2).mean())
            for s,f in world.STATE_TO_FIELD.items()}


def lesion(net, obs, targets, nf, seed):
    masks = {k: (v > 0).cpu().numpy() for k,v in nf.items()}
    rng = np.random.default_rng(seed + 100)
    base = per_error(net, nf, obs, targets)
    rows, details = [], []
    for a,b in itertools.combinations(world.CATEGORIES, 2):
        aa, bb = masks[a], masks[b]
        shared = np.flatnonzero(aa & bb)
        k = len(shared)
        groups = [[shared]]
        for pool in [aa & ~bb, bb & ~aa, aa | bb]:
            groups.append([rng.choice(np.flatnonzero(pool), size=k, replace=False) for _ in range(3)])
        for group, draws in zip(GROUPS, groups):
            errors = []
            for draw, units in enumerate(draws):
                damaged = copy.deepcopy(net)
                damaged_fields = {f: v.clone() for f,v in nf.items()}
                mask = np.zeros_like(aa)
                mask[units] = True
                F.kill_units(damaged, mask, damaged_fields, layer=0, h_dead=1e6)
                err = per_error(damaged, damaged_fields, obs, targets)
                errors.append(err)
                details.append(dict(seed=seed, a=a, b=b, group=group, draw=draw, n_units=k,
                                    units=" ".join(map(str, sorted(units))),
                                    base_a=base[a], base_b=base[b], damaged_a=err[a], damaged_b=err[b]))
            rows.append(dict(seed=seed, a=a, b=b, group=group, n_units=k,
                             delta_err_a=float(np.mean([e[a] for e in errors])-base[a]),
                             delta_err_b=float(np.mean([e[b] for e in errors])-base[b])))
    return rows, details


def evaluate(kind, args):
    device = S.get_device(args.device)
    expected = config()
    checkpoints = {s: model_root(args.run_name) / "checkpoints" / f"seed{s}.pt" for s in SEEDS}
    saved = {s: load_saved(p, expected, device, s) for s,p in checkpoints.items()}
    hashes = {str(s): S.sha256(p) for s,p in checkpoints.items()}
    contexts = {}
    if kind == "interpolation":
        previous = evaluation_root("intensity", args.run_name, args.intensity_evaluation)
        meta = json.loads((previous / "evaluation.json").read_text(encoding="utf-8"))
        if not meta["complete"] or meta["checkpoint_sha256"] != hashes or meta["config"] != expected:
            raise ValueError("Intensity run does not match these checkpoints")
        for seed in SEEDS:
            p = previous / f"after_sweep_seed{seed}.pt"
            if S.sha256(p) != meta["rng_after_sweep_sha256"][str(seed)]:
                raise ValueError("Intensity RNG file checksum mismatch")
            contexts[seed] = torch.load(p, map_location="cpu", weights_only=True)
    out = evaluation_root(kind, args.run_name, args.evaluation_name)
    out.mkdir(parents=True)
    objects, obs, targets, nf = scene(device)
    rows, draws, rng_hashes = [], [], {}
    for seed in SEEDS:
        net = model().to(device)
        net.load_state_dict(saved[seed]["model_state"], strict=True)
        net.eval()
        S.restore_rng(contexts.get(seed, saved[seed]["rng"]), device)
        if kind == "intensity":
            rows.extend(intensity(net, obs, targets, nf, seed))
            p = out / f"after_sweep_seed{seed}.pt"
            S.save_checkpoint(p, S.capture_rng())
            rng_hashes[str(seed)] = S.sha256(p)
        elif kind == "interpolation":
            rows.extend(interpolation(net, obs, targets, nf, seed))
        else:
            summary, raw = lesion(net, obs, targets, nf, seed)
            rows.extend(summary)
            draws.extend(raw)
        print(f"{kind}: seed {seed} evaluated", flush=True)
    if not all(np.isfinite(v) for row in rows + draws for v in row.values() if isinstance(v, float)):
        raise ValueError("Nonfinite result; evaluation not marked complete")
    write_csv(out / "results.csv", list(rows[0]), rows)
    if draws:
        write_csv(out / "lesion_draws.csv", list(draws[0]), draws)
    settings = {"intensity": dict(c=list(C)),
                "interpolation": dict(c=1, w=list(W), pairs=PAIRS, lambda_clip=False,
                                      intensity_evaluation=getattr(args, "intensity_evaluation", None)),
                "lesion": dict(c=1, pairs=list(itertools.combinations(world.CATEGORIES, 2)), groups=GROUPS,
                               layer_zero_based=0, dead_h=1e6, draws=[1,3,3,3], mask_seed="seed + 100")}[kind]
    S.write_json(out / "evaluation.json", dict(complete=True, kind=kind, config=expected, settings=settings,
                environment=S.runtime(device), checkpoint_paths={str(s): str(p) for s,p in checkpoints.items()},
                checkpoint_sha256=hashes, rng_after_sweep_sha256=rng_hashes, rows=len(rows),
                initial_objects={k: v.tolist() for k,v in objects.items()}))
    print(f"Saved {out}")


def main(kind):
    parser = argparse.ArgumentParser(description=f"Spatial binary-field {kind}")
    commands = parser.add_subparsers(dest="command", required=True)
    modes = {"evaluate": commands.add_parser("evaluate", help="Evaluate saved binary models; never train")}
    if kind == "intensity":
        modes["train"] = commands.add_parser("train", help="Train the shared three binary models")
    for p in modes.values():
        p.add_argument("--run-name", required=True)
        p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    modes["evaluate"].add_argument("--evaluation-name", default="main")
    if kind == "interpolation":
        modes["evaluate"].add_argument("--intensity-evaluation", default="main")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(kind, args)
