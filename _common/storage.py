"""Shared storage, run names and RNG states. No model is built on import."""
import hashlib
import json
import os
import platform
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "_common/core"
SOURCE_COMMIT = "51fbdce03c2ff680b4230071f826caed26d021ba"
SEEDS = (7, 11, 23)


def checked_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Use a name of 1–80 letters, digits, underscores or hyphens")
    return name


def run_path(name):
    return ROOT / "01_behavior/runs" / checked_name(name)


def get_device(choice):
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU fallback")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def runtime(device):
    return dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                cuda=torch.version.cuda, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                invocation=list(sys.argv))


def capture_rng():
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                python=random.getstate(), numpy=dict(kind=kind, keys=keys.tolist(), pos=pos,
                                                   has_gauss=has_gauss, cached_gaussian=cached))


def restore_rng(state, device):
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from saved RNG state")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32), n["pos"],
                         n["has_gauss"], n["cached_gaussian"]))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def save_checkpoint(path, value):
    # Publish a complete model atomically on the WSL Linux filesystem.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=True) as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
        os.link(f.name, path)  # refuses to overwrite a saved model
