"""Deterministic state, portable JSON, and atomic experiment artifacts."""
from __future__ import annotations
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built() and backend.is_available())


def device_report() -> dict[str, bool]:
    return {"cuda_available": torch.cuda.is_available(), "mps_available": mps_available()}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)






def choose_device(name: str):
    """Explicit cuda/mps/cpu selection; auto prefers cuda, then mps, then cpu.

    An explicitly requested unavailable device raises instead of silently
    falling back, so logs and result files never claim a device that was not
    actually used.
    """
    report = device_report()
    if name == "auto":
        name = "cuda" if report["cuda_available"] else "mps" if report["mps_available"] else "cpu"
    if name.startswith("cuda") and not report["cuda_available"]:
        raise RuntimeError(
            "CUDA was explicitly requested but is unavailable. "
            f"Available: {report}. Use device=cpu or device=mps only when intended.")
    if name == "mps" and not report["mps_available"]:
        raise RuntimeError(
            "MPS was explicitly requested but torch.backends.mps.is_available() is false. "
            f"Available: {report}.")
    return torch.device(name)


def jsonl_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def jsonl_read(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
