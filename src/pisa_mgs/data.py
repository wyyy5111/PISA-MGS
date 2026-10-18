"""Portable JSONL manifests: empirical ordinal labels and verified face crops."""
from __future__ import annotations
from pathlib import PurePosixPath
import numpy as np
from .utils import jsonl_read

AU_NAMES = ("ot", "nb", "cb", "ep", "wc")


def empirical_labels(ratings):
    """Build q/mask from lists of valid 0/1/2 grades in the fixed five-AU order.

    Unrateable (9), missing, and invalid values must be removed before calling.
    This helper rejects them instead of silently changing the scoring scale.
    """
    if len(ratings) != 5:
        raise ValueError("Provide one rating list for each of the five AUs")
    q, mask = [], []
    for values in ratings:
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v not in (0, 1, 2) for v in values):
            raise ValueError("Only valid integer 0/1/2 ratings may form soft labels")
        counts = np.bincount(values, minlength=3).astype(float) if values else np.zeros(3)
        q.append((counts / len(values)).tolist() if values else [0.0, 0.0, 0.0])
        mask.append(bool(values))
    return q, mask


def load_manifest(path, *, require_labels=False):
    rows = jsonl_read(path)
    ids = [r["sample_id"] for r in rows]
    if not rows or any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain unique, nonempty sample IDs")
    for row in rows:
        path = PurePosixPath(row["image_path"])
        if path.is_absolute() or ".." in path.parts or "\\" in str(path):
            raise ValueError("image_path must be relative to its supplied data root")
        if row.get("roi_verified") is not True:
            raise ValueError("Every input requires a reviewed face ROI or reviewed full-face crop")
        if "q" not in row and "mask" not in row:
            row["q"], row["mask"] = [[0.0] * 3 for _ in range(5)], [False] * 5
        q, mask = np.asarray(row["q"], dtype=float), np.asarray(row["mask"])
        if q.shape != (5, 3) or mask.shape != (5,) or not all(isinstance(m, bool) for m in row["mask"]):
            raise ValueError("q and mask must have shapes [5,3] and [5 booleans]")
        valid = q[mask]
        if valid.size and (not np.isfinite(valid).all() or (valid < 0).any()
                           or not np.allclose(valid.sum(-1), 1, atol=1e-5)):
            raise ValueError("Observed q rows must be finite probability distributions")
        # Missing placeholders are never interpreted as grade zero.
        row["q"] = np.where(mask[:, None], q, 0.0).tolist()
        if require_labels and (not mask.any() or not row.get("source") or not row.get("animal_uid")):
            raise ValueError("Source records need observed labels, source, and verified animal_uid")
    return rows


def assert_source_split(records, split):
    lookup = {r["sample_id"]: r for r in records}
    train, val = split["train"], split["val"]
    if not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Source train/val lists must be nonempty and unique")
    if not (set(train) | set(val)) <= set(lookup) or set(train) & set(val):
        raise ValueError("Unknown or overlapping source train/val IDs")
    animals = [{lookup[i].get("animal_uid") for i in group} for group in (train, val)]
    if any(None in a or "" in a for a in animals) or animals[0] & animals[1]:
        raise ValueError("Train/val need verified, disjoint animal identities")
    if any(not any(lookup[i]["mask"]) for i in train + val):
        raise ValueError("Every selected source image needs at least one observed AU")
