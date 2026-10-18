"""Evaluation at the annotation, image, animal and source levels.

Predictions must be probabilities, in the fixed order ot, nb, cb, ep, wc.
Missing annotations never become grade zero.  All metrics use only observed labels; predictions are never changed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

AU_NAMES = ("ot", "nb", "cb", "ep", "wc")
_GRADES = np.arange(3, dtype=np.float64)


def _probability(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name}: expected {shape}, received {out.shape}")
    if not np.isfinite(out).all() or np.any(out < -1e-7):
        raise ValueError(f"{name}: probabilities must be finite and nonnegative")
    if not np.allclose(out.sum(axis=-1), 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"{name}: probability rows must sum to one")
    return out


def _aligned(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    records, predictions = list(records), list(predictions)
    ids = [str(r["sample_id"]) for r in records]
    pids = [str(p["sample_id"]) for p in predictions]
    if len(set(ids)) != len(ids) or len(set(pids)) != len(pids):
        raise ValueError("Each sample_id must occur exactly once in evaluation")
    if set(ids) != set(pids):
        raise ValueError("Prediction IDs and evaluation record IDs must match exactly")
    lookup = dict(zip(pids, predictions))
    ordered = [lookup[i] for i in ids]
    q = np.zeros((len(records), 5, 3), dtype=np.float64)
    mask = np.zeros((len(records), 5), dtype=bool)
    prob = np.zeros_like(q)
    for i, (record, pred) in enumerate(zip(records, ordered)):
        m = np.asarray(record["mask"])
        if m.shape != (5,) or not np.isin(m, [0, 1, False, True]).all():
            raise ValueError(f"{ids[i]}: mask must contain five booleans")
        mask[i] = m.astype(bool)
        raw_q = np.asarray(record["q"], dtype=np.float64)
        if raw_q.shape != (5, 3):
            raise ValueError(f"{ids[i]}: q must have shape (5, 3)")
        if mask[i].any():
            q[i, mask[i]] = _probability(raw_q[mask[i]], (int(mask[i].sum()), 3), f"{ids[i]}.q")
        prob[i] = _probability(pred["prob"], (5, 3), f"{ids[i]}.prob")
    return records, ordered, q, mask, prob


def _source_keys(records: Sequence[Mapping[str, Any]]) -> list[str]:
    pairs = [(str(r.get("dataset", "unknown")), str(r.get("source", "unknown"))) for r in records]
    datasets_by_source: dict[str, set[str]] = defaultdict(set)
    for dataset, source in pairs:
        datasets_by_source[source].add(dataset)
    return [f"{d}::{s}" if len(datasets_by_source[s]) > 1 else s for d, s in pairs]


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def _au_image(errors: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float | None], float | None]:
    vals = [float(errors[mask[:, a], a].mean()) if mask[:, a].any() else None for a in range(5)]
    # A five-action metric is undefined when an action has no labels.
    macro = float(np.mean(vals)) if all(v is not None for v in vals) else None
    return dict(zip(AU_NAMES, vals)), macro


def _balanced(
    records: Sequence[Mapping[str, Any]], errors: np.ndarray, mask: np.ndarray,
) -> tuple[float | None, dict[str, dict[str, Any]]]:
    source_keys = _source_keys(records)
    sources: dict[str, list[int]] = defaultdict(list)
    for i, source in enumerate(source_keys):
        sources[source].append(i)
    details: dict[str, dict[str, Any]] = {}
    for source in sorted(sources):
        ix = np.asarray(sources[source], dtype=int)
        local_mask = mask[ix]
        image_au, image_macro = _au_image(errors[ix], local_mask)
        animals: dict[str, list[int]] = defaultdict(list)
        unknown = False
        for i in ix:
            uid = records[i].get("animal_uid")
            if not mask[i].any():
                continue
            if uid is None or not str(uid).strip():
                unknown = True
            else:
                animals[str(uid)].append(int(i))
        action_values: list[float | None] = []
        action_animal_counts: dict[str, int] = {}
        for a, au in enumerate(AU_NAMES):
            animal_values = []
            for indices in animals.values():
                valid = [i for i in indices if mask[i, a]]
                if valid:
                    animal_values.append(float(errors[valid, a].mean()))
            action_animal_counts[au] = len(animal_values)
            action_values.append(float(np.mean(animal_values)) if animal_values and not unknown else None)
        macro = float(np.mean(action_values)) if all(x is not None for x in action_values) else None
        details[source] = {
            "primary_au_mae": macro,
            "animal_au_mae": dict(zip(AU_NAMES, action_values)),
            "image_au_mae": image_macro,
            "au_image_mae": image_au,
            "n_images": len(ix),
            "n_labeled_images": int(local_mask.any(axis=1).sum()),
            "n_animals": len(animals),
            "animal_identity_verified": not unknown,
            "au_image_counts": dict(zip(AU_NAMES, local_mask.sum(axis=0).astype(int).tolist())),
            "au_animal_counts": action_animal_counts,
            "undefined_reason": "unknown_animal_identity" if unknown else ("missing_action_labels" if macro is None else None),
        }
    values = [item["primary_au_mae"] for item in details.values()]
    primary = float(np.mean(values)) if values and all(v is not None for v in values) else None
    return primary, details


def _correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float | None:
    if len(x) < 2:
        return None
    if rank:
        x, y = rankdata(x), rankdata(y)
    # A constant score is not evidence for zero correlation.
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else None


def _scalar_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if len(target) == 0:
        return {"count": 0, "mae": None, "rmse": None, "pearson": None, "spearman": None}
    error = predicted - target
    return {
        "count": int(len(target)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson": _correlation(target, predicted),
        "spearman": _correlation(target, predicted, rank=True),
    }


def evaluate(
    records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute metrics without filling missing ratings or inventing animal IDs.

    ``primary_au_mae`` averages images within animal, animals within AU, the
    five AUs within source, then sources equally. Unknown animal identities or
    an entirely unlabeled AU make that metric undefined (JSON ``null``).
    ``image_au_mae`` is the explicitly separate image-weighted AU macro metric.

    ``checkpoint_available_au_mae`` is solely for checkpoint selection when
    an internal low-budget validation source lacks an entire action. It keeps
    equal animal/source weighting, but averages only the available actions in
    each source. Its coverage is explicit; it must never replace the strict
    five-action metric in paper test results or main experiment summaries.
    """
    records, ordered, q, mask, prob = _aligned(records, predictions)
    target, predicted = q @ _GRADES, prob @ _GRADES
    errors = np.abs(predicted - target)
    primary, per_source = _balanced(records, errors, mask)
    checkpoint_values, checkpoint_coverage = [], {}
    for source, source_metrics in per_source.items():
        available = [au for au in AU_NAMES if source_metrics["animal_au_mae"][au] is not None]
        checkpoint_coverage[source] = {
            "available_aus": available,
            "n_available_aus": len(available),
            "au_image_counts": source_metrics["au_image_counts"],
            "au_animal_counts": source_metrics["au_animal_counts"],
            "animal_identity_verified": source_metrics["animal_identity_verified"],
        }
        checkpoint_values.append(
            float(np.mean([source_metrics["animal_au_mae"][au] for au in available]))
            if available and source_metrics["animal_identity_verified"] else None
        )
    checkpoint_mae = float(np.mean(checkpoint_values)) if checkpoint_values and all(v is not None for v in checkpoint_values) else None
    image_au, image_macro = _au_image(errors, mask)
    complete = mask.all(axis=1)
    partial = mask.any(axis=1) & ~complete
    available_target = (target[partial] * mask[partial]).sum(axis=1) / np.maximum(mask[partial].sum(axis=1), 1)
    available_pred = (predicted[partial] * mask[partial]).sum(axis=1) / np.maximum(mask[partial].sum(axis=1), 1)
    identities = {
        (str(r.get("dataset", "unknown")), str(r.get("source", "unknown")), str(r["animal_uid"]))
        for r, m in zip(records, mask) if m.any() and r.get("animal_uid") is not None and str(r["animal_uid"]).strip()
    }
    result = {
        "primary_au_mae": primary,
        "primary_mae": primary,
        "checkpoint_available_au_mae": checkpoint_mae,
        "checkpoint_coverage": checkpoint_coverage,
        "primary_definition": "equal-source/equal-AU/equal-animal/image-mean absolute expected-grade error",
        "image_au_mae": image_macro,
        "au_image_mae": image_au,
        "per_source": per_source,
        "n_images": len(records),
        "n_labeled_images": int(mask.any(axis=1).sum()),
        "n_complete5_images": int(complete.sum()),
        "n_valid_au_labels": int(mask.sum()),
        "au_image_counts": dict(zip(AU_NAMES, mask.sum(axis=0).astype(int).tolist())),
        "n_animals": len(identities),
        "mgs5": _scalar_metrics(target[complete].mean(axis=1), predicted[complete].mean(axis=1)),
        "partial_available_au_mean": _scalar_metrics(available_target, available_pred),
    }
    return result

