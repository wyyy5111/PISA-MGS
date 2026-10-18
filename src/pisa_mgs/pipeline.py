"""Portable extraction, source-only training, and unlabeled-pool transfer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .data import load_manifest, assert_source_split
from .features import extract_features, feature_signature
from .metrics import evaluate
from .models import AUHead
from .training import train_protocol, train_convnext
from .transfer import transfer_predict, probability_ensemble
from .utils import atomic_json, atomic_torch_save, canonical_hash, jsonl_write, load_json


def _stack(bank, ids):
    values = [bank.get(i, 0) for i in ids]
    return torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values])


def run(source_manifest, source_data_root, target_manifest, target_data_root, split_path,
        config_path, work_dir, extract_device="auto", allow_download=False):
    config, split = load_json(config_path), load_json(split_path)
    if config.get("schema_version") != 1 or not config.get("seeds"):
        raise ValueError("Unsupported config schema or missing seeds")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seed list contains duplicates")
    source, target = load_manifest(source_manifest), load_manifest(target_manifest)
    assert_source_split(source, split)
    lookup = {r["sample_id"]: r for r in source}
    train_ids, val_ids = split["train"], split["val"]
    target_ids = sorted(r["sample_id"] for r in target)
    target_lookup = {r["sample_id"]: r for r in target}
    if (set(train_ids) | set(val_ids)) & set(target_ids):
        raise ValueError("Source and target sample IDs must be distinct")
    source_animals = {lookup[i]["animal_uid"] for i in train_ids + val_ids}
    target_animals = {r["animal_uid"] for r in target if r.get("animal_uid")}
    if source_animals & target_animals:
        raise ValueError("Source/target known animal identities overlap; use verified globally namespaced IDs")
    out = Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = [lookup[i]["q"] for i in train_ids]
    mask = [lookup[i]["mask"] for i in train_ids]
    member_predictions = []
    summary = {"schema_version": 1, "source_train": len(train_ids), "source_val": len(val_ids),
               "target_pool": len(target_ids), "target_labels_used_for_fitting": False,
               "seeds": config["seeds"], "members": {}}
    for tag, spec in config["members"].items():
        if tag not in ("dinov2", "aimv2", "convnext"):
            raise ValueError("Only the three released ensemble members are supported")
        enc = dict(spec["encoder"])
        enc["allow_download"] = bool(allow_download)
        extraction_config = {"device": extract_device, "encoder": enc}
        # Roots and caches are separate: no dependence on an old, already-filled source cache.
        source_bank = extract_features(source, source_data_root, out / "features" / tag / "source",
                                       extraction_config, train_ids + val_ids, train_ids=train_ids)
        target_bank = extract_features(target, target_data_root, out / "features" / tag / "target",
                                       extraction_config, target_ids)
        source_p, source_g = _stack(source_bank, train_ids)
        target_p, target_g = _stack(target_bank, target_ids)
        seed_predictions = []
        for seed in config["seeds"]:
            checkpoint = out / "heads" / f"{tag}_s{seed}.pt"
            signature = canonical_hash({"seed": seed, "member": spec,
                         "train": [lookup[i] for i in train_ids], "val": [lookup[i] for i in val_ids],
                         "features": feature_signature(source_bank, train_ids + val_ids)})
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved["signature"] != signature:
                    raise ValueError("Cached head differs from the source data/recipe; choose a fresh work directory")
                head = AUHead(source_bank.patch_dim, hidden=128, dropout=0.1)
                head.load_state_dict(saved["model"])
                head.eval()
            else:
                if spec["loop"] == "protocol":
                    head, info = train_protocol(source, source_bank, train_ids, val_ids, spec["training"], seed)
                elif spec["loop"] == "constant":
                    head, _ = train_convnext(source, source_bank, train_ids, val_ids,
                                             spec["training"]["lr"], seed,
                                             epochs=spec["training"]["max_epochs"])
                    info = {"loop": "historical constant-lr ConvNeXt path"}
                else:
                    raise ValueError("Unknown training loop")
                atomic_torch_save(checkpoint, {"signature": signature, "model": head.state_dict()})
                atomic_json(out / "heads" / f"{tag}_s{seed}.json", info)
            probs, diagnostics = transfer_predict(head, source_p, source_g, target_p, target_g,
                      q, mask, spec["transform"], config.get("pisa_min_shift_norm_sq", 1e-12))
            seed_predictions.append(probs)
            atomic_json(out / "calibration" / f"{tag}_s{seed}.json", diagnostics)
            print(f"{tag} seed {seed}: prediction complete", flush=True)
        member = probability_ensemble(seed_predictions)
        member_predictions.append(member)
        jsonl_write(out / f"{tag}_predictions.jsonl",
                    [{"sample_id": i, "prob": p.tolist()} for i, p in zip(target_ids, member)])
        summary["members"][tag] = {"encoder": enc["name"], "transform": spec["transform"],
                                   "preprocess": enc["preprocess"], "loop": spec["loop"]}
    ensemble = probability_ensemble(member_predictions)
    predictions = [{"sample_id": i, "prob": p.tolist(),
                    "au_scores": (p @ np.arange(3)).tolist(),
                    "mgs5": float((p @ np.arange(3)).mean())} for i, p in zip(target_ids, ensemble)]
    jsonl_write(out / "predictions.jsonl", predictions)
    # This block runs only after the predictions are fixed; target labels are optional.
    if any(any(r["mask"]) for r in target):
        summary["evaluation"] = evaluate([target_lookup[i] for i in target_ids], predictions)
    atomic_json(out / "summary.json", summary)
    return summary
