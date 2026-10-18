"""Blinded face ROI preparation and versioned, frozen-encoder feature caches."""
from __future__ import annotations
import hashlib
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch

from .utils import atomic_json, canonical_hash, load_json, sha256_file

PREPROCESS_VERSION = "gray-letterbox-224-v1"
VIEWS = ("original", "horizontal_flip", "brightness_contrast_a", "brightness_contrast_b")


def face_image(record, data_root):
    """Require an explicit verified face region, retaining ears and whiskers."""
    if not record.get("roi_verified", False):
        raise ValueError(f"Unverified face ROI: {record['sample_id']}; import blinded ROI CSV first")
    path = (Path(data_root) / record["image_path"]).resolve()
    root = Path(data_root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path leaves the supplied data root")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        roi = record.get("roi")
        if roi is not None:
            x, y, w, h = map(float, roi)
            if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
                raise ValueError("Nonpositive/invalid ROI")
            if x < 0 or y < 0 or x + w > im.width or y + h > im.height:
                raise ValueError(f"ROI outside image: {record['sample_id']}")
            im = im.crop((int(math.floor(x)), int(math.floor(y)),
                          int(math.ceil(x + w)), int(math.ceil(y + h))))
        return im.copy()


def preprocess(record, data_root, view=0, augmentation_seed=101, image_size=224, mode="standard"):
    if not 0 <= view < len(VIEWS):
        raise ValueError("Unknown cached training view")
    im = face_image(record, data_root).convert("L")
    if mode == "equalize":
        im = ImageOps.equalize(im).convert("RGB")
        # Historical equalize_preprocess mirrored every nonzero cached view.
        if view:
            im = ImageOps.mirror(im)
    elif mode != "standard":
        raise ValueError("Preprocessing must be standard or equalize")
    im = im.convert("RGB")
    if mode == "standard" and view == 1:
        im = ImageOps.mirror(im)
    elif mode == "standard" and view >= 2:
        h = hashlib.sha256(f"{augmentation_seed}:{record['sample_id']}:{view}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        # Two deterministic opposite mild views; no crop or erasing of actions.
        sign = -1 if view == 2 else 1
        im = ImageEnhance.Brightness(im).enhance(1 + sign * float(rng.uniform(0.02, 0.10)))
        im = ImageEnhance.Contrast(im).enhance(1 + sign * float(rng.uniform(0.01, 0.05)))
    ratio = min(image_size / im.width, image_size / im.height)
    resized = im.resize((max(1, round(im.width * ratio)), max(1, round(im.height * ratio))),
                        Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    canvas.paste(resized, ((image_size - resized.width) // 2, (image_size - resized.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    # Fixed ImageNet normalization, never fitted on any dataset or split.
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / \
          np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return torch.from_numpy(arr.copy())


def encoder_spec(config):
    c = config["encoder"]
    return {"name": c["name"], "hub_ref": c["hub_ref"],
            "checkpoint_sha256": sha256_file(c["checkpoint"]) if c.get("checkpoint") else None,
            "preprocess": "gray-equalize-letterbox-224-v1" if c["preprocess"] == "equalize" else PREPROCESS_VERSION, "size": c["image_size"],
            "augmentation_seed": c["augmentation_seed"], "views": VIEWS}


def feature_signature(bank, ids):
    if not set(ids) <= bank.allowed_ids:
        raise ValueError("Feature signature outside assigned partition")
    return {"spec": bank.index["spec"],
            "samples": {sid: bank.index["samples"][sid]["key"] for sid in sorted(set(ids))}}


class FeatureBank:
    """Feature access is restricted to an explicitly assigned ID set."""
    def __init__(self, cache_dir, allowed_ids=None, expected_spec=None, *,
                 memory_budget_bytes=256 * 1024 * 1024, _memory_cache=None):
        self.root = Path(cache_dir)
        self.index = load_json(self.root / "index.json")
        if expected_spec is not None and self.index["spec"] != expected_spec:
            # JSON changes tuples to lists, compare canonical representation instead.
            if canonical_hash(self.index["spec"]) != canonical_hash(expected_spec):
                raise ValueError("Feature cache encoder/preprocessing configuration mismatch")
        self.allowed_ids = set(allowed_ids) if allowed_ids is not None else set(self.index["samples"])
        missing = self.allowed_ids - set(self.index["samples"])
        if missing:
            raise ValueError(f"Missing cached features for {len(missing)} samples")
        self.patch_dim = self.index["patch_dim"]
        self.global_dim = self.index["global_dim"]
        self.patch_count = self.index["patch_count"]
        if memory_budget_bytes < 0:
            raise ValueError("Feature memory budget cannot be negative")
        self._memory_cache = _memory_cache if _memory_cache is not None else {
            "items": OrderedDict(), "nbytes": 0, "limit": memory_budget_bytes}

    def restrict(self, ids):
        if not set(ids) <= self.allowed_ids:
            raise ValueError("Attempt to widen feature access beyond assigned partition")
        return FeatureBank(self.root, ids, self.index["spec"], _memory_cache=self._memory_cache)

    def views(self, sample_id):
        if sample_id not in self.allowed_ids:
            raise ValueError("Feature access outside assigned partition")
        return self.index["samples"][sample_id]["views"]

    def validate_records(self, records, receipt, ids):
        lookup = {r["sample_id"]: r for r in records}
        for sid in ids:
            if sid not in self.allowed_ids:
                raise ValueError("Validation of cache outside assigned partition")
            row, entry = lookup[sid], self.index["samples"][sid]
            if not row.get("roi_verified") or not entry.get("roi_verified"):
                raise ValueError(f"Feature cache has unverified face input: {sid}")
            if row.get("roi") != entry.get("roi"):
                raise ValueError(f"ROI changed after feature extraction: {sid}")
            if entry["image_sha256"] != receipt["hashes"][sid]["file_sha256"]:
                raise ValueError(f"Image changed after feature extraction: {sid}")

    def get(self, sample_id, view=0):
        if sample_id not in self.allowed_ids:
            raise ValueError(f"Feature access outside assigned partition: {sample_id}")
        entry = self.index["samples"][sample_id]
        if view >= entry["views"] or view < 0:
            raise ValueError("Training augmentation not extracted; run cache with train IDs")
        path = self.root / entry["file"]
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns, entry["key"], entry["views"])
        cache = self._memory_cache
        cache_key = str(path.resolve())
        previous = cache["items"].pop(cache_key, None)
        if previous is not None:
            cache["nbytes"] -= previous[1].nbytes + previous[2].nbytes
        if previous is not None and previous[0] == fingerprint:
            p, g = previous[1:]
        else:
            with np.load(path, allow_pickle=False) as f:
                if str(f["key"].item()) != entry["key"]:
                    raise ValueError("Corrupted or misplaced feature cache shard")
                p, g = f["patches"], f["global"]
                if p.shape != (entry["views"], self.patch_count, self.patch_dim) or g.shape != (entry["views"], self.global_dim):
                    raise ValueError("Feature cache shard has incorrect dimensions")
        nbytes = p.nbytes + g.nbytes
        if nbytes <= cache["limit"]:
            while cache["items"] and cache["nbytes"] + nbytes > cache["limit"]:
                _, evicted = cache["items"].popitem(last=False)
                cache["nbytes"] -= evicted[1].nbytes + evicted[2].nbytes
            cache["items"][cache_key] = (fingerprint, p, g)
            cache["nbytes"] += nbytes
        if not np.isfinite(p[view]).all() or not np.isfinite(g[view]).all():
            raise ValueError("Feature cache shard contains nonfinite values")
        return (torch.from_numpy(p[view].astype(np.float32)), torch.from_numpy(g[view].astype(np.float32)))


@torch.no_grad()
def extract_features(records, data_root, cache_dir, config, ids, train_ids=(), encoder=None):
    """Extract only requested images. Frozen target features are used for inference only."""
    from .encoders import FrozenEncoder
    from .utils import choose_device
    records_by_id = {r["sample_id"]: r for r in records}
    ids, train_ids = sorted(set(ids)), set(train_ids)
    if not train_ids <= set(ids):
        raise ValueError("Training cache IDs must be a subset of requested IDs")
    if any(i not in records_by_id for i in ids):
        raise ValueError("Unknown cache ID")
    c = config["encoder"]
    spec = encoder_spec(config)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    test_only = encoder is not None and (not isinstance(encoder, FrozenEncoder) or getattr(encoder,"test_only",True))
    index = load_json(index_path) if index_path.exists() else {"schema_version": 1, "spec": spec, "samples": {},"test_only":test_only}
    if index.get("test_only",False) != test_only:
        raise ValueError("Never mix structural-test features with pretrained production features")
    if canonical_hash(index["spec"]) != canonical_hash(spec):
        raise ValueError("Use a separate cache directory for a different encoder/preprocessing")
    device = choose_device(config["device"])
    pending = []
    for sid in ids:
        r = records_by_id[sid]
        if not r.get("roi_verified"):
            raise ValueError(f"Unverified face ROI: {sid}")
        image_hash = sha256_file(Path(data_root) / r["image_path"])
        key = canonical_hash({"sample_id": sid, "image_sha256": image_hash,
                              "roi": r.get("roi"), "roi_verified": r.get("roi_verified"), "spec": spec})
        nviews = 4 if sid in train_ids else 1
        previous = index["samples"].get(sid)
        if previous and previous["key"] == key and previous["views"] >= nviews and (root / previous["file"]).is_file():
            continue
        pending.append((sid, key, image_hash, nviews))
    if not pending:
        return FeatureBank(root, ids, spec)
    if encoder is None:
        encoder = FrozenEncoder(kind=c["name"], repo_ref="facebookresearch/dinov2:" + c["hub_ref"],
                                local_repo=c["local_repo"], checkpoint=c["checkpoint"],
                                allow_download=c["allow_download"], pretrained=True,
                                input_size=c["image_size"])
    encoder = encoder.to(device).eval()
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Caching requires a fully frozen encoder")
    batch_size = c["extract_batch"]
    if batch_size < 1:
        raise ValueError("extract_batch must be positive")
    for start in range(0, len(pending), max(1, batch_size // 4)):
        chunk = pending[start:start + max(1, batch_size // 4)]
        jobs = [(sid, v) for sid, _, _, n in chunk for v in range(n)]
        all_patches, all_global = [], []
        for j in range(0, len(jobs), batch_size):
            x = torch.stack([preprocess(records_by_id[sid], data_root, v, c["augmentation_seed"], c["image_size"], c["preprocess"])
                             for sid, v in jobs[j:j + batch_size]]).to(device)
            patches, global_feat = encoder(x)
            all_patches.append(patches.float().cpu().numpy())
            all_global.append(global_feat.float().cpu().numpy())
        p, g = np.concatenate(all_patches), np.concatenate(all_global)
        dimensions = (int(p.shape[1]), int(p.shape[2]), int(g.shape[1]))
        if "patch_dim" in index and dimensions != (index["patch_count"], index["patch_dim"], index["global_dim"]):
            raise ValueError("Encoder feature dimensions changed")
        index.update(patch_count=dimensions[0], patch_dim=dimensions[1], global_dim=dimensions[2])
        offset = 0
        for sid, key, image_hash, nviews in chunk:
            name = key + ".npz"
            new_p = p[offset:offset + nviews].astype(np.float16)
            new_g = g[offset:offset + nviews].astype(np.float32)
            previous = index["samples"].get(sid)
            # Adding training views must not change already cached inference
            # features through small batch-dependent CUDA numerical differences.
            if previous and previous["key"] == key and previous["views"] < nviews and (root / previous["file"]).is_file():
                with np.load(root / previous["file"], allow_pickle=False) as old:
                    old_p, old_g = old["patches"], old["global"]
                    count = previous["views"]
                    if str(old["key"].item()) != key or old_p.shape != (count, dimensions[0], dimensions[1]) or old_g.shape != (count, dimensions[2]):
                        raise ValueError("Invalid existing feature shard during view expansion")
                    if not np.isfinite(old_p).all() or not np.isfinite(old_g).all():
                        raise ValueError("Nonfinite existing feature shard during view expansion")
                    new_p[:count], new_g[:count] = old_p, old_g
            fd, temporary = tempfile.mkstemp(dir=root, suffix=".npz")
            os.close(fd)
            try:
                np.savez_compressed(temporary, patches=new_p,
                                    **{"global": new_g}, key=np.array(key))
                os.replace(temporary, root / name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            index["samples"][sid] = {"file": name, "key": key, "views": nviews,
                                     "image_sha256": image_hash, "roi": records_by_id[sid].get("roi"),
                                     "roi_verified": records_by_id[sid].get("roi_verified")}
            offset += nviews
        atomic_json(index_path, index)
        print(f"Features: {min(start + len(chunk), len(pending))}/{len(pending)} images", flush=True)
    return FeatureBank(root, ids, spec)
