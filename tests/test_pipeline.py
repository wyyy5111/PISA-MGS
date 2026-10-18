"""End-to-end synthetic integration of all member loops and unlabeled inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pisa_mgs import pipeline


class SyntheticBank:
    patch_dim = global_dim = 8
    patch_count = 4

    def __init__(self, ids, train_ids, target=False):
        self.allowed_ids, self.train_ids = set(ids), set(train_ids)
        self.index = {"spec": {"synthetic": True, "target": target},
                      "samples": {i: {"key": i} for i in ids}}
        self.target = target

    def views(self, sample_id):
        return 4 if sample_id in self.train_ids else 1

    def get(self, sample_id, view=0):
        assert sample_id in self.allowed_ids
        seed = int(hashlib.sha256(f"{sample_id}:{view}".encode()).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)
        patches = torch.randn(4, 8, generator=generator) + (1 if self.target else 0)
        return patches, patches.mean(0)


def test_full_pipeline_with_unlabeled_target_and_separate_roots(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    roots_seen = []
    def extract(records, data_root, cache_dir, config, ids, train_ids=()):
        roots_seen.append((data_root, str(cache_dir)))
        return SyntheticBank(ids, train_ids, data_root == "target_images")
    monkeypatch.setattr(pipeline, "extract_features", extract)
    source = []
    for k in range(6):
        q = np.zeros((5, 3))
        q[:, k % 3] = 1
        source.append({"sample_id": f"s{k}", "dataset": "synthetic", "source": "site_A",
                       "animal_uid": f"animal_{k}", "image_path": f"s{k}.png", "roi_verified": True,
                       "q": q.tolist(), "mask": [True] * 5})
    targets = [{"sample_id": f"t{k}", "image_path": f"t{k}.png", "roi_verified": True}
               for k in range(3)]
    source_path, target_path = tmp_path / "source.jsonl", tmp_path / "target.jsonl"
    source_path.write_text(''.join(json.dumps(r) + '\n' for r in source))
    target_path.write_text(''.join(json.dumps(r) + '\n' for r in targets))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": [f"s{k}" for k in range(4)], "val": ["s4", "s5"]}))
    config = json.loads((Path(__file__).parents[1] / "configs/frozen.json").read_text())
    config["seeds"] = [2026]
    for spec in config["members"].values():
        spec["training"]["max_epochs"] = 1
        if spec["loop"] == "protocol":
            spec["training"]["min_epochs"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    work = tmp_path / "work"
    summary = pipeline.run(source_path, "source_images", target_path, "target_images", split_path,
                           config_path, work)
    assert summary["target_pool"] == 3 and "evaluation" not in summary
    before = (work / "predictions.jsonl").read_text()
    assert len(before.splitlines()) == 3
    assert all("source" in cache if root == "source_images" else "target" in cache
               for root, cache in roots_seen)
    assert {root for root, _ in roots_seen} == {"source_images", "target_images"}
    # Add target labels: predictions must be unchanged, while evaluation becomes available.
    for row in targets:
        row["q"], row["mask"] = [[0, 0, 1]] * 5, [True] * 5
    target_path.write_text(''.join(json.dumps(r) + '\n' for r in targets))
    after_summary = pipeline.run(source_path, "source_images", target_path, "target_images", split_path,
                                 config_path, work)
    assert (work / "predictions.jsonl").read_text() == before
    assert after_summary["evaluation"]["n_complete5_images"] == 3
