"""Verified-animal source partitions with label-value-independent image selection."""
from __future__ import annotations
import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any
Record = dict[str, Any]

def _seed(seed: int, *parts: Any) -> int:
    return int(hashlib.sha256((str(seed) + "::" + "::".join(map(str, parts))).encode()).hexdigest()[:16], 16)

def _image_order(records: list[Record], seed: int, source: str) -> list[str]:
    """Seeded animal round-robin; never uses score values or AU classes."""
    groups = defaultdict(list)
    for record in records:
        if any(record["mask"]):
            groups[record["animal_uid"]].append(record["sample_id"])
    animals = sorted(groups)
    random.Random(_seed(seed, source, "animals")).shuffle(animals)
    for animal in animals:
        groups[animal].sort()
        random.Random(_seed(seed, source, animal, "images")).shuffle(groups[animal])
    ordered = []
    i = 0
    while True:
        added = False
        for animal in animals:
            if i < len(groups[animal]):
                ordered.append(groups[animal][i])
                added = True
        if not added:
            return ordered
        i += 1

def _select(records: list[Record], n: int | str, seed: int, source: str) -> list[str]:
    ordered = _image_order(records, seed, source)
    if str(n).lower() == "full":
        return ordered
    if int(n) <= 0:
        raise ValueError("Label budget must be positive or full")
    if len(ordered) < int(n):
        raise ValueError(f"Source {source}: budget {n} exceeds {len(ordered)} eligible labeled images")
    return ordered[:int(n)]

def _val_animals(records: list[Record], seed: int, source: str, fraction: float, required_images: int = 0) -> set[str]:
    animal_set = {r["animal_uid"] for r in records}
    if None in animal_set:
        raise ValueError("Animal-isolated split requires verified animal identities")
    all_animals = sorted(animal_set)
    if len(all_animals) < 2:
        raise ValueError(f"Source {source} needs at least two animals to separate train and validation")
    n = min(len(all_animals) - 1, max(2 if fraction >= .2 else 1, math.ceil(fraction * len(all_animals))))
    # Availability is permitted metadata; no pain grade enters selection.
    labeled_counts = Counter(r["animal_uid"] for r in records if any(r["mask"]))
    candidates = sorted(all_animals)
    random.Random(_seed(seed, source, "validation_animals", fraction)).shuffle(candidates)
    scored = [a for a in candidates if labeled_counts[a]]
    unscored = [a for a in candidates if not labeled_counts[a]]
    candidates = scored + unscored
    chosen = set(candidates[:n])
    if required_images and sum(labeled_counts[a] for a in chosen) < required_images:
        raise ValueError(f"Source {source}: fixed validation animal pool cannot supply {required_images} images. Change the predeclared protocol, not silently the budget.")
    if not any(labeled_counts[a] for a in all_animals if a not in chosen):
        raise ValueError(f"Source {source}: no scored training animals after validation split")
    return chosen


def source_split(records, budget=50, seed=2026, val_per_source=20):
    """All-source union using the archived per-source animal partition recipe.

    This generates splits for a user's supplied source manifest. It does not
    pretend to recover an unpublished archive if the manifest/exclusions differ.
    """
    from .data import assert_source_split
    split = {"train": [], "val": [], "meta": {"seed": seed, "budget_per_source": budget,
             "validation_per_source": val_per_source, "uses_label_values": False}}
    for source in sorted({r["source"] for r in records}):
        rows = [r for r in records if r["source"] == source]
        va_animals = _val_animals(rows, seed, source, .2, val_per_source)
        tr = [r for r in rows if r["animal_uid"] not in va_animals]
        va = [r for r in rows if r["animal_uid"] in va_animals]
        split["train"].extend(_select(tr, budget, _seed(seed, "training_order"), source))
        split["val"].extend(_select(va, val_per_source, _seed(seed, "validation_order"), source))
    assert_source_split(records, split)
    return split
