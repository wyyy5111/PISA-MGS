"""Command-line entry point for split creation and the three-member pipeline."""
from __future__ import annotations
import argparse
import json
from .data import load_manifest
from .pipeline import run
from .splits import source_split
from .utils import atomic_json


def main():
    parser = argparse.ArgumentParser(prog="pisa-mgs", description="Frozen-feature MGS scoring with SCTA/PISA")
    commands = parser.add_subparsers(dest="command", required=True)
    splitting = commands.add_parser("split", help="Create an animal-separated all-source train/val union")
    splitting.add_argument("--source-manifest", required=True)
    splitting.add_argument("--out", required=True)
    splitting.add_argument("--budget", type=int, default=50)
    splitting.add_argument("--seed", type=int, default=2026)
    splitting.add_argument("--val-per-source", type=int, default=20)
    running = commands.add_parser("run", help="Extract features, fit source heads, and predict the full target pool")
    for arg in ("source-manifest", "source-data-root", "target-manifest", "target-data-root", "split", "config", "work-dir"):
        running.add_argument("--" + arg, required=True)
    running.add_argument("--extract-device", default="auto")
    running.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.command == "split":
        records = load_manifest(args.source_manifest)
        if any(not r.get("source") or not r.get("animal_uid") for r in records):
            raise ValueError("Source split creation requires source and verified animal_uid for every record")
        result = source_split(records, args.budget, args.seed, args.val_per_source)
        atomic_json(args.out, result)
        print(json.dumps({"train": len(result["train"]), "val": len(result["val"])}, indent=2))
    else:
        result = run(args.source_manifest, args.source_data_root, args.target_manifest,
                     args.target_data_root, args.split, args.config, args.work_dir,
                     args.extract_device, args.allow_download)
        print(json.dumps(result, indent=2))
