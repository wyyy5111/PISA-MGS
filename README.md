# PISA-MGS

Frozen-feature mouse grimace scale (MGS) scoring with **source-calibrated target alignment (SCTA)** and **per-image shift adaptation (PISA)**. Five learned queries pool frozen visual tokens into five ordinal grade distributions. Source-fitted affine calibration and equal-weight probability averaging combine DINOv2-S/14, AIMv2-L/14, and ConvNeXt-T.

Authors: **Wenji Wang, Yiming Yang, Dehua Zhang**. Associated work: *Per-image shift adaptation for annotation-efficient facial action scoring across domains* (2026, unpublished). See [CITATION.cff](CITATION.cff).

This repository packages the implemented core scoring and transfer path. It contains no experimental search, ablation runners, datasets, feature caches, trained heads, or third-party weights. The archived transfer analyses were exploratory: target labels informed retrospective method/member selection. Packaging and synthetic checks do not establish new performance or reproduce every reported experiment.

## Install and check

Python 3.12 is the reference environment. Install a matching PyTorch build for your hardware if needed; the archived CPU runtime used PyTorch 2.8.0 and torchvision 0.23.0.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pisa_mgs --help
python -m pytest -q
```

The tests use synthetic tensors and temporary example manifests. They require no animal images, pretrained weights, or network access. Feature extraction supports CPU, CUDA, and MPS through `--extract-device`; the retained head-training path runs on CPU, matching the frozen driver.

## Data and manifests

Obtain data separately from its providers. MGF (*Mouse grimace faces*) is associated with [Andresen et al., 2026](https://doi.org/10.64898/2026.02.16.706098); GrimACE is associated with [Sturman et al., 2026](https://doi.org/10.1038/s41684-026-01695-9) and [the official code](https://github.com/ETHZ-INS/GrimACE_manuscript). Follow their access and reuse terms. The code expects reviewed face crops; a face detector is not included.

A JSONL manifest contains one object per image. See [source_manifest.jsonl](examples/source_manifest.jsonl), [target_manifest.jsonl](examples/target_manifest.jsonl), and [source_split.json](examples/source_split.json). These are illustrative metadata; the referenced images are not supplied.

- `sample_id`: unique string, distinct between source and target.
- `image_path`: portable relative path under the corresponding data root.
- `roi`: `[x,y,width,height]` in original image pixels, or `null` for an already cropped face. `roi_verified: true` records a completed human review; it is not a detector confidence score. Include ears and whiskers.
- `source`: acquisition source; `animal_uid`: verified, globally namespaced animal identity, such as `source_A::animal_17`. Source training/validation require known disjoint animal identities. Do not substitute filename prefixes or operational sessions for verified animals.
- `q`: shape `[5,3]`, empirical distributions over valid grades `0,1,2`. AU order is `ot, nb, cb, ep, wc`: orbital tightening, nose bulge, cheek bulge, ear position, whisker change.
- `mask`: five booleans. An AU without valid ratings has `false` and a zero placeholder distribution. Unrateable `9`, missing entries, and invalid grades are excluded from the empirical distribution, never converted to grade zero. `pisa_mgs.data.empirical_labels` builds labels from already validated rating lists.

Target inference requires neither `q` nor `mask`; omit both for an unlabeled image. **Every supplied target image participates in the target mean**, including images without ratings. Optional target labels are read only for evaluation after predictions are fixed. MGS5 is the mean of all five expected AU scores; its evaluation includes only images with all five observed labels.

Provide an existing animal-separated split, or generate an all-source union with the original label-value-independent animal/round-robin selection primitives:

```bash
pisa-mgs split --source-manifest source_manifest.jsonl \
  --budget 50 --val-per-source 20 --seed 2026 --out source_split.json
```

With all five MGF sources and sufficient eligible records this selects 250 training and 100 validation images. Exact archive membership also depends on the supplied manifest and audited exclusions; illustrative manifests cannot recover it. Validation animals are excluded from training and source alignment statistics. Label availability, animal identity, and source metadata may affect selection; grade values do not.

## Run the core pipeline

```bash
pisa-mgs run \
  --source-manifest source_manifest.jsonl --source-data-root source_images \
  --target-manifest target_manifest.jsonl --target-data-root target_images \
  --split source_split.json --config configs/frozen.json \
  --work-dir work/example --extract-device auto --allow-download
```

Downloads are disabled unless `--allow-download` is supplied. For offline use, populate the official caches or set `encoder.local_repo` / `encoder.checkpoint` in the configuration. DINOv2 code is pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Read [THIRD_PARTY.md](THIRD_PARTY.md) before obtaining weights, especially AIMv2's separate research license.

The command writes source and target feature caches separately, reusable source-head checkpoints, per-member source OLS coefficients and target alpha values, predictions, and a summary under the chosen work directory. Repeated runs validate source checkpoint signatures before reuse. There is no partial-epoch resume. The full target pool must be available at prediction time: transfer is transductive, and changing that pool can change every prediction.

## Preserved algorithms and training recipes

A linear projection maps patch features to 128 dimensions. Five learned queries apply scaled dot-product attention over patches; each AU uses its own `128 → 64 → 3` classifier with GELU and dropout 0.1. These queries have no supervised anatomical localization guarantee. Expected score is `p[1] + 2*p[2]`. AIMv2 and ConvNeXt optimize squared error against the empirical soft-label expectation, with observed-AU masks.

[configs/frozen.json](configs/frozen.json) follows the **executed frozen driver**, whose DINOv2 defaults differ from the accompanying high-level protocol text:

| Member | Preprocessing | Loss / schedule | EMA and selection | Transfer |
|---|---|---|---|---|
| DINOv2-S/14 | grayscale + PIL equalize | RPS + 0.1 soft CE; lr 1e-3; warmup/cosine | no EMA; max 100 epochs; min 20; patience 15 | PISA |
| AIMv2-L/14 `apple_pt` | grayscale + PIL equalize | expected-score MSE; lr 3e-4; warmup/cosine | EMA 0.99 **twice per optimizer step**; 100 epochs | PISA |
| torchvision ConvNeXt-T | standard grayscale | expected-score MSE; constant lr 3e-4 | EMA 0.99 once per step; 100 epochs | SCTA |

All use AdamW and weight decay 1e-4. DINOv2/AIMv2 draw source → animal → image uniformly with replacement, average observed AUs within each image, and use source/animal-balanced validation MAE. Effective batches are 16 / 32 / 64 for at most 80 / 200 / more images, accumulated in physical batches of at most 32; gradients are clipped to norm 1. ConvNeXt shuffles images without replacement in batches of 32, averages observed image–AU pairs, and selects by pair-weighted source validation MAE. The best validation weights are deployed. Five seeds, 2026–2030, give 15 equal-weight ensemble members.

Images use grayscale, RGB replication, bicubic letterboxing to 224 with gray padding, and fixed ImageNet normalization. The historical equalization function mirrors **every nonzero training view**; its three augmentation slots therefore all contain the horizontal flip. Standard-grayscale caches contain original, horizontal flip, and two deterministic mild brightness/contrast views. ConvNeXt training uses only the original view. These existing training semantics are retained, including AIMv2's double EMA update (effective per-step decay `0.99²`), rather than silently normalized across members.

For source-training patch mean `mu_s`, full unlabeled-target patch mean `mu_t`, and `u = mu_t - mu_s`, SCTA predicts on `X_t - mu_t`. PISA computes

```text
alpha_i = clip(mean_over_patches(((X_i - mu_s) @ u) / dot(u, u)), 0, 1)
X_i' = X_i - alpha_i * mu_t
```

The subtraction uses **`mu_t`**, and the projection is averaged over patches **before** clipping. Global/CLS features remain unchanged. Source calibration always predicts on `X_s - mu_t`, including for PISA; source validation images are excluded. Per-AU OLS fits `[shifted source expectation, 1]` to observed source training expectations. Target scores are calibrated, clipped to `[0,2]`, converted to adjacent-grade probabilities preserving their expectation, then averaged across seeds/backbones. ViTs use PISA and ConvNeXt uses SCTA.

The archived implementation divides by `dot(u,u)` without a zero/near-zero rule. This package explicitly raises `ValueError` when the squared norm is at most `1e-12` (configurable input-validation threshold). No epsilon denominator, alpha-zero fallback, or unvalidated degenerate-direction prediction is introduced. Use SCTA as a separately specified method if desired; the code never switches silently.

## Member count

The default frozen external-validation recipe uses five seeds per backbone (15 heads). The associated retrospective analysis used seven per backbone (21 heads). The seed list is explicit and configurable. To request 21 heads with this release's retained recipes:

```bash
python - <<'PYCONFIG'
import json
from pathlib import Path
config = json.loads(Path("configs/frozen.json").read_text())
config["seeds"] = list(range(2026, 2033))
Path("configs/seven_seeds.json").write_text(json.dumps(config, indent=2) + "\n")
PYCONFIG
```

Then pass `--config configs/seven_seeds.json` and use a new work directory. This changes the member count; it is not a claim that packaging reran or independently validated the historical final analysis. The default configuration remains the executable five-seed external-validation recipe.

## Provenance and scope

[SOURCE_MAP.md](SOURCE_MAP.md) maps retained modules to their original implementations and records packaging changes. The query head, losses, training semantics, feature interfaces, shift/calibration order, and arithmetic probability averaging are retained. Portable roots, genuine unlabeled-target support, cache separation, manifest checks, and explicit degenerate-input rejection repair the driver interface. This is a focused core release, not a claim of bitwise equivalence for the complete historical driver or reproduction of all analyses. No new real-data training or performance validation was performed for packaging.

No project-code license has been selected; this repository does not include a `LICENSE` grant. Third-party materials retain their own terms.
