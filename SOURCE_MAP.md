# Core source provenance

Paths below refer to the original working project, relative to its root. Only the listed core implementation paths were retained; no original search, baseline, ablation, paper, or result directory is distributed.

| Released file | Original implementation | Retained behavior / packaging changes |
|---|---|---|
| `src/pisa_mgs/models.py` | `src/mouse_mgs/models.py`: `AUHead`, `expected_score`, label validation, ordinal/expected-score losses | Exact query forward and two executed losses; unused heads, relationship/gain models, and research regularizers omitted. |
| `src/pisa_mgs/encoders.py` | `src/mouse_mgs/models.py`: `_load_checkpoint`, `FrozenEncoder` | Exact DINOv2-S, AIMv2-L, torchvision ConvNeXt-T token interfaces; unused backbone branches removed. Explicit local AIM checkpoints construct the architecture without an unrelated pretrained download. Input size restricted to the released 224-pixel recipes. |
| `src/pisa_mgs/features.py` | `src/mouse_mgs/features.py`; `scripts/run_frozen_protocol.py`: `equalize_preprocess` | ROI checks, deterministic standard views, cache integrity and partition access preserved. Equalize mode is an explicit cache specification rather than a global monkey patch; all three nonzero equalized views remain mirrors. |
| `src/pisa_mgs/training.py` | `src/mouse_mgs/training.py`: balanced draws, accumulation, warmup/cosine, loss normalization, EMA, validation; `scripts/run_frozen_protocol.py`: `train_member` | Retains DINO/AIM protocol path and ConvNeXt constant path, including twice-per-step protocol EMA versus once-per-step custom EMA. Research alternatives and partial-run resume omitted. Head-training CPU behavior preserved. |
| `src/pisa_mgs/transfer.py` | `scripts/run_frozen_protocol.py`: `transform_predict` | Exact nondegenerate PISA/SCTA shift, full-source-shift OLS, clipping, adjacent-grade reconstruction. Exposes tensors and separate source/target banks; explicit zero/near-zero rejection replaces undefined division, without defining a fallback algorithm. |
| `src/pisa_mgs/metrics.py` | `src/mouse_mgs/metrics.py`: label alignment, animal/source-balanced checkpoint metric and AU/MGS evaluation | Core metrics preserved; relationship-selection/oracle diagnostics and bootstrap analyses omitted. MGS5 uses complete-five labels. |
| `src/pisa_mgs/splits.py` | `src/mouse_mgs/splits.py`: `_seed`, `_image_order`, `_select`, `_val_animals` | Exact label-value-independent selection primitives; focused all-source union wrapper. No research split matrix or ablation launcher. |
| `src/pisa_mgs/utils.py` | `src/mouse_mgs/utils.py` | Hashes, atomic local writes, seeding, device and JSONL helpers. Unused RNG resume helpers omitted. |
| `configs/frozen.json` | `scripts/run_frozen_protocol.py`: `MEMBERS`, `SEEDS`; `src/mouse_mgs/config.py`: inherited defaults | Records the executed member defaults explicitly. DINO is RPS+CE/no-EMA/patience15; AIM is MSE/EMA0.99/patience100; ConvNeXt uses the custom constant path. |
| `src/pisa_mgs/data.py`, `pipeline.py`, `cli.py` | Portable interfaces around the functions above | New manifest validation and argument wiring: true unlabeled-target inference, separate roots/caches, no target-label filtering, source checkpoint signatures, optional evaluation after inference. These interface repairs are not a claim of bitwise equivalence for the complete old driver. |
| `THIRD_PARTY.md` | `experiments_local_v2/protocol/weights_provenance.json`, upstream model/data metadata | Keeps model identifiers, official origins and recorded hashes, strips local paths, clarifies that code licenses do not license weights/data. |

The high-level `FROZEN_EXTERNAL_PROTOCOL.md` says MSE/EMA for all members, but its accompanying executable actually inherits different DINO defaults. The release follows executable behavior. It also preserves the historical double EMA update in the protocol training path instead of silently changing training semantics.

## Verification for this extraction

- Synthetic release tests: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider` — 8 tests, passed. Covers query reload, masks/NaNs, coefficient ordering and degenerate directions, calibration expectation, animal split checks, all three training loops and unlabeled end-to-end inference; adding target labels leaves predictions unchanged.
- Independent comparison with the original frozen transform on five random fixtures: PISA and SCTA probabilities, source OLS and triangular reconstruction matched with maximum absolute error 0.
- Original versus released query head and both losses: maximum absolute error 0. Both preprocessing modes across four cached views: maximum absolute error 0. Balanced draws and scheduler matched.
- Original `train_base` versus extracted protocol path on the same synthetic feature bank, two epochs each for DINO RPS+CE and AIM MSE/EMA: state-dictionary maximum absolute error 0. Original `train_member` versus extracted ConvNeXt constant path for 100 synthetic epochs: state-dictionary maximum absolute error 0. This checks implementation extraction, not animal-data performance.

## Original implementation fingerprints

SHA256 values identify the read-only source versions used for extraction. No original source or result archive is required at runtime.

| Original file | SHA256 |
|---|---|
| `src/mouse_mgs/models.py` | `49b0288ae867d48774592aadaa9e72da76017d3d4d3bee8d8d455b633db5acc5` |
| `src/mouse_mgs/features.py` | `8146208e716debc68e2dff29f7e37d58226faab7a0e73a4b0b0b535129e9cd4a` |
| `src/mouse_mgs/training.py` | `7d4ffd1db7c6deaa87d81b957b2871719b494295677b508106f0ec1cd51c13c5` |
| `src/mouse_mgs/metrics.py` | `7b3fb9ab1d58e23fd74361c60223eb8070352ce6ac3b47a18d13057ec01d74e8` |
| `src/mouse_mgs/splits.py` | `0e8a434059bfe9bc636a8c7f14bc03453e8dd9fe58cdf417add00affce25a168` |
| `src/mouse_mgs/utils.py` | `d706c3ca93916f42b00646174d29c8b6e14afbb9d9ff7a50430bb5511bfec7e9` |
| `src/mouse_mgs/config.py` | `d5917285f409a25a99551a463ad546c50527638afe2dcbc5da12633e734b207b` |
| `scripts/run_frozen_protocol.py` | `0219af2d55361bcd06c1c4b3824af5dcb6840413dfbab971a5f56a6bd68ff37c` |
| `experiments_local_v2/protocol/FROZEN_EXTERNAL_PROTOCOL.md` | `3de7369d60043f790da36cc0457611b997fd3450fd72cd604b50333b58e0d10c` |
| `experiments_local_v2/protocol/weights_provenance.json` | `e4c5f0580d34adfb6cda39dbda94fe733f37e3dbc82ed1212075096bb9394dc5` |
