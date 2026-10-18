# External models and data

This repository distributes project source code and illustrative metadata only. It does not bundle upstream source trees, pretrained weights, dataset images, annotations, caches, or trained heads. The license of one component does not grant rights to another component or to its training data.

| Component | Official origin and terms | Released usage |
|---|---|---|
| DINOv2-S/14 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), [Apache-2.0 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE), [model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) | Frozen official checkpoint, pinned code commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| AIMv2-L/14 | [Apple ml-aim](https://github.com/apple/ml-aim), [official model weight license](https://huggingface.co/apple/aimv2-large-patch14-224/blob/main/LICENSE), [timm distribution](https://huggingface.co/timm/aimv2_large_patch14_224.apple_pt) | Frozen `timm:aimv2_large_patch14_224.apple_pt`; the Apple weight terms restrict use to noncommercial scientific research / academic development. Follow the full weight license; do not infer commercial rights from code licenses. |
| ConvNeXt-T | [torchvision implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py), [torchvision BSD-3-Clause code license](https://github.com/pytorch/vision/blob/main/LICENSE), [official model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) | `torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT`, `convnext_tiny-983f1562.pth`. The torchvision code license is not a blanket license for pretrained weights or ImageNet data. Consult the weight/data terms. |
| MGF | [Associated dataset study](https://doi.org/10.64898/2026.02.16.706098) | User-supplied source face crops, ordinal ratings, verified animal IDs |
| GrimACE | [Associated study](https://doi.org/10.1038/s41684-026-01695-9), [official repository](https://github.com/ETHZ-INS/GrimACE_manuscript), [Zenodo resources](https://zenodo.org/records/18299984) | Optional user-supplied target images/ratings; no cropper weights included |

Recorded upstream asset identifiers retained from the original metadata:

- DINOv2-S checkpoint: [official URL](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth), SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- ConvNeXt-T checkpoint: [official URL](https://download.pytorch.org/models/convnext_tiny-983f1562.pth), SHA256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.
- AIMv2 is fetched through the upstream Hugging Face/timm cache; the original metadata did not record a content hash, so none is invented here.

The full three-backbone pipeline requires satisfying the terms of every selected weight and dataset. No third-party license is relicensed by this project.
