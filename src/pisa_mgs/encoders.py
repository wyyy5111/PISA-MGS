"""Official frozen DINOv2-S, AIMv2-L, and ConvNeXt-T feature interfaces."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import torch
from torch import Tensor, nn

def _load_checkpoint(backbone: nn.Module, checkpoint: str | Path) -> None:
    obj = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError("Encoder checkpoint must contain a model state_dict.")
    if obj and all(key.startswith("module.") for key in obj):
        obj = {key[7:]: value for key, value in obj.items()}
    backbone.load_state_dict(obj, strict=True)

class FrozenEncoder(nn.Module):
    """Uniform fixed patch/global interface for official pretrained encoders.

    ``allow_download=False`` refuses absent code or weights. A provided local
    DINO repository and checkpoint work completely offline. Random torchvision
    models are restricted to an explicit ``test_only=True`` construction.
    DINO repo_ref should be pinned by the experiment configuration.
    """
    DIMS = {"dinov2_vits14": 384, "convnext_tiny": 768}
    DINO_WEIGHT_FILES = {"dinov2_vits14": "dinov2_vits14_pretrain.pth"}

    def __init__(self, kind: str = "dinov2_vits14", repo_ref: str = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                 local_repo: str | Path | None = None, checkpoint: str | Path | None = None,
                 allow_download: bool = False, pretrained: bool = True, test_only: bool = False,
                 backbone: nn.Module | None = None, input_size: int = 224):
        super().__init__()
        if kind not in ("dinov2_vits14", "timm:aimv2_large_patch14_224.apple_pt", "convnext_tiny"):
            raise ValueError(f"Unsupported encoder: {kind}.")
        if input_size != 224:
            raise ValueError("The released core recipes use 224-pixel encoder inputs")
        self.kind, self.input_size = kind, input_size
        if kind.startswith("timm:"):
            self.patch_dim = self.global_dim = None  # resolved from the constructed model
        else:
            self.patch_dim = self.global_dim = self.DIMS[kind]
        self.repo_ref, self.test_only = repo_ref, test_only
        if backbone is not None and not test_only:
            raise ValueError("Backbone injection is restricted to explicit structural tests (test_only=True).")
        if backbone is None and not pretrained and checkpoint is None and not test_only:
            raise ValueError("Production experiments require pretrained encoder weights.")
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        if backbone is None:
            if kind.startswith("timm:"):
                backbone = self._make_timm(kind, allow_download, pretrained and checkpoint is None, input_size)
            elif kind.startswith("dinov2_"):
                backbone = self._make_dino(kind, repo_ref, local_repo, checkpoint, allow_download, pretrained)
            else:
                backbone = self._make_torchvision(kind, checkpoint, allow_download, pretrained)
        self.backbone = backbone
        if kind.startswith("timm:"):
            # num_features is forward_features' final channel count (what the
            # patch interface needs); embed_dim is only the first stage for
            # hierarchical models like Swin, so prefer num_features.
            dim = getattr(backbone, "num_features", None) or getattr(backbone, "embed_dim", None)
            if dim is None:
                raise ValueError(f"Cannot determine feature dimension for {kind}")
            self.patch_dim = self.global_dim = int(dim)
        if checkpoint is not None:
            _load_checkpoint(self.backbone, checkpoint)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    @staticmethod
    def _make_dino(kind, repo_ref, local_repo, checkpoint, allow_download, pretrained):
        if local_repo is not None:
            repo = Path(local_repo)
            if not (repo / "hubconf.py").is_file():
                raise FileNotFoundError(f"No hubconf.py in local DINO repository: {repo}")
        else:
            owner_repo, sep, ref = repo_ref.partition(":")
            ref = ref if sep else "main"
            owner, repo_name = owner_repo.split("/")
            cached = Path(torch.hub.get_dir()) / f"{owner}_{repo_name}_{ref.replace('/', '_')}"
            repo = cached if (cached / "hubconf.py").is_file() else None
        load_pretrained = pretrained and checkpoint is None
        if load_pretrained and not allow_download:
            filename = FrozenEncoder.DINO_WEIGHT_FILES.get(kind)
            weights = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not weights.is_file():
                raise RuntimeError(f"DINO pretrained weights for {kind} are absent. "
                                   "Provide checkpoint or explicitly allow download.")
        if repo is None and not allow_download:
            raise RuntimeError("DINO repository is absent. Provide local_repo or explicitly allow download.")
        return torch.hub.load(str(repo) if repo is not None else repo_ref, kind,
                              source="local" if repo is not None else "github", pretrained=load_pretrained,
                              trust_repo=True, skip_validation=True)

    @staticmethod
    def _make_timm(kind, allow_download, pretrained, input_size=224):
        """AIMv2-L through its official timm distribution and weight cache."""
        import timm
        name = kind.split(":", 1)[1]
        if pretrained and not allow_download:
            try:
                from huggingface_hub import try_to_load_from_cache
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("huggingface_hub is required for cached timm weights") from exc
            repo_id = f"timm/{name}"
            cached = any(
                isinstance(path, str)
                for file in ("pytorch_model.bin", "model.safetensors")
                for path in [try_to_load_from_cache(repo_id, file)]
            )
            if not cached:
                raise RuntimeError(
                    f"timm weights for {name} are not in the local Hugging Face cache and "
                    "downloads are disabled. Re-run with allow_download to fetch official weights.")
        # timm enforces the construction-time img_size for some families
        # (AIMv2), so pass the declared input size through; None keeps the
        # model's native resolution (224) for the common case.
        img = None if input_size == 224 else input_size
        return timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img)

    @staticmethod
    def _make_torchvision(kind, checkpoint, allow_download, pretrained):
        from torchvision import models
        factory, weights = models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT
        load_pretrained = pretrained and checkpoint is None
        if load_pretrained and not allow_download:
            filename = Path(urlparse(weights.url).path).name
            if not (Path(torch.hub.get_dir()) / "checkpoints" / filename).is_file():
                raise RuntimeError(f"{kind} pretrained weights are absent. Provide checkpoint or explicitly allow download.")
        return factory(weights=weights if load_pretrained else None)

    def train(self, mode: bool = True):
        # Keeping dropout/batchnorm permanently in eval is part of the protocol.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1:] != (3, self.input_size, self.input_size):
            raise ValueError(f"Encoder images must be [B,3,{self.input_size},{self.input_size}].")
        if self.kind.startswith("timm:"):
            feats = self.backbone.forward_features(images)
            if feats.ndim == 4 and self.patch_dim and feats.shape[-1] == self.patch_dim \
                    and feats.shape[1] != self.patch_dim:
                # Swin-style NHWC map [B,H,W,C]: channel is last.
                patches, global_feat = feats.flatten(1, 2), feats.mean((1, 2))
            elif feats.ndim == 4:  # CNN feature map [B,C,H,W]
                patches, global_feat = feats.flatten(2).transpose(1, 2), feats.mean((2, 3))
            else:
                n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
                n_prefix = int(1 if n_prefix is None else n_prefix)
                patches = feats[:, n_prefix:]
                # Leading token is the class token; registers are excluded from
                # the global summary, matching each backbone's own convention.
                global_feat = feats[:, 0] if n_prefix > 0 else feats.mean(1)
        elif self.kind.startswith("dinov2_"):
            features = self.backbone.forward_features(images)
            patches, global_feat = features["x_norm_patchtokens"], features["x_norm_clstoken"]
            expected_patches = (self.input_size // 14) ** 2
            if patches.shape[1] != expected_patches:
                raise ValueError(f"DINOv2 {self.kind} must provide {expected_patches} "
                                 f"patches at {self.input_size}px.")
        elif self.kind == "convnext_tiny":
            # ConvNeXt classifier head is dropped; stage-4 feature map [B,C,7,7].
            x = self.backbone.features(images)
            patches, global_feat = x.flatten(2).transpose(1, 2), x.mean((2, 3))
        if patches.ndim != 3 or patches.shape[-1] != self.patch_dim or global_feat.shape != (images.shape[0], self.global_dim):
            raise ValueError("The encoder checkpoint does not match the declared feature interface.")
        return patches.detach(), global_feat.detach()
