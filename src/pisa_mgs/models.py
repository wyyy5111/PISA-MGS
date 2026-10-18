"""Five-query AU head and the two losses used by the released members."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
AU_NAMES = ("ot", "nb", "cb", "ep", "wc")

def expected_score(prob: Tensor) -> Tensor:
    """Expected 0/1/2 AU score, retaining continuous predictions."""
    if prob.shape[-1] != 3:
        raise ValueError("The last probability dimension must be three.")
    return prob[..., 1] + 2.0 * prob[..., 2]

def _validate_labels(q: Tensor, mask: Tensor) -> None:
    if q.ndim != 3 or q.shape[1:] != (5, 3) or mask.shape != q.shape[:2]:
        raise ValueError("q and mask must have shapes [B,5,3] and [B,5].")
    valid = q[mask.bool()]
    if valid.numel() and (
        not torch.isfinite(valid).all()
        or (valid < 0).any()
        or not torch.allclose(valid.sum(-1), torch.ones_like(valid[:, 0]), atol=1e-5)
    ):
        raise ValueError("Every observed AU must have a valid probability target.")

def _masked_image_average(values: Tensor, mask: Tensor) -> Tensor:
    """Equal weight per image after averaging its observed AUs."""
    valid = mask.bool()
    per_image = (values * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    observed_images = valid.any(-1)
    if not observed_images.any():
        return values.sum() * 0.0
    return per_image[observed_images].mean()

def ordinal_loss(logits: Tensor, q: Tensor, mask: Tensor, ce_weight: float = 0.1,
                 reduction: str = "mean") -> Tensor:
    """Normalized RPS + soft CE, only for valid AUs.

    RPS averages squared errors at the two cumulative ordinal thresholds.
    Each observed AU is averaged within its image; images are then averaged.
    An all-missing image contributes no term. ``reduction='none'`` returns
    per-image losses [B] for group objectives, with all-missing images zeroed.
    ``reduction='au'`` returns the masked individual AU terms [B,5].
    """
    if logits.shape != q.shape:
        raise ValueError("logits and q shapes must agree.")
    _validate_labels(q, mask)
    # Missing entries may contain NaNs in callers' manifests. Mask before any
    # arithmetic: multiplying a NaN by zero would still poison the loss.
    clean_q = torch.where(mask.bool().unsqueeze(-1), q, torch.zeros_like(q))
    prob = logits.softmax(-1)
    rps = (prob.cumsum(-1)[..., :2] - clean_q.cumsum(-1)[..., :2]).square().mean(-1)
    ce = -(clean_q * F.log_softmax(logits, -1)).sum(-1)
    values = torch.where(mask.bool(), rps + ce_weight * ce, torch.zeros_like(rps))
    if reduction == "au":
        return values
    if reduction == "none":
        valid = mask.bool()
        return values.sum(-1) / valid.sum(-1).clamp_min(1)
    if reduction != "mean":
        raise ValueError("reduction must be 'mean', 'none', or 'au'.")
    return _masked_image_average(values, mask)

def expected_score_loss(logits: Tensor, q: Tensor, mask: Tensor, kind: str = "huber",
                        beta: float = 0.25, reduction: str = "mean") -> Tensor:
    """Regression on the expected 0/1/2 grade (continuous AU track).

    The target is the empirical-distribution expectation. Probabilities are
    still produced by the softmax, so the same head serves both tracks; this
    loss compares expectations numerically and is not a probability-quality
    term. Only valid AUs contribute.
    """
    if kind not in ("huber", "mse", "mae"):
        raise ValueError("kind must be 'huber', 'mse', or 'mae'.")
    if beta <= 0:
        raise ValueError("Huber beta must be positive.")
    if logits.shape != q.shape:
        raise ValueError("logits and q shapes must agree.")
    _validate_labels(q, mask)
    clean_q = torch.where(mask.bool().unsqueeze(-1), q, torch.zeros_like(q))
    target = expected_score(clean_q)
    prediction = expected_score(logits.softmax(-1))
    if kind == "huber":
        values = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    elif kind == "mae":
        values = (prediction - target).abs()
    else:
        values = (prediction - target).square()
    values = torch.where(mask.bool(), values, torch.zeros_like(values))
    if reduction == "au":
        return values
    if reduction == "none":
        valid = mask.bool()
        return values.sum(-1) / valid.sum(-1).clamp_min(1)
    if reduction != "mean":
        raise ValueError("reduction must be 'mean', 'none', or 'au'.")
    return _masked_image_average(values, mask)

class AUHead(nn.Module):
    """Five AU queries pool frozen patch features into independent classifiers."""
    def __init__(self, patch_dim: int = 384, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.patch_dim, self.hidden = patch_dim, hidden
        self.project = nn.Linear(patch_dim, hidden)
        self.queries = nn.Parameter(torch.randn(5, hidden) * 0.02)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))
            for _ in AU_NAMES
        ])

    def forward(self, patches: Tensor, global_feat: Tensor | None = None, return_attention: bool = False):
        if patches.ndim != 3 or patches.shape[-1] != self.patch_dim or patches.shape[1] < 1:
            raise ValueError("patches must be [B,N,patch_dim] with N >= 1.")
        tokens = self.project(patches)
        attention = torch.einsum("bnh,ah->ban", tokens, self.queries) / math.sqrt(self.hidden)
        attention = attention.softmax(-1)
        pooled = torch.einsum("ban,bnh->bah", attention, tokens)
        logits = torch.stack([head(pooled[:, a]) for a, head in enumerate(self.heads)], dim=1)
        return (logits, attention) if return_attention else logits

