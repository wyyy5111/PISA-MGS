"""SCTA, PISA, source-only OLS calibration, and probability averaging."""
from __future__ import annotations
import numpy as np
import torch


def pisa_coefficients(source_patches, target_patches, min_shift_norm_sq=1e-12):
    """Image coefficient: mean patch projection, then clip to [0, 1].

    The archived implementation has no zero-denominator policy. We reject a
    degenerate direction explicitly; no unvalidated alpha fallback is invented.
    """
    mu_s = source_patches.reshape(-1, source_patches.shape[-1]).mean(0)
    mu_t = target_patches.reshape(-1, target_patches.shape[-1]).mean(0)
    u = mu_t - mu_s
    uu = float((u * u).sum())
    if not np.isfinite(uu) or uu <= min_shift_norm_sq:
        raise ValueError("PISA source/target mean direction is zero or near zero; no validated fallback is defined")
    projection = (((target_patches.reshape(len(target_patches), -1, target_patches.shape[-1])
                    - mu_s) @ u) / uu).mean(-1)
    return projection.clamp(0, 1)


def triangular_probabilities(expectations):
    """Adjacent-grade distribution preserving each clipped calibrated expectation."""
    e = np.clip(np.asarray(expectations), 0, 2)
    p0, p2 = np.clip(1 - e, 0, 1), np.clip(e - 1, 0, 1)
    return np.stack([p0, 1 - p0 - p2, p2], -1)


@torch.no_grad()
def transfer_predict(head, source_patches, source_global, target_patches, target_global,
                     source_q, source_mask, transform="pisa", min_shift_norm_sq=1e-12):
    """Predict without target labels. Calibration always uses fully shifted source.

    PISA subtracts alpha_i * mu_t (not alpha_i * (mu_t-mu_s)). Global/CLS
    summaries are deliberately left unchanged, as in the frozen implementation.
    """
    if transform not in ("pisa", "scta"):
        raise ValueError("transform must be pisa or scta")
    for patches in (source_patches, target_patches):
        if patches.ndim != 3 or not len(patches) or not torch.isfinite(patches).all():
            raise ValueError("Features must be nonempty finite [N,P,D] tensors")
    if source_patches.shape[-1] != target_patches.shape[-1]:
        raise ValueError("Source and target feature dimensions differ")
    head.eval()
    mu_t = target_patches.reshape(-1, target_patches.shape[-1]).mean(0)
    source_prob = head(source_patches - mu_t, source_global).softmax(-1).cpu().numpy()
    alpha = (pisa_coefficients(source_patches, target_patches, min_shift_norm_sq)
             if transform == "pisa" else torch.ones(len(target_patches), device=target_patches.device))
    target_prob = head(target_patches - alpha[:, None, None] * mu_t, target_global).softmax(-1).cpu().numpy()
    source_e, target_e = source_prob @ np.arange(3), target_prob @ np.arange(3)
    q, mask = np.asarray(source_q, dtype=float), np.asarray(source_mask, dtype=bool)
    if q.shape != (len(source_patches), 5, 3) or mask.shape != q.shape[:2]:
        raise ValueError("Source q/mask shapes must match source features")
    target_y = q[..., 1] + 2 * q[..., 2]
    coefficients = []
    for a in range(5):
        observed = mask[:, a]
        if not observed.any():
            raise ValueError("Every AU needs source training labels for OLS calibration")
        A = np.stack([source_e[observed, a], np.ones(int(observed.sum()))], 1)
        coef, *_ = np.linalg.lstsq(A, target_y[observed, a], rcond=None)
        coefficients.append(coef.tolist())
        target_e[:, a] = coef[0] * target_e[:, a] + coef[1]
    return triangular_probabilities(target_e), {"alpha": alpha.cpu().tolist(), "ols": coefficients}


def probability_ensemble(member_probabilities):
    members = np.asarray(member_probabilities)
    if members.ndim != 4 or members.shape[2:] != (5, 3) or not len(members):
        raise ValueError("Expected member probabilities [members,N,5,3]")
    if not np.isfinite(members).all() or (members < -1e-7).any() or not np.allclose(members.sum(-1), 1):
        raise ValueError("Members must provide finite probability distributions")
    return members.mean(0)
