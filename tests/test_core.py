"""Synthetic behavioral checks; no datasets, pretrained weights, or downloads."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from pisa_mgs.data import empirical_labels, load_manifest, assert_source_split
from pisa_mgs.models import AUHead, expected_score_loss
from pisa_mgs.transfer import pisa_coefficients, transfer_predict, probability_ensemble, triangular_probabilities


def test_queries_pool_each_action_and_reload(tmp_path):
    torch.manual_seed(11)
    head = AUHead(8).eval()
    patches = torch.randn(4, 7, 8)
    logits, attn = head(patches, return_attention=True)
    assert logits.shape == (4, 5, 3)
    torch.testing.assert_close(attn.sum(-1), torch.ones(4, 5))
    path = tmp_path / "head.pt"
    torch.save(head.state_dict(), path)
    loaded = AUHead(8).eval()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(loaded(patches), logits, rtol=0, atol=0)


def test_soft_expectation_masks_nan_and_has_no_missing_gradient():
    logits = torch.randn(2, 5, 3, requires_grad=True)
    q = torch.zeros(2, 5, 3)
    q[..., 0] = .5
    q[..., 2] = .5
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 1] = False
    q[0, 1] = float("nan")
    loss = expected_score_loss(logits, q, mask, kind="mse")
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))
    assert logits.grad[1].abs().sum() > 0


def test_empirical_labels_and_missing_are_distinct():
    q, mask = empirical_labels([[0, 2], [1], [], [2, 2, 1], [0]])
    assert q[0] == [.5, 0, .5] and mask[0]
    assert q[2] == [0, 0, 0] and not mask[2]
    with pytest.raises(ValueError):
        empirical_labels([[9], [], [], [], []])


def test_projection_averages_before_clipping_and_subtracts_target_mean():
    source = torch.zeros(2, 2, 1)
    target = torch.tensor([[[-2.], [2.]], [[2.], [2.]]])
    alpha = pisa_coefficients(source, target)
    # target mean=1, image0 mean=0: mean-then-clip gives 0, clip-then-mean gives .5.
    torch.testing.assert_close(alpha, torch.tensor([0., 1.]))
    with pytest.raises(ValueError, match="zero or near zero"):
        pisa_coefficients(source, source.clone())


def test_calibrated_probabilities_preserve_expected_scores():
    e = np.array([[-1., 0., .25, 1., 2.5], [.1, .9, 1.2, 1.8, 2.]])
    probs = triangular_probabilities(e)
    np.testing.assert_allclose(probs.sum(-1), 1)
    np.testing.assert_allclose(probs @ np.arange(3), np.clip(e, 0, 2))
    np.testing.assert_allclose(probability_ensemble([probs, probs]), probs)


def test_transfer_fits_only_observed_source_aus():
    torch.manual_seed(3)
    head = AUHead(8).eval()
    s, t = torch.randn(8, 4, 8), torch.randn(3, 4, 8) + 1
    q = np.tile([0., 1., 0.], (8, 5, 1))
    mask = np.ones((8, 5), dtype=bool)
    mask[0, 0] = False
    q[0, 0] = np.nan
    p, info = transfer_predict(head, s, s.mean(1), t, t.mean(1), q, mask)
    assert np.isfinite(p).all() and len(info["ols"]) == 5
    # Constant source targets yield calibrated expectation 1, independently of target predictions.
    np.testing.assert_allclose(p @ np.arange(3), 1., atol=1e-6)
    mask[:, 4] = False
    with pytest.raises(ValueError, match="Every AU"):
        transfer_predict(head, s, s.mean(1), t, t.mean(1), q, mask)


def test_unlabeled_manifest_and_animal_overlap_guard(tmp_path):
    row = {"sample_id": "t", "image_path": "faces/t.png", "roi_verified": True}
    path = tmp_path / "target.jsonl"
    path.write_text(json.dumps(row) + "\n")
    parsed = load_manifest(path)
    assert parsed[0]["mask"] == [False] * 5
    sources = [{"sample_id": "a", "animal_uid": "same", "mask": [True] * 5},
               {"sample_id": "b", "animal_uid": "same", "mask": [True] * 5}]
    with pytest.raises(ValueError, match="animal identities"):
        assert_source_split(sources, {"train": ["a"], "val": ["b"]})
