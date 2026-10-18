"""Source-only head fitting with the historical member-specific training recipes."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from .models import AUHead, ordinal_loss, expected_score_loss
from .metrics import evaluate
from .utils import seed_all

def effective_batch(config, n):
    value = config["base"]["effective_batch"]
    if value == "auto":
        return 16 if n <= 80 else 32 if n <= 200 else 64
    if int(value) < 1:
        raise ValueError("effective_batch must be positive")
    return int(value)

class _EMA:
    """Exponential moving average of model weights (validation/deploy weights)."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError("ema_decay must be in (0,1)")
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                shadow.copy_(v)

    def swap_into(self, model: nn.Module):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def state(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}

def balanced_draws(records, ids, n, seed):
    """Uniform source -> uniform animal -> uniform image, with replacement."""
    lookup = {r["sample_id"]: r for r in records}
    groups = defaultdict(lambda: defaultdict(list))
    for sid in sorted(ids):
        r = lookup[sid]
        if r.get("animal_uid") is None:
            raise ValueError("Training requires verified animal IDs")
        groups[r["source"]][r["animal_uid"]].append(sid)
    if not groups:
        raise ValueError("Empty training partition")
    sources = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        source = sources[int(rng.integers(len(sources)))]
        animals = sorted(groups[source])
        animal = animals[int(rng.integers(len(animals)))]
        images = groups[source][animal]
        draws.append(images[int(rng.integers(len(images)))])
    return draws, rng

def _targets(lookup, ids, device):
    q = torch.tensor([lookup[i]["q"] for i in ids], dtype=torch.float32, device=device)
    mask = torch.tensor([lookup[i]["mask"] for i in ids], dtype=torch.bool, device=device)
    if not mask.any(dim=1).all():
        raise ValueError("A training/evaluation label record has no valid AU")
    return q, mask

def _scheduler(optimizer, cfg):
    def factor(epoch):
        warmup = cfg["warmup"]
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, cfg["max_epochs"] - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

def assert_partitions(records, train_ids, val_ids):
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("Train and validation must be nonempty and disjoint")
    lookup = {r["sample_id"]: r for r in records}
    ta = {lookup[i]["animal_uid"] for i in train_ids}
    va = {lookup[i]["animal_uid"] for i in val_ids}
    if None in ta or None in va or ta & va:
        raise ValueError("Train/validation animal overlap or unknown identity")

@torch.no_grad()
def base_logits(head, bank, ids, device, batch_size=128, view=0):
    head.eval()
    outputs, globals_ = [], []
    for start in range(0, len(ids), batch_size):
        features = [bank.get(sid, view) for sid in ids[start:start + batch_size]]
        p = torch.stack([x[0] for x in features]).to(device)
        g = torch.stack([x[1] for x in features]).to(device)
        outputs.append(head(p, g).float().cpu())
        globals_.append(g.cpu())
    if not outputs:
        return torch.empty((0, 5, 3)), torch.empty((0, bank.global_dim))
    return torch.cat(outputs), torch.cat(globals_)

def probabilities_to_rows(ids, prob):
    return [{"sample_id": sid, "prob": p.tolist()} for sid, p in zip(ids, prob.cpu())]

def train_convnext(records_list, cache, tr_ids, va_ids, lr, seed, out_dir=None, extra_ids=(), epochs=100):
    """Custom constant-lr loop (frozen recipe), EMA 0.99, best on source val."""
    bank = cache
    Ptr = torch.stack([bank.get(i, 0)[0] for i in tr_ids])
    Gtr = torch.stack([bank.get(i, 0)[1] for i in tr_ids])
    Pva = torch.stack([bank.get(i, 0)[0] for i in va_ids])
    Gva = torch.stack([bank.get(i, 0)[1] for i in va_ids])
    records = {r["sample_id"]: r for r in records_list}
    Y = torch.tensor(np.stack([[records[i]["q"][a][1] + 2 * records[i]["q"][a][2]
                                for a in range(5)] for i in tr_ids]), dtype=torch.float32)
    M = torch.tensor(np.stack([records[i]["mask"] for i in tr_ids]), dtype=torch.bool)
    vq = torch.tensor([records[i]["q"] for i in va_ids], dtype=torch.float32)
    vm = torch.tensor([records[i]["mask"] for i in va_ids], dtype=torch.bool)
    ve_t = vq[..., 1] + 2 * vq[..., 2]
    torch.manual_seed(seed)
    head = AUHead(bank.patch_dim, hidden=128, dropout=0.1)
    ema = {k: v.clone().float() for k, v in head.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_ema = float("inf"), None
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(tr_ids))
        for st in range(0, len(order), 32):
            pos = torch.tensor(order[st:st + 32])
            opt.zero_grad(set_to_none=True)
            pr = head(Ptr[pos], Gtr[pos]).softmax(-1)
            exp = pr[..., 1] + 2 * pr[..., 2]
            mm = M[pos].float()
            loss = (((exp - Y[pos]) ** 2) * mm).sum() / mm.sum().clamp_min(1)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for k, v in head.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.99).add_(v.float(), alpha=0.01)
                    else:
                        ema[k].copy_(v)
        head.eval()
        with torch.no_grad():
            backup = {k: v.clone() for k, v in head.state_dict().items()}
            head.load_state_dict({k: v for k, v in ema.items()})
            vpr = head(Pva, Gva).softmax(-1)
            ve = vpr[..., 1] + 2 * vpr[..., 2]
            mm = vm.float()
            vmae = float((((ve - ve_t).abs()) * mm).sum() / mm.sum())
            head.load_state_dict(backup)
        if vmae < best - 1e-10:
            best, best_ema = vmae, {k: v.clone() for k, v in ema.items()}
    if best_ema is None:
        raise ValueError("No finite validation checkpoint")
    head.load_state_dict(best_ema)
    head.eval()
    return head, bank


def train_protocol(records, bank, train_ids, val_ids, recipe, seed=2026):
    """CPU protocol path, stripped of research alternatives and resume machinery.

    Preserves balanced draws, cached-view selection, effective-batch accumulation,
    warmup/cosine, clipping, validation metric, and the original EMA update count.
    """
    assert_partitions(records, train_ids, val_ids)
    seed_all(seed)
    device = torch.device("cpu")
    head = AUHead(bank.patch_dim, hidden=128, dropout=0.1)
    cfg = dict(recipe)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = _scheduler(optimizer, cfg)
    ema = _EMA(head, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    lookup = {r["sample_id"]: r for r in records}
    eff = effective_batch({"base": {"effective_batch": "auto"}}, len(train_ids))
    physical = min(32, eff)
    best, best_state, bad, logs = float("inf"), None, 0, []
    for epoch in range(cfg["max_epochs"]):
        head.train()
        draws, view_rng = balanced_draws(records, train_ids, len(train_ids), seed + epoch * 7919)
        losses = []
        for start in range(0, len(draws), eff):
            big_ids = draws[start:start + eff]
            optimizer.zero_grad(set_to_none=True)
            for pos in range(0, len(big_ids), physical):
                ids = big_ids[pos:pos + physical]
                cached = [bank.get(sid, int(view_rng.integers(bank.views(sid)))) for sid in ids]
                p = torch.stack([x[0] for x in cached])
                g = torch.stack([x[1] for x in cached])
                q, mask = _targets(lookup, ids, device)
                logits = head(p, g)
                if cfg["loss"] == "mse":
                    per_image = expected_score_loss(logits, q, mask, kind="mse", reduction="none")
                elif cfg["loss"] == "rps_ce":
                    per_image = ordinal_loss(logits, q, mask, ce_weight=0.1, reduction="none")
                else:
                    raise ValueError("Only the two executed member losses are supported")
                loss = g.sum() * 0.0 + per_image.sum() / len(big_ids)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite source training loss")
                loss.backward()
                losses.extend(per_image.detach().tolist())
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                # Historical train_base path really updates twice per step.
                ema.update(head)
                ema.update(head)
        if ema is not None:
            backup = ema.swap_into(head)
        with torch.no_grad():
            logits, _ = base_logits(head, bank, val_ids, device)
            validation = evaluate([lookup[i] for i in val_ids],
                                  probabilities_to_rows(val_ids, logits.softmax(-1)))
        val = validation["checkpoint_available_au_mae"]
        if val is None or not math.isfinite(val):
            raise ValueError("Validation MAE undefined; verify source identities and labels")
        if val < best - 1e-10:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if ema is not None:
            head.load_state_dict(backup)
        scheduler.step()
        logs.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                     "val_checkpoint_mae": val})
        if epoch + 1 >= cfg["min_epochs"] and bad >= cfg["patience"]:
            break
    if best_state is None:
        raise ValueError("No finite validation checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_val_checkpoint_mae": best, "history": logs}
