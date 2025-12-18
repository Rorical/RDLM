import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def get_model_fn(model, train=False):
    input_dim = (model.module if hasattr(model, "module") else model).input_dim
    def model_fn(x, t):
        if train:
            model.train()
        else:
            model.eval()
        return model(x[...,:input_dim], t)
    return model_fn


def get_drift_fn(model, sde, train=False, sampling=False, **kwargs):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    block_tokens = kwargs.get("block_tokens") or []
    block_tokens = [int(i) for i in block_tokens]

    pfm_cfg = kwargs.get("pfm", {})
    topk = pfm_cfg.get("topk", 0)
    topp = pfm_cfg.get("topp", 1.0)
    mc_samples = pfm_cfg.get("mc_samples", 1)
    stochastic = pfm_cfg.get("stochastic", False)
    use_inv_time = pfm_cfg.get("use_inv_time", False)
    use_scheduler_coeff = pfm_cfg.get("use_scheduler_coeff", True)

    def _truncate_probs(probs):
        out = probs
        if topk and topk < out.shape[-1]:
            top_vals, top_idx = out.topk(topk, dim=-1)
            mask = torch.zeros_like(out, dtype=torch.bool)
            mask.scatter_(-1, top_idx, True)
            out = out * mask
        if topp < 1.0:
            sorted_probs, sorted_idx = out.sort(dim=-1, descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            nucleus_mask = cum <= topp
            first_exceed = (~nucleus_mask).float().cumsum(dim=-1).eq(1)
            nucleus_mask = nucleus_mask | first_exceed
            mask = torch.zeros_like(out, dtype=torch.bool)
            mask.scatter_(-1, sorted_idx, nucleus_mask)
            out = out * mask
        out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return out

    def _mask_blocklist(probs):
        if not block_tokens:
            return probs
        vocab = probs.shape[-1]
        valid_idx = []
        for idx in block_tokens:
            if -vocab <= idx < vocab:
                valid_idx.append(idx if idx >= 0 else vocab + idx)
        if not valid_idx:
            return probs
        mask = torch.ones(vocab, device=probs.device, dtype=probs.dtype)
        mask[valid_idx] = 0
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs

    def _mc_weighted_sum(probs, x, mc):
        flat = probs.reshape(-1, probs.shape[-1])
        cat = Categorical(flat)
        samples = cat.sample((mc,))  # (mc, B*L)
        samples = samples.permute(1, 0).reshape(*probs.shape[:-1], mc)

        drifts = []
        for m in range(mc):
            onehot = F.one_hot(samples[..., m], num_classes=probs.shape[-1]).to(probs.dtype)
            drifts.append(sde.manifold.weighted_sum(onehot, x))
        return torch.stack(drifts, dim=0).mean(0)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def drift_fn(x, t):
            probs = F.softmax(model_fn(x, t).to(torch.float32), dim=-1)
            probs = torch.cat(
                [probs, torch.zeros((*probs.shape[:-1], x.shape[-1]-probs.shape[-1]), device=x.device)],
                dim=-1
            )
            probs = _mask_blocklist(probs)

            if pfm_cfg:
                probs = _truncate_probs(probs)
                if stochastic and mc_samples > 1:
                    drift = _mc_weighted_sum(probs, x, mc_samples)
                else:
                    drift = sde.manifold.weighted_sum(probs, x)
                if use_inv_time:
                    drift = drift / (1 - t).clamp_min(1e-4).view(-1, *([1] * (drift.dim()-1)))
                if use_scheduler_coeff:
                    drift = sde.scale_by_coeff(drift, t)
            else:
                drift = sde.manifold.weighted_sum(probs, x)
                drift = sde.scale_by_coeff(drift, t)
            drift = sde.manifold.to_tangent(drift, x)
            return drift

    return drift_fn
