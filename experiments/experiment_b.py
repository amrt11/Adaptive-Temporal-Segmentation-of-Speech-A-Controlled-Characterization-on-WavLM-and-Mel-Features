"""Experiment B: frozen-feature rate-distortion study (WavLM-base layer 6).

Pipeline (see docs/experiment_b_plan.md):
  1. DP-optimal K-segmentation of the frozen features, cached per utterance.
  2. Model = Experiment A segmenter + phase/duration channels + small Conv
     decoder. Warm start from DP boundaries, then unsupervised annealing.
  3. Arms: learned adaptive, uniform-warp@K (identical model class, density
     forced constant — the primary fair comparison), hard uniform@K,
     bit-matched hard uniform@K_u, DP@K, and a frame-level PCA latent at
     matched bits. Ablations: no u/d channels; boundary perturbation.

PCA is fit on train only and applied to val/test. All arms share decoder
capacity, d_z, and the reconstruction family, so D(K) and gap-closed G are
internally consistent. Distortion is per-element MSE on unpadded frames only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment_a import DensityWarpSegmenter, set_seed

FEATURE_DIR = Path("experiments/features_wavlm_l6")
BITS_PER_DIM = 16  # fixed bottleneck assumption, identical across arms


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_items(split: str, device: torch.device) -> list[dict]:
    index = json.loads((FEATURE_DIR / "index.json").read_text())["utterances"]
    items = []
    for utt_id, meta in index.items():
        if meta["split"] != split:
            continue
        feats = np.load(FEATURE_DIR / f"{utt_id}.npy").astype(np.float32)
        items.append({"id": utt_id, "features": feats, "features_full": feats,
                      "T": int(feats.shape[0])})
    items.sort(key=lambda it: it["T"])
    print(
        f"split={split}: {len(items)} utterances, "
        f"T mean={np.mean([it['T'] for it in items]):.0f} max={max(it['T'] for it in items)}",
        flush=True,
    )
    return items


def fit_pca(items: list[dict]) -> dict:
    stack = np.concatenate([it["features"] for it in items], axis=0)
    mean = stack.mean(axis=0)
    centered = stack - mean
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    _, vectors = np.linalg.eigh(cov)
    components = vectors[:, ::-1].copy()  # all H components, descending eigenvalue
    return {"mean": mean, "components": components}


def apply_pca(items: list[dict], pca: dict, pca_dim: int, device: torch.device) -> None:
    mean = pca["mean"]
    comps = pca["components"][:, :pca_dim]
    for it in items:
        f = (it["features"] - mean) @ comps
        it["features"] = torch.from_numpy(np.ascontiguousarray(f)).to(device)


def make_batches(items: list[dict], batch_size: int) -> list[list[int]]:
    return [list(range(i, min(i + batch_size, len(items)))) for i in range(0, len(items), batch_size)]


def pad_batch(items: list[dict], idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    t_max = max(items[i]["T"] for i in idxs)
    dim = items[idxs[0]]["features"].shape[1]
    device = items[idxs[0]]["features"].device
    x = torch.zeros(len(idxs), t_max, dim, device=device)
    mask = torch.zeros(len(idxs), t_max, device=device)
    for row, i in enumerate(idxs):
        t = items[i]["T"]
        x[row, :t] = items[i]["features"]
        mask[row, :t] = 1.0
    return x, mask


# --------------------------------------------------------------------------
# DP oracle
# --------------------------------------------------------------------------


def dp_bounds_for(x: np.ndarray, segments: int) -> list[int]:
    """Exact K-segmentation via prefix-sum SSE and O(K T^2) DP."""
    t, d = x.shape
    prefix = np.vstack([np.zeros((1, d)), np.cumsum(x, axis=0)])
    prefix2 = np.concatenate([[0.0], np.cumsum((x * x).sum(axis=1))])
    idx = np.arange(t + 1)
    lengths = idx[None, :] - idx[:, None]
    sums = prefix[None, :, :] - prefix[:, None, :]
    sse = prefix2[None, :] - prefix2[:, None] - (sums * sums).sum(axis=2) / np.maximum(lengths, 1)
    sse[lengths <= 0] = np.inf
    dp = np.full((segments + 1, t + 1), np.inf)
    prev = np.full((segments + 1, t + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, segments + 1):
        for end in range(k, t + 1):
            starts = np.arange(k - 1, end)
            vals = dp[k - 1, starts] + sse[starts, end]
            best = int(np.argmin(vals))
            dp[k, end] = vals[best]
            prev[k, end] = int(starts[best])
    bounds = []
    end = t
    for k in range(segments, 1, -1):
        end = int(prev[k, end])
        bounds.append(end)
    bounds.reverse()
    return bounds


def hard_segment_mse(x: np.ndarray, bounds: list[int]) -> float:
    starts = [0] + list(bounds)
    ends = list(bounds) + [x.shape[0]]
    total = 0.0
    for s, e in zip(starts, ends):
        seg = x[s:e]
        total += ((seg - seg.mean(axis=0, keepdims=True)) ** 2).sum()
    return total / x.size


def uniform_bounds(t: int, segments: int) -> list[int]:
    return [round(i * t / segments) for i in range(1, segments)]


def bit_matched_k(t: int, k: int, dz: int) -> int:
    boundary_bits = math.log2(math.comb(t, k - 1)) if k > 1 else 0.0
    return k + max(0, math.ceil(boundary_bits / (dz * BITS_PER_DIM)))


def frame_level_dims(k: int, dz: int, mean_t: float, pca_dim: int) -> int:
    """Per-frame latent dims matching K*dz total dims of content (bits cancel)."""
    return max(1, min(pca_dim, round(k * dz / mean_t)))


def frame_level_mse(items: list[dict], pca: dict, dims: int) -> float:
    """Per-frame linear latent at `dims` PCA components (orthonormal).

    Uses the original (pre-PCA) features: the baseline transmits `dims`
    components per frame, which is exactly the residual below the top-`dims`
    principal subspace.
    """
    comps = pca["components"][:, :dims]
    total = 0.0
    total_elems = 0
    for it in items:
        f = it["features_full"]
        centered = f - pca["mean"]
        energy = (centered * centered).sum(axis=1)
        captured = ((centered @ comps) ** 2).sum(axis=1)
        total += float((energy - captured).sum())
        total_elems += f.shape[0] * f.shape[1]
    return total / total_elems


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class EMAVQ(nn.Module):
    """EMA vector quantizer with straight-through estimator.

    Codes are row indices into a shared codebook; commitment loss is returned
    separately. Tracks EMA code usage for perplexity-based bit accounting.
    """

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.95, eps: float = 1e-5,
                 dead_code_reset: bool = False, reset_every: int = 16):
        super().__init__()
        self.codebook_size = codebook_size
        self.dead_code_reset = dead_code_reset
        self.reset_every = reset_every
        self.reset_counter = 0
        self.register_buffer("codebook", torch.randn(codebook_size, dim) * 0.02)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_weight_sum", torch.zeros(codebook_size, dim))
        self.register_buffer("inited", torch.tensor(False))
        self.decay = decay
        self.eps = eps

    def forward(self, flat: torch.Tensor):
        """flat: [N, dim] -> (quantized, codes, vq_loss, perplexity)."""
        flat = flat.detach()
        if not self.inited.item():
            # First call: sample data points as initial codes (with replacement
            # when a single call provides fewer tokens than the codebook).
            n = flat.shape[0]
            if n >= self.codebook_size:
                idx = torch.randperm(n, device=flat.device)[: self.codebook_size]
            else:
                idx = torch.randint(n, (self.codebook_size,), device=flat.device)
            self.codebook.copy_(flat[idx])
            self.ema_cluster_size.fill_(1.0)
            self.ema_weight_sum.zero_()
            self.inited.fill_(True)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.T
                + self.codebook.pow(2).sum(1)[None, :])
        codes = dist.argmin(dim=1)
        qt = self.codebook[codes]
        if self.training:
            onehot = nn.functional.one_hot(codes, self.codebook_size).type(flat.dtype)
            counts = onehot.sum(0)
            embed_sum = onehot.T @ flat
            self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.ema_weight_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster = ((self.ema_cluster_size + self.eps)
                       / (n + self.codebook_size * self.eps)
                       * n).unsqueeze(1)
            self.codebook.copy_(self.ema_weight_sum / cluster.clamp_min(1e-6))
            if self.dead_code_reset:
                self.reset_counter += 1
                if self.reset_counter % self.reset_every == 0:
                    usage = self.ema_cluster_size
                    dead = usage < 1.0
                    n_dead = int(dead.sum().item())
                    if n_dead > 0:
                        n = flat.shape[0]
                        ridx = torch.randint(n, (n_dead,), device=flat.device)
                        self.codebook[dead] = flat[ridx]
                        self.ema_cluster_size[dead] = 1.0
                        self.ema_weight_sum[dead] = flat[ridx]
        vq_loss = nn.functional.mse_loss(qt, flat.detach()) \
            + 0.25 * nn.functional.mse_loss(flat, qt.detach())
        qt_st = flat + (qt - flat).detach()
        counts = torch.bincount(codes, minlength=self.codebook_size).float()
        p = counts / counts.sum().clamp_min(1)
        p = p[p > 0]
        perplexity = float(torch.exp(-(p * p.log()).sum()).item())
        return qt_st, codes, vq_loss, perplexity


class PhaseDurationSegmenter(DensityWarpSegmenter):
    """Experiment A segmenter + phase/duration channels, pooled latents, decoder.

    h_t = [ sum_k m_k(t) z_k ; u(t) ; log d(t) ]
      u(t) = frac(C(t))          (within-segment warped phase)
      d(t) = (sum(rho)/K) / rho_t  (local frames-per-segment estimate)
    z_k = g( sum_t m_k(t) P F_t / sum_t m_k(t) ),  P a linear projection
    F_hat = small bidirectional ConvDecoder(h)
    """

    def __init__(self, dims: int, segments: int, latent_dim: int = 32, hidden: int = 96,
                 decoder_hidden: int = 192, decoder_layers: int = 2,
                 rho_min: float = 0.05, tau: float = 0.15,
                 use_phase: bool = True, use_duration: bool = True,
                 fixed_uniform: bool = False, codebook_size: int = 0,
                 timing_mode: str = "density"):
        super().__init__(dims, segments, rho_min=rho_min, tau=tau)
        self.latent_dim = latent_dim
        self.timing_mode = timing_mode
        if timing_mode == "none":
            use_phase = use_duration = False
        elif timing_mode == "billed":
            # Timing derived from the hard partition only (b_k): u is a linear
            # ramp inside each hard segment, d is the hard segment length.
            # Both are functions of the transmitted boundaries, hence covered
            # by the log2 C(T, K-1) side-information charge - no per-frame
            # feature information.
            use_phase = use_duration = True
        self.use_phase = use_phase
        self.use_duration = use_duration
        # Uniform-warp arm: identical model/decoder, density forced constant.
        # Equal-width segments; all capacity/parameters/channels unchanged.
        self.fixed_uniform = fixed_uniform
        self.vq = EMAVQ(latent_dim, codebook_size) if codebook_size > 0 else None
        extra = int(use_phase) + int(use_duration)
        self.pool_proj = nn.Linear(dims, latent_dim)
        self.latent_gate = nn.Linear(latent_dim, latent_dim)
        self.density = nn.Sequential(
            nn.Conv1d(dims, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        decoder_in = latent_dim + extra
        layers: list[nn.Module] = []
        for layer in range(decoder_layers):
            layers.append(nn.Conv1d(decoder_in if layer == 0 else decoder_hidden,
                                    decoder_hidden, kernel_size=7, padding=3))
            layers.append(nn.GELU())
        layers.append(nn.Conv1d(decoder_hidden, dims, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward_full(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Returns recon [B,T,D], masks [B,K,T], edge_c [B,T+1], z [B,K,dz]."""
        b, t, _ = x.shape
        if self.fixed_uniform:
            rho = self.rho_min + torch.ones(b, t, device=x.device)
        else:
            delta = torch.zeros_like(x)
            if t > 1:
                delta[:, 1:] = x[:, 1:] - x[:, :-1]
            rho = self.rho_min + nn.functional.softplus(
                self.density((x + 0.1 * delta.abs()).transpose(1, 2)).squeeze(1)
            )
        cumulative = torch.cumsum(torch.cat([torch.zeros_like(rho[:, :1]), rho], dim=1), dim=1)
        total = cumulative[:, -1:].clamp_min(1e-6)
        edge_c = self.segments * cumulative / total
        u = 0.5 * (edge_c[:, :-1] + edge_c[:, 1:])
        transitions = [torch.ones_like(u[:, :1])]
        for k in range(1, self.segments):
            transitions.append(torch.sigmoid((u - float(k)) / self.tau))
        transitions.append(torch.zeros_like(u[:, :1]))
        masks = torch.stack([transitions[k] - transitions[k + 1] for k in range(self.segments)], dim=1)

        if mask is not None:
            m = mask.unsqueeze(1)
            mass = (masks * m).sum(dim=2, keepdim=True).clamp_min(1e-6)
            pooled = torch.einsum("bkt,btd->bkd", masks * m, x) / mass
        else:
            mass = masks.sum(dim=2, keepdim=True).clamp_min(1e-6)
            pooled = torch.einsum("bkt,btd->bkd", masks, x) / mass
        z = self.latent_gate(self.pool_proj(pooled))
        vq_out = None
        if self.vq is not None:
            zb, seg, dz = z.shape
            zq, codes, vq_loss, perplexity = self.vq(z.reshape(zb * seg, dz))
            z = zq.view(zb, seg, dz)
            vq_out = {"codes": codes.view(zb, seg), "vq_loss": vq_loss, "perplexity": perplexity}

        channels = [torch.einsum("bkt,bkd->btd", masks, z)]
        if self.use_phase and self.timing_mode == "billed":
            # Boundary-derived timing: per-frame hard segment id, then a linear
            # ramp within the segment (u_k) and the segment length (d_k).
            hard_idx = torch.argmax(masks, dim=1)  # [B,T]
            b_size = hard_idx.shape[0]
            ar = torch.arange(t, device=x.device).float()[None, :].expand(b_size, -1)
            is_start = torch.cat([torch.ones_like(ar[:, :1]),
                                  (hard_idx[:, 1:] != hard_idx[:, :-1]).float()], dim=1)
            start_pos = torch.cummax(ar * is_start - (1 - is_start) * 1e9, dim=1).values
            is_end = torch.cat([(hard_idx[:, 1:] != hard_idx[:, :-1]).float(),
                                torch.ones_like(ar[:, :1])], dim=1)
            end_pos = torch.flip(torch.cummin(
                torch.flip(ar * is_end + (1 - is_end) * 1e9, [1]), dim=1).values, [1])
            seg_len = (end_pos - start_pos).clamp_min(1.0)
            u_k = (ar - start_pos) / seg_len  # linear ramp in [0, 1)
            channels.append(u_k.unsqueeze(-1))
        elif self.use_phase:
            channels.append(u.frac().unsqueeze(-1))
        if self.use_duration and self.timing_mode == "billed":
            channels.append(torch.log(seg_len).unsqueeze(-1))
        elif self.use_duration:
            # Local frames-per-segment: total frames per warped unit / local rho.
            frames_per_unit = total.detach() / self.segments  # [B,1]
            duration = frames_per_unit / rho.clamp_min(1e-6)  # [B,T]
            channels.append(torch.log(duration).unsqueeze(-1))
        h = torch.cat(channels, dim=-1)
        recon = self.decoder(h.transpose(1, 2)).transpose(1, 2)
        return recon, masks, edge_c, z, vq_out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def warp_boundaries(edge_c_row: torch.Tensor, t: int, segments: int) -> list[int]:
    """Hard boundaries from the warp: b_k = argmin_t |C(t) - k|.

    Far more accurate than cumulative-mass crossings when masks are soft,
    especially at large K where segments are a few frames wide.
    """
    c = edge_c_row[:t]
    return [int(torch.argmin((c - float(k)).abs()).item()) for k in range(1, segments)]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (pred - target) ** 2 * mask.unsqueeze(-1)
    return float((diff.sum() / mask.sum().clamp_min(1) / target.shape[-1]).item())


def evaluate_split(model: PhaseDurationSegmenter, items: list[dict],
                   batches: list[list[int]], segments: int, pca: dict) -> dict:
    model.eval()
    learned_weighted = 0.0
    learned_hard_weighted = 0.0
    uniform_weighted = 0.0
    dp_weighted = 0.0
    frames = 0
    partition_err = 0.0
    perplexity_sum = 0.0
    perplexity_count = 0
    min_mass = 1.0
    density_cv: list[float] = []
    with torch.no_grad():
        for idxs in batches:
            x, mask = pad_batch(items, idxs)
            recon, masks, edge_c, _, vq_out = model.forward_full(x, mask)
            learned_weighted += masked_mse(recon, x, mask) * float(mask.sum())
            if vq_out is not None:
                perplexity_sum += float(vq_out["perplexity"])
                perplexity_count += 1
            frames += int(mask.sum())
            partition_err = max(partition_err, float(((masks.sum(dim=1) - 1).abs() * mask).max().item()))
            mass = (masks * mask.unsqueeze(1)).sum(dim=2)  # [B,K]
            min_mass = min(min_mass, float((mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-6)).min().item()))
            delta = torch.zeros_like(x)
            if x.shape[1] > 1:
                delta[:, 1:] = x[:, 1:] - x[:, :-1]
            rho = model.rho_min + nn.functional.softplus(
                model.density((x + 0.1 * delta.abs()).transpose(1, 2)).squeeze(1)
            )
            for row, i in enumerate(idxs):
                t = items[i]["T"]
                r = rho[row, :t]
                density_cv.append(float((r.std() / r.mean().clamp_min(1e-8)).item()))
                feats_np = items[i]["features"].detach().cpu().numpy()
                uniform_weighted += hard_segment_mse(feats_np, uniform_bounds(t, segments)) * t
                dp_weighted += hard_segment_mse(feats_np, items[i]["dp_bounds"][segments]) * t
                pred_b = warp_boundaries(edge_c[row], t, segments)
                learned_hard_weighted += hard_segment_mse(feats_np, pred_b) * t
    total_frames = sum(it["T"] for it in items)
    dim = items[0]["features"].shape[1]
    mean_t = total_frames / len(items)
    ku = bit_matched_k(int(round(mean_t)), segments, model.latent_dim)
    ku_weighted = sum(
        hard_segment_mse(it["features"].detach().cpu().numpy(),
                         uniform_bounds(it["T"], bit_matched_k(it["T"], segments, model.latent_dim))) * it["T"]
        for it in items
    )
    fl_dims = frame_level_dims(segments, model.latent_dim, mean_t, dim)
    out = {
        "learned_mse": learned_weighted / frames,
        "learned_hard_mse": learned_hard_weighted / total_frames,
        "uniform_mse": uniform_weighted / total_frames,
        "bit_matched_k_u": ku,
        "bit_matched_uniform_mse": ku_weighted / total_frames,
        "dp_mse": dp_weighted / total_frames,
        "frame_level_dims": fl_dims,
        "frame_level_mse": frame_level_mse(items, pca, fl_dims),
        "partition_max_error": partition_err,
        "min_segment_mass": min_mass,
        "density_cv_mean": float(np.mean(density_cv)),
        "vq_perplexity_mean": (perplexity_sum / perplexity_count) if perplexity_count else None,
    }
    gap = out["uniform_mse"] - out["dp_mse"]
    out["gap_closed_hard"] = ((out["uniform_mse"] - out["learned_hard_mse"]) / gap
                              if gap > 1e-12 else None)
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def arm_tag(use_phase: bool, use_duration: bool) -> str:
    parts = []
    if use_phase:
        parts.append("u")
    if use_duration:
        parts.append("d")
    return "+".join(parts) if parts else "noud"


def train_arm(items: list[dict], batches: list[list[int]], eval_items: list[dict],
              eval_batches: list[list[int]], segments: int, args: argparse.Namespace,
              pca: dict, use_phase: bool = True, use_duration: bool = True,
              fixed_uniform: bool = False) -> tuple[PhaseDurationSegmenter, dict]:
    device = items[0]["features"].device
    dim = items[0]["features"].shape[1]
    model = PhaseDurationSegmenter(
        dim, segments, latent_dim=args.latent_dim,
        use_phase=use_phase, use_duration=use_duration, tau=args.tau_start,
        fixed_uniform=fixed_uniform,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tag = ("uniform" if fixed_uniform else arm_tag(use_phase, use_duration))
    history = []
    step = 0
    t0 = time.time()
    for stage, n_steps in (("warmup", args.warmup_steps), ("anneal", args.anneal_steps)):
        for s in range(1, n_steps + 1):
            step += 1
            model.train()
            if stage == "anneal" and n_steps > 0:
                progress = s / n_steps
                model.tau = args.tau_start * (args.tau_end / args.tau_start) ** progress
            idxs = list(random.choice(batches))
            x, mask = pad_batch(items, idxs)
            recon, masks, edge_c, _, vq_out = model.forward_full(x, mask)
            diff = (recon - x) ** 2 * mask.unsqueeze(-1)
            recon_loss = diff.sum() / mask.sum().clamp_min(1) / x.shape[-1]
            loss = recon_loss
            if vq_out is not None:
                loss = loss + args.vq_weight * vq_out["vq_loss"]
            if stage == "warmup" and not fixed_uniform:
                targets = torch.arange(1, segments, device=device).float()[None, :].expand(x.shape[0], -1)
                rows = []
                for r, i in enumerate(idxs):
                    bounds = items[i]["dp_bounds"][segments]
                    rows.append(edge_c[r, torch.tensor(bounds, device=device, dtype=torch.long)])
                c_at_bounds = torch.stack(rows)
                boundary_loss = ((c_at_bounds - targets) ** 2).mean()
                loss = loss + args.boundary_weight * boundary_loss
            entropy = -(masks.clamp_min(1e-8) * masks.clamp_min(1e-8).log())
            entropy = (entropy * mask.unsqueeze(1)).sum() / (mask.sum() * segments)
            loss = loss + args.entropy_weight * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.eval_every == 0 or step == args.warmup_steps + args.anneal_steps:
                metrics = evaluate_split(model, eval_items, eval_batches, segments, pca)
                metrics.update({"stage": stage, "step": step, "tau": model.tau,
                                "train_recon": float(recon_loss.item()),
                                "elapsed_s": round(time.time() - t0, 1)})
                history.append(metrics)
                g = metrics.get("gap_closed_hard")
                print(f"[K={segments} {tag} {stage} {step:5d}] soft={metrics['learned_mse']:.5f} "
                      f"hard={metrics['learned_hard_mse']:.5f} "
                      f"Ghard={'n/a' if g is None else round(g, 3)} tau={model.tau:.3f} "
                      f"train={float(recon_loss.item()):.5f} t={metrics['elapsed_s']}s", flush=True)
    final = evaluate_split(model, eval_items, eval_batches, segments, pca)
    return model, {"final": final, "history": history}


def perturbation_test(model: PhaseDurationSegmenter, items: list[dict], segments: int,
                      deltas: list[int]) -> dict:
    """Move each learned boundary; hard re-pooling; cost by distance to DP boundary."""
    records = []
    base_costs: list[float] = []
    with torch.no_grad():
        for it in items[:32]:
            feats_np = it["features"].detach().cpu().numpy()
            t = it["T"]
            feats = it["features"].unsqueeze(0)
            mask = torch.ones(1, t, device=feats.device)
            _, masks, edge_c, _, _ = model.forward_full(feats, mask)
            pred_b = warp_boundaries(edge_c[0], t, segments)
            dp_b = it["dp_bounds"][segments]
            base = hard_segment_mse(feats_np, pred_b)
            base_costs.append(base)
            for k in range(segments - 1):
                lo = (pred_b[k - 1] + 1) if k > 0 else 1
                hi = (pred_b[k + 1] - 1) if k < segments - 2 else t - 1
                for delta in deltas:
                    moved = int(np.clip(pred_b[k] + delta, lo, hi))
                    if moved == pred_b[k]:
                        continue
                    moved_b = list(pred_b)
                    moved_b[k] = moved
                    cost = hard_segment_mse(feats_np, moved_b) - base
                    records.append({
                        "distance_to_dp": int(abs(moved - dp_b[k])) if k < len(dp_b) else -1,
                        "delta_mse": round(float(cost), 8),
                    })
    buckets: dict[str, list[float]] = {}
    for r in records:
        d = r["distance_to_dp"]
        key = "0-1" if d <= 1 else "2-4" if d <= 4 else "5-8" if d <= 8 else "9+"
        buckets.setdefault(key, []).append(r["delta_mse"])
    return {
        "n_utts": min(32, len(items)),
        "hard_mse_at_learned": float(np.mean(base_costs)),
        "hard_mse_at_dp": float(np.mean([
            hard_segment_mse(it["features"].detach().cpu().numpy(), it["dp_bounds"][segments]) for it in items[:32]
        ])),
        "cost_by_distance_to_dp": {
            b: {"mean": float(np.mean(v)), "count": len(v)} for b, v in sorted(buckets.items())
        },
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B training/eval")
    parser.add_argument("--ks", type=str, default="8,16,32,64")
    parser.add_argument("--only-ks", type=str, default="")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--anneal-steps", type=int, default=1700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tau-start", type=float, default=0.35)
    parser.add_argument("--tau-end", type=float, default=0.08)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--deltas", type=str, default="-16,-4,4,16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("experiments/results_experiment_b.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"device={device}", flush=True)

    train_items = load_items("train", device)
    pca = fit_pca(train_items)
    val_items = load_items("val", device)
    test_items = load_items("test", device)
    for split_items in (train_items, val_items, test_items):
        apply_pca(split_items, pca, args.pca_dim, device)
    eval_items = test_items[: args.eval_size]
    eval_batches = make_batches(eval_items, args.batch_size)

    ks = [int(k) for k in (args.only_ks or args.ks).split(",")]

    # DP oracle, cached per K (boundaries are on the PCA-reduced features).
    dp_cache_path = FEATURE_DIR / f"dp_bounds_pca{args.pca_dim}.json"
    dp_store = json.loads(dp_cache_path.read_text()) if dp_cache_path.exists() else {}
    all_items = train_items + val_items + test_items
    for it in all_items:
        it["dp_bounds"] = {}
    for k in sorted(set(ks)):
        key = str(k)
        if key not in dp_store:
            print(f"computing DP bounds for K={k} on {len(all_items)} utterances...", flush=True)
            t0 = time.time()
            bounds: dict[str, list[int]] = {}
            for it in all_items:
                b = dp_bounds_for(it["features"].detach().cpu().numpy(), k)
                it["dp_bounds"][k] = b
                bounds[it["id"]] = b
            dp_store[key] = bounds
            dp_cache_path.write_text(json.dumps(dp_store), encoding="utf-8")
            print(f"DP K={k} done in {time.time() - t0:.0f}s", flush=True)
        else:
            stored = dp_store[key]
            for it in all_items:
                it["dp_bounds"][k] = stored[it["id"]]
            print(f"loaded DP bounds K={k} from cache", flush=True)

    results: dict = {"config": {**vars(args), "output": str(args.output)}, "arms": {}}
    deltas = [int(d) for d in args.deltas.split(",")]

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    for k in ks:
        print(f"=== K={k} ===", flush=True)
        batches = make_batches(train_items, args.batch_size)
        model, out = train_arm(train_items, batches, eval_items, eval_batches, k, args, pca)
        results["arms"][f"K{k}"] = out
        save()
        model_u, out_u = train_arm(train_items, batches, eval_items, eval_batches, k, args, pca,
                                   fixed_uniform=True)
        results["arms"][f"K{k}_uniform_warp"] = out_u
        save()
        if k == 32:
            model_noud, out_noud = train_arm(train_items, batches, eval_items, eval_batches, k, args,
                                             pca, use_phase=False, use_duration=False)
            results["arms"][f"K{k}_no_ud"] = out_noud
            save()
            model_u_noud, out_u_noud = train_arm(train_items, batches, eval_items, eval_batches, k,
                                                 args, pca, use_phase=False, use_duration=False,
                                                 fixed_uniform=True)
            results["arms"][f"K{k}_uniform_warp_no_ud"] = out_u_noud
            save()
            results["perturbation"] = {
                "with_ud": perturbation_test(model, eval_items, k, deltas),
                "without_ud": perturbation_test(model_noud, eval_items, k, deltas),
            }
            save()
            print("perturbation tests saved", flush=True)

    print("\n=== summary (test) ===", flush=True)
    for arm, out in results["arms"].items():
        f = out["final"]
        g = f.get("gap_closed_hard")
        print(f"{arm:22s} soft={f['learned_mse']:.5f} hard={f['learned_hard_mse']:.5f} "
              f"uwarp={f['uniform_mse']:.5f} uniform@Ku={f['bit_matched_uniform_mse']:.5f} "
              f"dp={f['dp_mse']:.5f} frame={f['frame_level_mse']:.5f} "
              f"Ghard={'n/a' if g is None else round(g, 3)}", flush=True)
    if "perturbation" in results:
        for cond, p in results["perturbation"].items():
            print(f"perturb[{cond}]: {json.dumps(p['cost_by_distance_to_dp'], default=str)}", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
