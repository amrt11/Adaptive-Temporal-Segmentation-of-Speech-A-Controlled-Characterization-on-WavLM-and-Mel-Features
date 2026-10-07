"""Experiment B2: quantized rate-distortion comparison.

Frame-VQ vs Uniform segment-VQ vs Adaptive segment-VQ at matched real bits.

All arms share the same decoder capacity, latent_dim, and training recipe;
only the bottleneck structure differs. Every run gets a fresh EMA codebook.

Arms (per K and codebook size N):
  frame    - one latent per 50 Hz frame, quantized (frame rate = 50/s)
  uniform  - exact-partition segmenter, density forced constant (K segments)
  adaptive - learned density-warp segmenter (DP warm start, then anneal)

Per-utterance bit accounting (exact, plus a perplexity-adjusted variant):
  frame    - T * d * log2(N)                    (d = latent_dim)
  uniform  - K * d * log2(N)
  adaptive - K * d * log2(N) + log2 C(T, K-1)   (boundary side information)

Output: experiments/results_experiment_b_vq.json, one entry per arm point.
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

from experiment_a import set_seed
from experiment_b import (
    FEATURE_DIR,
    EMAVQ,
    PhaseDurationSegmenter,
    apply_pca,
    fit_pca,
    load_items,
    make_batches,
    masked_mse,
    pad_batch,
)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class FrameVQ(nn.Module):
    """Per-frame latent bottleneck: same latent_dim and decoder as the
    segmenter arms, so the only difference is the bottleneck structure."""

    def __init__(self, dims: int, latent_dim: int = 32, codebook_size: int = 256,
                 decoder_hidden: int = 192, decoder_layers: int = 2,
                 dead_code_reset: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.proj = nn.Linear(dims, latent_dim)
        self.vq = EMAVQ(latent_dim, codebook_size, dead_code_reset=dead_code_reset)
        layers: list[nn.Module] = []
        for layer in range(decoder_layers):
            layers.append(nn.Conv1d(latent_dim if layer == 0 else decoder_hidden,
                                    decoder_hidden, kernel_size=7, padding=3))
            layers.append(nn.GELU())
        layers.append(nn.Conv1d(decoder_hidden, dims, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        b, t, _ = x.shape
        z = self.proj(x)  # [B,T,dz]
        zb, td, dz = z.shape
        zq, codes, vq_loss, perplexity = self.vq(z.reshape(zb * td, dz))
        zq = zq.view(zb, td, dz)
        recon = self.decoder(zq.transpose(1, 2)).transpose(1, 2)
        return recon, {"codes": codes.view(zb, td), "vq_loss": vq_loss,
                       "perplexity": perplexity}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def prepare_data(pca_dim: int, device: torch.device):
    train_items = load_items("train", device)
    pca = fit_pca(train_items)
    val_items = load_items("val", device)
    test_items = load_items("test", device)
    for split_items in (train_items, val_items, test_items):
        apply_pca(split_items, pca, pca_dim, device)
    return train_items, test_items, pca


def eval_test_mse(model, eval_items, eval_batches) -> tuple[float, float]:
    model.eval()
    weighted = 0.0
    frames = 0
    perplexity_sum = 0.0
    perplexity_count = 0
    with torch.no_grad():
        for idxs in eval_batches:
            x, mask = pad_batch(eval_items, idxs)
            if hasattr(model, "forward_full"):
                recon, masks, edge_c, z, vq_out = model.forward_full(x, mask)
            else:
                recon, vq_out = model(x, mask)
            weighted += masked_mse(recon, x, mask) * float(mask.sum())
            frames += int(mask.sum())
            perplexity_sum += float(vq_out["perplexity"])
            perplexity_count += 1
    return weighted / frames, perplexity_sum / max(perplexity_count, 1)


# --------------------------------------------------------------------------


def utt_bits(T: int, arm: str, K: int, codebook: int, dims: int) -> tuple[float, float]:
    """(exact bits incl. all side info, content-only bits) per utterance.

    A single-codebook VQ transmits ONE index per token: log2(N) bits per
    segment (segment arms) or per frame (frame arm). The vector values behind
    an index are decoder-side and cost nothing on the wire.
    """
    logn = math.log2(codebook)
    if arm == "frame":
        content = T * logn
        return content, content
    if arm == "uniform":
        content = K * logn
        return content, content
    if arm == "adaptive":
        content = K * logn
        boundaries = math.log2(math.comb(T, K - 1)) if K > 1 else 0.0
        return content + boundaries, content
    raise ValueError(arm)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train_frame_vq(model: FrameVQ, train_items, batches, eval_items, eval_batches,
                   args) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for _ in range(args.warmup_steps + args.anneal_steps):
        model.train()
        idxs = list(random.choice(batches))
        x, mask = pad_batch(train_items, idxs)
        recon, vq_out = model(x, mask)
        diff = (recon - x) ** 2 * mask.unsqueeze(-1)
        recon_loss = diff.sum() / mask.sum().clamp_min(1) / x.shape[-1]
        loss = recon_loss + args.vq_weight * vq_out["vq_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    mse, perplexity = eval_test_mse(model, eval_items, eval_batches)
    return {"mse": mse, "perplexity": perplexity,
            "train_seconds": round(time.time() - t0, 1)}


def train_segmenter_vq(model: PhaseDurationSegmenter, train_items, batches,
                       eval_items, eval_batches, args, segments: int,
                       fixed_uniform: bool) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for stage, n_steps in (("warmup", args.warmup_steps), ("anneal", args.anneal_steps)):
        for s in range(1, n_steps + 1):
            model.train()
            if stage == "anneal" and n_steps > 0 and not fixed_uniform:
                progress = s / n_steps
                model.tau = args.tau_start * (args.tau_end / args.tau_start) ** progress
            idxs = list(random.choice(batches))
            x, mask = pad_batch(train_items, idxs)
            recon, masks, edge_c, _, vq_out = model.forward_full(x, mask)
            diff = (recon - x) ** 2 * mask.unsqueeze(-1)
            recon_loss = diff.sum() / mask.sum().clamp_min(1) / x.shape[-1]
            loss = recon_loss + args.vq_weight * vq_out["vq_loss"]
            if stage == "warmup" and not fixed_uniform:
                targets = torch.arange(1, segments, device=x.device).float()[None, :].expand(x.shape[0], -1)
                rows = []
                for r, i in enumerate(idxs):
                    bounds = train_items[i]["dp_bounds"][segments]
                    rows.append(edge_c[r, torch.tensor(bounds, device=x.device, dtype=torch.long)])
                boundary_loss = ((torch.stack(rows) - targets) ** 2).mean()
                loss = loss + args.boundary_weight * boundary_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    mse, perplexity = eval_test_mse(model, eval_items, eval_batches)
    return {"mse": mse, "perplexity": perplexity,
            "train_seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B2: VQ rate-distortion")
    parser.add_argument("--ks", type=str, default="16,32")
    parser.add_argument("--codebooks", type=str, default="64,256,1024")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--anneal-steps", type=int, default=1700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tau-start", type=float, default=0.35)
    parser.add_argument("--tau-end", type=float, default=0.08)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--no-phase", action="store_true",
                        help="disable the u (phase) decoder channel")
    parser.add_argument("--no-duration", action="store_true",
                        help="disable the d (duration) decoder channel")
    parser.add_argument("--timing-mode", type=str, default="density",
                        choices=["density", "billed", "none"],
                        help="density: u/d from the learned density (un-billed "
                             "per-frame info); billed: u/d derived from the hard "
                             "partition only (covered by the boundary side-info "
                             "charge); none: no timing channels")
    parser.add_argument("--dead-code-reset", action="store_true",
                        help="reset unused VQ codes to random data points every "
                             "16 EMA updates (codebook-collapse control)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results_experiment_b_vq.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"device={device}", flush=True)

    train_items, test_items, pca = prepare_data(args.pca_dim, device)
    eval_items = test_items[: args.eval_size]
    eval_batches = make_batches(eval_items, args.batch_size)

    dp_store = json.loads(
        (FEATURE_DIR / f"dp_bounds_pca{args.pca_dim}.json").read_text()
    )
    ks = [int(k) for k in args.ks.split(",")]
    codebooks = [int(c) for c in args.codebooks.split(",")]

    results = {"config": {**vars(args), "output": str(args.output)}, "points": []}

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    total_runs = len(ks) * len(codebooks) * 3
    done = 0
    for K in ks:
        # DP bounds for this K (cached from Experiment B).
        for it in train_items + test_items:
            it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}
        for codebook in codebooks:
            batches = make_batches(train_items, args.batch_size)
            for arm in ("adaptive", "uniform", "frame"):
                done += 1
                print(f"[{done}/{total_runs}] arm={arm} K={K} codebook={codebook}",
                      flush=True)
                t0 = time.time()
                if arm == "frame":
                    model = FrameVQ(args.pca_dim, args.latent_dim, codebook,
                                    dead_code_reset=args.dead_code_reset).to(device)
                    out = train_frame_vq(model, train_items, batches, eval_items,
                                         eval_batches, args)
                else:
                    model = PhaseDurationSegmenter(
                        args.pca_dim, K, latent_dim=args.latent_dim,
                        codebook_size=codebook, fixed_uniform=(arm == "uniform"),
                        timing_mode=args.timing_mode,
                        use_phase=not args.no_phase,
                        use_duration=not args.no_duration,
                        tau=args.tau_start,
                    ).to(device)
                    out = train_segmenter_vq(model, train_items, batches, eval_items,
                                             eval_batches, args, K,
                                             fixed_uniform=(arm == "uniform"))
                mean_t = float(np.mean([it["T"] for it in eval_items]))
                bits_exact, bits_adj = zip(*[
                    utt_bits(it["T"], arm, K, codebook, args.latent_dim)
                    for it in eval_items
                ])
                point = {
                    "arm": arm, "K": K, "codebook": codebook,
                    "dims": args.latent_dim,
                    "mse": out["mse"],
                    "perplexity": out["perplexity"],
                    "bits_per_utt_exact": float(np.mean(bits_exact)),
                    "bits_per_utt_adj": float(np.mean(bits_adj)),
                    "bits_per_sec_exact": float(np.mean(bits_exact)) / (mean_t / 50.0),
                    "train_seconds": out["train_seconds"],
                    "wall_seconds": round(time.time() - t0, 1),
                }
                results["points"].append(point)
                save()
                print(f"    mse={point['mse']:.5f} bits/s={point['bits_per_sec_exact']:.0f} "
                      f"perplexity={point['perplexity']:.1f} ({out['train_seconds']}s)",
                      flush=True)

    print("\n=== RD table (exact bits/s) ===", flush=True)
    for p in sorted(results["points"], key=lambda q: (q["arm"], q["K"], q["codebook"])):
        print(f"{p['arm']:9s} K={p['K']:3d} N={p['codebook']:5d} "
              f"mse={p['mse']:.5f} bits/s={p['bits_per_sec_exact']:8.0f}", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
