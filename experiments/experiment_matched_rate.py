"""Experiment: Matched-Rate Rate-Distortion Curves.

Creates genuinely overlapping rate-distortion curves by sweeping codebook sizes
more densely to ensure frame, adaptive, and uniform arms have overlapping bitrates.

Goal: Produce R->D(R) curves with overlapping points so we can legitimately claim
dominance or lack thereof at matched bitrates.

Strategy:
- Frame: N = {2,3,4,5,6,8,10,12,16,20,24,32,40,48,64} to cover ~25-300 b/s
- Adaptive (K=32): N = {4,8,16,32,64,128,256,512} to cover ~30-80 b/s
- Adaptive (K=16): N = {4,8,16,32,64,128,256,512} to cover ~15-40 b/s
- Uniform (K=32): N = {4,8,16,32,64,128,256,512}
- Uniform (K=16): N = {4,8,16,32,64,128,256,512}

This should create overlapping rate ranges around 30-40 b/s and allow interpolation
to compare at exactly matched bitrates.

Output: experiments/results_matched_rate_{seed}.json
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
    EMAVQ,
    PhaseDurationSegmenter,
    apply_pca,
    fit_pca,
    load_items,
    make_batches,
    masked_mse,
    pad_batch,
)

# Fix FEATURE_DIR path and patch experiment_b module
import experiment_b
RESEARCH_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = RESEARCH_ROOT / "experiments" / "features_wavlm_l6"
experiment_b.FEATURE_DIR = FEATURE_DIR


class FrameVQ(nn.Module):
    """Per-frame latent bottleneck."""

    def __init__(self, dims: int, latent_dim: int = 32, codebook_size: int = 256,
                 decoder_hidden: int = 192, decoder_layers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.proj = nn.Linear(dims, latent_dim)
        self.vq = EMAVQ(latent_dim, codebook_size)
        layers: list[nn.Module] = []
        for layer in range(decoder_layers):
            layers.append(nn.Conv1d(latent_dim if layer == 0 else decoder_hidden,
                                    decoder_hidden, kernel_size=7, padding=3))
            layers.append(nn.GELU())
        layers.append(nn.Conv1d(decoder_hidden, dims, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        b, t, _ = x.shape
        z = self.proj(x)
        zb, td, dz = z.shape
        zq, codes, vq_loss, perplexity = self.vq(z.reshape(zb * td, dz))
        zq = zq.view(zb, td, dz)
        recon = self.decoder(zq.transpose(1, 2)).transpose(1, 2)
        return recon, {"codes": codes.view(zb, td), "vq_loss": vq_loss,
                       "perplexity": perplexity}


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


def utt_bits(T: int, arm: str, K: int, codebook: int, dims: int) -> tuple[float, float]:
    """(exact bits incl. all side info, content-only bits) per utterance."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-rate RD curves")
    parser.add_argument("--ks", type=str, default="16,32")
    parser.add_argument("--frame-codebooks", type=str, default="2,3,4,5,6,8,10,12,16,20,24,32,40,48,64")
    parser.add_argument("--segment-codebooks", type=str, default="4,8,16,32,64,128,256,512")
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
    parser.add_argument("--timing-mode", type=str, default="billed",
                        choices=["density", "billed", "none"],
                        help="Use billed timing for honest comparison")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results_matched_rate.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"device={device}", flush=True)

    train_items, test_items, pca = prepare_data(args.pca_dim, device)
    # Sort all items consistently by T for proper batch indexing
    train_items = sorted(train_items, key=lambda it: it["T"])
    test_items = sorted(test_items, key=lambda it: it["T"])
    eval_items = test_items[: args.eval_size]
    eval_batches = make_batches(eval_items, args.batch_size)

    dp_store = json.loads(
        (FEATURE_DIR / f"dp_bounds_pca{args.pca_dim}.json").read_text()
    )
    ks = [int(k) for k in args.ks.split(",")]
    frame_codebooks = [int(c) for c in args.frame_codebooks.split(",")]
    segment_codebooks = [int(c) for c in args.segment_codebooks.split(",")]

    results = {"config": {**vars(args), "output": str(args.output)}, "points": []}

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Calculate total runs
    total_runs = (len(frame_codebooks) +  # frame
                  len(ks) * len(segment_codebooks) * 2)  # adaptive + uniform for each K
    done = 0

    # Frame arm - denser sweep
    for codebook in frame_codebooks:
        done += 1
        print(f"[{done}/{total_runs}] arm=frame codebook={codebook}", flush=True)
        t0 = time.time()
        batches = make_batches(train_items, args.batch_size)
        model = FrameVQ(args.pca_dim, args.latent_dim, codebook).to(device)
        out = train_frame_vq(model, train_items, batches, eval_items, eval_batches, args)
        mean_t = float(np.mean([it["T"] for it in eval_items]))
        bits_exact, bits_adj = zip(*[
            utt_bits(it["T"], "frame", 0, codebook, args.latent_dim)
            for it in eval_items
        ])
        point = {
            "arm": "frame", "K": 0, "codebook": codebook,
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
              f"perplexity={point['perplexity']:.1f}", flush=True)

    # Segment arms (adaptive and uniform) for each K
    for K in ks:
        # Load DP bounds for this K
        for it in train_items + test_items:
            it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}

        for codebook in segment_codebooks:
            for arm in ("adaptive", "uniform"):
                done += 1
                print(f"[{done}/{total_runs}] arm={arm} K={K} codebook={codebook}",
                      flush=True)
                t0 = time.time()
                batches = make_batches(train_items, args.batch_size)
                model = PhaseDurationSegmenter(
                    args.pca_dim, K, latent_dim=args.latent_dim,
                    codebook_size=codebook, fixed_uniform=(arm == "uniform"),
                    timing_mode=args.timing_mode,
                    use_phase=True,
                    use_duration=True,
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
                      f"perplexity={point['perplexity']:.1f}", flush=True)

    print("\n=== Matched-rate RD table (exact bits/s) ===", flush=True)
    for p in sorted(results["points"], key=lambda q: (q["arm"], q["K"], q["codebook"])):
        k_str = f"K={p['K']:3d}" if p["K"] > 0 else "   frame"
        print(f"{p['arm']:9s} {k_str:12s} N={p['codebook']:5d} "
              f"mse={p['mse']:.5f} bits/s={p['bits_per_sec_exact']:8.0f}", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
