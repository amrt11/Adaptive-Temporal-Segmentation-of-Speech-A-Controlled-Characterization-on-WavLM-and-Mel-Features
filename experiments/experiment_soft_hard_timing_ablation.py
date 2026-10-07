"""Experiment: Soft/Hard × Timing Ablation Matrix.

This experiment investigates the discrepancy between soft and hard reconstruction
performance when masks are nearly hard (88.5% hard mass), yet soft MSE (0.090)
is dramatically better than hard MSE (0.321).

Each trained model is evaluated two ways, giving the 2x3 matrix:
  soft reconstruction  - decoder output from soft mask composition (+ channels)
  hard reconstruction  - per-segment mean at the model's own warp boundaries
                         (channel-free; depends only on the learned partition)

Timing configurations (soft arm):
1. density  - u/d from the learned density field (the original, un-billed leak)
2. none     - no u/d channels (isolates soft composition value)
3. billed   - u/d derived from the hard partition only (honest timing)

Hard references (uniform@K and DP@K segment means) are loaded from the
existing Experiment B results for context.

Output: experiments/results_soft_hard_timing_ablation.json
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

from experiment_a import set_seed
from experiment_b import (
    FEATURE_DIR,
    PhaseDurationSegmenter,
    apply_pca,
    fit_pca,
    hard_segment_mse,
    load_items,
    make_batches,
    masked_mse,
    pad_batch,
    warp_boundaries,
)


def prepare_data(pca_dim: int, device: torch.device):
    train_items = load_items("train", device)
    pca = fit_pca(train_items)
    val_items = load_items("val", device)
    test_items = load_items("test", device)
    for split_items in (train_items, val_items, test_items):
        apply_pca(split_items, pca, pca_dim, device)
    return train_items, test_items, pca


def eval_test_mse(model, eval_items, eval_batches, segments: int) -> tuple[float, float, float, float]:
    """Evaluate soft MSE, hard partition-mean MSE, perplexity, hard mass."""
    model.eval()
    weighted = 0.0
    hard_weighted = 0.0
    total_frames = 0
    perplexity_sum = 0.0
    perplexity_count = 0
    hard_mass_sum = 0.0
    with torch.no_grad():
        for idxs in eval_batches:
            x, mask = pad_batch(eval_items, idxs)
            recon, masks, edge_c, z, vq_out = model.forward_full(x, mask)
            weighted += masked_mse(recon, x, mask) * float(mask.sum())
            total_frames += int(mask.sum())
            perplexity_sum += float(vq_out["perplexity"])
            perplexity_count += 1
            # Compute hard mass: proportion of mask mass on the argmax channel
            hard_mask_idx = torch.argmax(masks, dim=1)
            hard_onehot = torch.zeros_like(masks)
            hard_onehot.scatter_(1, hard_mask_idx.unsqueeze(1), 1.0)
            hard_mass = (masks * hard_onehot).sum() / masks.sum().clamp_min(1e-9)
            hard_mass_sum += float(hard_mass)
            # Hard reconstruction: per-segment mean at this model's own warp
            # boundaries (matches Experiment B's learned_hard_mse definition).
            for row, i in enumerate(idxs):
                it = eval_items[i]
                t = it["T"]
                feats_np = it["features"].detach().cpu().numpy()
                pred_b = warp_boundaries(edge_c[row], t, segments)
                hard_weighted += float(hard_segment_mse(feats_np, pred_b)) * t
    mse = float(weighted) / float(total_frames)
    hard_mse = float(hard_weighted) / float(total_frames)
    perplexity = perplexity_sum / max(perplexity_count, 1)
    hard_mass = float(hard_mass_sum) / max(perplexity_count, 1) if hard_mass_sum > 0 else 0.0
    return mse, hard_mse, perplexity, hard_mass


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
    mse, hard_mse, perplexity, hard_mass = eval_test_mse(model, eval_items, eval_batches, segments)
    return {"mse": mse, "hard_mse": hard_mse, "perplexity": perplexity,
            "hard_mass": hard_mass, "train_seconds": round(time.time() - t0, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft/Hard × Timing Ablation")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--codebook", type=int, default=256)
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results_soft_hard_timing_ablation.json"))
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
    K = args.k
    for it in train_items + test_items:
        it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}

    # Ablation configurations for soft reconstruction
    ablations = [
        {"use_phase": True, "use_duration": True, "timing_mode": "density",
         "name": "soft_with_ud"},
        {"use_phase": False, "use_duration": False, "timing_mode": "none",
         "name": "soft_no_ud"},
        {"use_phase": True, "use_duration": True, "timing_mode": "billed",
         "name": "soft_billed_ud"},
    ]

    results = {"config": {**vars(args), "output": str(args.output)}, "points": []}

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    total_runs = len(ablations)
    done = 0

    for ablation in ablations:
        done += 1
        print(f"[{done}/{total_runs}] ablation={ablation['name']}", flush=True)
        t0 = time.time()
        batches = make_batches(train_items, args.batch_size)
        model = PhaseDurationSegmenter(
            args.pca_dim, K, latent_dim=args.latent_dim,
            codebook_size=args.codebook, fixed_uniform=False,
            timing_mode=ablation["timing_mode"],
            use_phase=ablation["use_phase"],
            use_duration=ablation["use_duration"],
            tau=args.tau_start,
        ).to(device)
        out = train_segmenter_vq(model, train_items, batches, eval_items,
                                 eval_batches, args, K,
                                 fixed_uniform=False)
        point = {
            "ablation": ablation["name"],
            "K": K,
            "codebook": args.codebook,
            "dims": args.latent_dim,
            "mse": out["mse"],
            "hard_mse": out["hard_mse"],
            "perplexity": out["perplexity"],
            "hard_mass": out["hard_mass"],
            "use_phase": ablation["use_phase"],
            "use_duration": ablation["use_duration"],
            "timing_mode": ablation["timing_mode"],
            "train_seconds": out["train_seconds"],
            "wall_seconds": round(time.time() - t0, 1),
        }
        results["points"].append(point)
        save()
        print(f"    mse={point['mse']:.5f} hard_mass={point['hard_mass']:.3f} "
              f"perplexity={point['perplexity']:.1f}", flush=True)

    # Load hard references from Experiment B (arms structure: learned/uniform/DP
    # segment-mean MSE at K=32).
    print("\n=== Loading hard references from Experiment B ===", flush=True)
    try:
        exp_b_results = json.loads(
            (FEATURE_DIR.parent / "results_experiment_b.json").read_text()
        )
        final = exp_b_results["arms"][f"K{K}"]["final"]
        reference = {
            "learned_hard_mse_at_learned_boundaries": final["learned_hard_mse"],
            "uniform_hard_mse": final["uniform_mse"],
            "dp_hard_mse": final["dp_mse"],
            "soft_decoder_mse_density_timing": final["learned_mse"],
        }
        results["hard_reference_experiment_b"] = reference
        for k, v in reference.items():
            try:
                print(f"  {k}: {float(v):.5f}", flush=True)
            except (TypeError, ValueError):
                print(f"  {k}: {v}", flush=True)
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load Experiment B references: {e}", flush=True)

    print("\n=== Ablation matrix results (2x3: reconstruction x timing) ===", flush=True)
    print("Ablation              soft MSE   hard MSE   Hard Mass  Perplexity", flush=True)
    for p in results["points"]:
        print(f"{p['ablation']:20s} {float(p['mse']):.5f}    {float(p['hard_mse']):.5f}    "
              f"{float(p['hard_mass']):.3f}      {float(p['perplexity']):.1f}", flush=True)

    print(f"\nsaved={args.output}", flush=True)


if __name__ == "__main__":
    main()
