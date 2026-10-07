"""Boundary drift experiment: does DP-initialized refinement move boundaries?

Reviewer question (paper review, Oct 2026): every real-speech adaptive arm is
warm-started from the DP oracle for 300 steps, and "learned ~ DP" was inferred
from tied phone-edge F1, never measured as boundary displacement. This script:

  1. trains one K=32 adaptive arm with the exact Experiment B/C recipe
     (non-VQ, 300 warmup + 1700 anneal, seed 7),
  2. snapshots the warp-exact boundaries at the end of warmup (step 300),
     at anneal steps +250/+500/.../+1700, and at convergence,
  3. reports, per snapshot, the mean |displacement| (SSL frames) of each
     boundary from its step-300 position and from the DP-oracle position,
     over the 48 test utterances.

If final-vs-warmstart displacement is small, "learned ~ DP" is inherited from
initialization; if it is large while learned still ties DP on F1, refinement
rediscovers DP-like placement. Either way the mechanism claim becomes direct.

Output: experiments/results_experiment_drift.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiment_a import set_seed
from experiment_b import (
    FEATURE_DIR,
    PhaseDurationSegmenter,
    apply_pca,
    fit_pca,
    load_items,
    make_batches,
    pad_batch,
    warp_boundaries,
)

RESEARCH = Path(__file__).resolve().parent.parent


def boundary_displacement(pred: list[int], ref: list[int]) -> float:
    """Mean |pred_k - ref_k| over matched (monotonic) boundaries."""
    if not ref:
        return float("nan")
    n = min(len(pred), len(ref))
    if n == 0:
        return float("nan")
    return float(np.mean([abs(pred[i] - ref[i]) for i in range(n)]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start drift measurement")
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--anneal-steps", type=int, default=1700)
    parser.add_argument("--snapshot-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tau-start", type=float, default=0.35)
    parser.add_argument("--tau-end", type=float, default=0.08)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=RESEARCH / "experiments" / "results_experiment_drift.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"device={device}", flush=True)

    train_items = load_items("train", device)
    pca = fit_pca(train_items)
    test_items = load_items("test", device)
    for split_items in (train_items, test_items):
        apply_pca(split_items, pca, args.pca_dim, device)
    eval_items = test_items[: args.eval_size]
    eval_batches = make_batches(eval_items, args.batch_size)

    dp_store = json.loads((FEATURE_DIR / f"dp_bounds_pca{args.pca_dim}.json").read_text())
    K = args.segments
    for it in train_items + test_items:
        it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}

    # --- exact Experiment B/C training loop, with boundary snapshots ---------
    model = PhaseDurationSegmenter(
        args.pca_dim, K, latent_dim=args.latent_dim, tau=args.tau_start
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    batches = make_batches(train_items, args.batch_size)

    snapshots: dict[str, list[list[list[int]]]] = {}  # stage -> per-utterance bounds
    step = 0

    def snapshot(stage: str) -> None:
        model.eval()
        per_utt = []
        with torch.no_grad():
            for it in eval_items:
                t = it["T"]
                feats = it["features"].unsqueeze(0)
                mask = torch.ones(1, t, device=device)
                _, _, edge_c, _, _ = model.forward_full(feats, mask)
                per_utt.append(warp_boundaries(edge_c[0], t, K))
        snapshots[stage] = per_utt
        print(f"[snapshot] {stage} captured ({len(per_utt)} utterances)", flush=True)

    for stage, n_steps in (("warmup", args.warmup_steps), ("anneal", args.anneal_steps)):
        for s in range(1, n_steps + 1):
            step += 1
            model.train()
            if stage == "anneal" and n_steps > 0:
                progress = s / n_steps
                model.tau = args.tau_start * (args.tau_end / args.tau_start) ** progress
            idxs = list(random.choice(batches))
            x, mask = pad_batch(train_items, idxs)
            recon, masks, edge_c, _, _ = model.forward_full(x, mask)
            diff = (recon - x) ** 2 * mask.unsqueeze(-1)
            recon_loss = diff.sum() / mask.sum().clamp_min(1) / x.shape[-1]
            loss = recon_loss
            if stage == "warmup":
                targets = torch.arange(1, K, device=device).float()[None, :].expand(x.shape[0], -1)
                rows = []
                for r, i in enumerate(idxs):
                    bounds = train_items[i]["dp_bounds"][K]
                    rows.append(edge_c[r, torch.tensor(bounds, device=device, dtype=torch.long)])
                c_at_bounds = torch.stack(rows)
                loss = loss + args.boundary_weight * (((c_at_bounds - targets) ** 2).mean())
            entropy = -(masks.clamp_min(1e-8) * masks.clamp_min(1e-8).log())
            entropy = (entropy * mask.unsqueeze(1)).sum() / (mask.sum() * K)
            loss = loss + args.entropy_weight * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if stage == "warmup" and s == n_steps:
                snapshot("warmup_end")
            if stage == "anneal" and (s % args.snapshot_every == 0 or s == n_steps):
                snapshot(f"anneal_step_{s}")

    # simple final soft-MSE check on eval set
    model.eval()
    tot, frames = 0.0, 0
    with torch.no_grad():
        for idxs in eval_batches:
            x, mask = pad_batch(eval_items, idxs)
            recon, _, _, _, _ = model.forward_full(x, mask)
            d = (recon - x) ** 2 * mask.unsqueeze(-1)
            tot += float(d.sum())
            frames += int(mask.sum()) * x.shape[-1]
    final_soft = tot / frames

    # --- displacement analysis ----------------------------------------------
    dp_bounds_per_utt = [it["dp_bounds"][K] for it in eval_items]
    analysis = {}
    for stage, per_utt in snapshots.items():
        d_warm = [boundary_displacement(b, snapshots["warmup_end"][u])
                  for u, b in enumerate(per_utt)]
        d_dp = [boundary_displacement(b, dp_bounds_per_utt[u]) for u, b in enumerate(per_utt)]
        analysis[stage] = {
            "mean_disp_vs_warmstart_ssl_frames": round(float(np.nanmean(d_warm)), 3),
            "mean_disp_vs_dp_ssl_frames": round(float(np.nanmean(d_dp)), 3),
            "median_disp_vs_dp_ssl_frames": round(float(np.nanmedian(d_dp)), 3),
            "p90_disp_vs_dp_ssl_frames": round(float(np.nanpercentile(d_dp, 90)), 3),
        }

    results = {
        "config": {**vars(args), "output": str(args.output)},
        "final_soft_mse": round(final_soft, 5),
        "reference_soft_mse_C": 0.0941,
        "analysis": analysis,
        "note": "Displacements in 50 Hz SSL frames; boundary k matched to boundary k "
                "(monotonic partition). warmup_end IS the step-300 DP warm start, so its "
                "disp_vs_dp column is the warm-start quality baseline.",
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== Drift summary (SSL frames) ===", flush=True)
    for stage, a in analysis.items():
        print(f"{stage:16s} vs_warmstart={a['mean_disp_vs_warmstart_ssl_frames']:7.3f} "
              f"vs_dp={a['mean_disp_vs_dp_ssl_frames']:7.3f} "
              f"(median {a['median_disp_vs_dp_ssl_frames']}, p90 {a['p90_disp_vs_dp_ssl_frames']})",
              flush=True)
    print(f"final_soft_mse={final_soft:.5f} (C reference: 0.0941)", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
