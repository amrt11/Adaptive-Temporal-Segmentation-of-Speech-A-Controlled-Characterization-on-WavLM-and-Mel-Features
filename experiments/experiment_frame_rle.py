"""Experiment: Frame-VQ + run-length coding (variable-rate allocation control).

Question (post-review): is the billed segment arm's low-rate advantage (50-79 b/s)
due to *learned segmentation*, or merely to *variable-rate allocation* - i.e. would
the cheapest variable-rate trick (run-length coding of per-frame codes) do as well?

Design:
  - Train FrameVQ exactly as in experiment_matched_rate_multiseed.py (same seeds,
    order, recipe), for every frame codebook N.
  - The DECODER OUTPUT IS IDENTICAL to plain frame-VQ: run-length coding only
    compresses the transmitted code sequence. MSE is therefore the frame arm's MSE,
    recomputed here per utterance.
  - Rate: ideal two-part enumerative description of (R, codes, run lengths):
        bits = log2(T)              [number of runs R]
             + log2(N)              [first code]
             + (R-1) * log2(N-1)    [each subsequent code differs from previous]
             + log2 C(T-1, R-1)     [composition of T frames into R positive runs]
    This matches the paper's conventions (ideal combinatorial lengths, same style
    as the segment arm's log2 C(T, K-1) boundary charge). No decoder-side
    information is used - the audit is passed trivially.

Output: experiments/results_frame_rle.json
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
from experiment_matched_rate_multiseed import FrameVQ, prepare_data
from experiment_b import make_batches, masked_mse, pad_batch


def utt_rle_bits(T: int, R: int, N: int) -> float:
    """Ideal two-part enumerative bits for an R-run code sequence over N codes."""
    bits = math.log2(T) + math.log2(N)
    if R > 1:
        bits += (R - 1) * math.log2(N - 1)
        bits += math.log2(math.comb(T - 1, R - 1))
    return bits


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-VQ + run-length control")
    parser.add_argument("--frame-codebooks", type=str,
                        default="2,3,4,5,6,8,10,12,16,20,24,32,40,48,64")
    parser.add_argument("--seeds", type=str, default="7,42,123")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--anneal-steps", type=int, default=1700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results_frame_rle.json"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    codebooks = [int(c) for c in args.frame_codebooks.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"config": {**vars(args), "output": str(args.output),
                          "accounting": ("ideal two-part RLE: log2 T (runs) + log2 N "
                                         "(first code) + (R-1) log2(N-1) (code choices) "
                                         "+ log2 C(T-1,R-1) (run-length composition)")},
               "points": []}

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    for seed in seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        set_seed(seed)
        print(f"device={device}", flush=True)
        train_items, test_items, _pca = prepare_data(args.pca_dim, device)
        train_items = sorted(train_items, key=lambda it: it["T"])
        test_items = sorted(test_items, key=lambda it: it["T"])
        eval_items = test_items[: args.eval_size]

        for codebook in codebooks:
            print(f"[seed={seed}] arm=frame_rle codebook={codebook}", flush=True)
            t0 = time.time()
            batches = make_batches(train_items, args.batch_size)
            model = FrameVQ(args.pca_dim, args.latent_dim, codebook).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            for _ in range(args.warmup_steps + args.anneal_steps):
                idxs = list(random.choice(batches))
                x, mask = pad_batch(train_items, idxs)
                recon, vq_out = model(x, mask)
                diff = (recon - x) ** 2 * mask.unsqueeze(-1)
                recon_loss = diff.sum() / mask.sum().clamp_min(1) / x.shape[-1]
                loss = recon_loss + args.vq_weight * vq_out["vq_loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            # Per-utterance evaluation: codes, runs, MSE (batch of 1 -> exact T).
            model.eval()
            mse_weighted, total_frames = 0.0, 0
            bits_list, rates_list, runs_list = [], [], []
            with torch.no_grad():
                for it in eval_items:
                    x = it["features"].unsqueeze(0)
                    mask = torch.ones(1, it["T"], device=device)
                    recon, vq_out = model(x, mask)
                    mse_weighted += masked_mse(recon, x, mask) * it["T"]
                    total_frames += it["T"]
                    codes = vq_out["codes"][0].tolist()
                    runs = 1 + sum(1 for a, b in zip(codes[1:], codes[:-1]) if a != b)
                    bits = utt_rle_bits(it["T"], runs, codebook)
                    bits_list.append(bits)
                    rates_list.append(bits / (it["T"] / 50.0))
                    runs_list.append(runs)

            point = {
                "arm": "frame_rle", "K": 0, "codebook": codebook,
                "dims": args.latent_dim, "seed": seed,
                "mse": mse_weighted / total_frames,
                "bits_per_utt_exact": float(np.mean(bits_list)),
                "bits_per_sec_exact": float(np.mean(rates_list)),
                "mean_runs": float(np.mean(runs_list)),
                "mean_T": float(np.mean([it["T"] for it in eval_items])),
                "train_seconds": round(time.time() - t0, 1),
            }
            results["points"].append(point)
            save()
            plain_rate = 50.0 * math.log2(codebook)
            print(f"    mse={point['mse']:.5f} rle_bits/s={point['bits_per_sec_exact']:.1f} "
                  f"(plain frame {plain_rate:.0f}; mean_runs={point['mean_runs']:.1f} "
                  f"of T~{point['mean_T']:.0f})", flush=True)

    # Summary across seeds.
    print("\n=== Summary (mean +/- std over seeds) ===", flush=True)
    summary = {}
    for codebook in codebooks:
        rows = [p for p in results["points"] if p["codebook"] == codebook]
        if not rows:
            continue
        key = f"frame_rle_{codebook}"
        summary[key] = {
            "arm": "frame_rle", "codebook": codebook,
            "mse_mean": float(np.mean([p["mse"] for p in rows])),
            "mse_std": float(np.std([p["mse"] for p in rows])),
            "bits_per_sec_mean": float(np.mean([p["bits_per_sec_exact"] for p in rows])),
            "bits_per_sec_std": float(np.std([p["bits_per_sec_exact"] for p in rows])),
            "n_seeds": len(rows),
        }
        s = summary[key]
        print(f"N={codebook:5d} rle_bits/s={s['bits_per_sec_mean']:7.1f} +/- {s['bits_per_sec_std']:5.1f}  "
              f"mse={s['mse_mean']:.5f} +/- {s['mse_std']:.5f}", flush=True)
    results["summary"] = summary
    save()
    print(f"\nsaved={args.output}", flush=True)


if __name__ == "__main__":
    main()
