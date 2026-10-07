"""Experiment F — does segmentation become useful on acoustic representations?

Frame-VQ vs Uniform segment-VQ vs Adaptive segment-VQ on MEL spectrograms,
mirroring Experiment B2's protocol exactly (same arms, codebooks, training
recipe, seed, bit accounting) so the WavLM panel (B2) and mel panel (F) are
comparable within representation class.

Representation-specific adaptations (pre-registered):
  - Native 80-dim log-mel, NO PCA: PCA would discard exactly the acoustic
    detail under test. Per-dimension standardization (train stats) for
    training stability, applied identically to all arms.
  - DP warm-start bounds recomputed on standardized mel (cached to
    experiments/dp_bounds_mel_K{K}.json), because the cached WavLM DP bounds
    are meaningless for mel features.
  - K=32 only (B2 showed K=16/32 tell the same story), codebooks {64,256,1024}.
  - Bit accounting at the mel frame rate (22050/256 Hz = 86.13 Hz).

Pre-registered decision rule (fixed before running; see docs/experiment_f.md):
  Primary: adaptive vs frame MSE at K=32 per codebook N, on mel.
  - F-NULL:    frame MSE < adaptive MSE at every N with a relative gap similar
               to WavLM (frame/adaptive ~ 0.5) -> the negative result is
               representation-class-INDEPENDENT.
  - F-POSITIVE: adaptive <= frame at some N, or frame/adaptive > 0.9
               -> segmentation becomes competitive on acoustic features;
               "where temporal abstraction happens" finding.
  - PARTIAL:   frame/adaptive in (0.7, 0.9) -> mel structure helps segments,
               not enough to flip.
  Sanity: adaptive must beat uniform on mel (as it did on WavLM in B); if not,
  the segmenter failed to train on this representation (confound, report).

Output: experiments/results_experiment_f.json

Run from research root with the fastspeech2.0 venv python.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiment_a import set_seed
from experiment_b import dp_bounds_for
from experiment_b_vq import (
    FrameVQ,
    PhaseDurationSegmenter,
    eval_test_mse,
    train_frame_vq,
    train_segmenter_vq,
)

RESEARCH = Path(__file__).resolve().parent.parent
FEATURE_DIR = RESEARCH / "experiments" / "features_wavlm_l6"
MEL_DIR = Path("C:/Users/Lenovo/Desktop/fastspeech2.0/dataset/processed/mel")
MEL_RATE = 22050.0 / 256.0  # 86.1328 frames/s


def load_mel_items(split: str, device: torch.device,
                   stats: tuple[np.ndarray, np.ndarray] | None = None
                   ) -> tuple[list[dict], tuple[np.ndarray, np.ndarray] | None]:
    """Mel items [T, 80] torch tensors on device, standardized by train stats.

    Split assignment comes from the WavLM feature index (same utterances).
    """
    index = json.loads((FEATURE_DIR / "index.json").read_text())["utterances"]
    items = []
    for utt_id, meta in index.items():
        if meta["split"] != split:
            continue
        mel = np.load(MEL_DIR / f"{utt_id}.npy").astype(np.float32)  # [80, T]
        items.append({"id": utt_id, "mel": mel.T.copy(), "T": int(mel.shape[1])})
    items.sort(key=lambda it: it["T"])
    if stats is None:
        stack = np.concatenate([it["mel"] for it in items], axis=0)
        mean = stack.mean(axis=0)
        std = np.clip(stack.std(axis=0), 1e-6, None)
        stats = (mean, std)
        standardized = True
    else:
        standardized = False
    mean, std = stats
    for it in items:
        f = (it["mel"] - mean) / std
        it["features"] = torch.from_numpy(np.ascontiguousarray(f)).to(device)
    print(f"split={split}: {len(items)} utterances, "
          f"T mean={np.mean([it['T'] for it in items]):.0f} "
          f"max={max(it['T'] for it in items)}", flush=True)
    return items, (stats if standardized else None)


def get_or_compute_dp_bounds(train_items: list[dict], test_items: list[dict],
                             K: int, cache: Path) -> dict:
    """DP reconstruction-optimal bounds on standardized mel, cached."""
    if cache.exists():
        store = json.loads(cache.read_text())
        if str(K) in store:
            print(f"loaded cached mel DP bounds K={K} ({len(store[str(K)])} utts)",
                  flush=True)
            return store
    print(f"computing mel DP bounds K={K} for {len(train_items) + len(test_items)} "
          f"utterances...", flush=True)
    store = {str(K): {}}
    t0 = time.time()
    for i, it in enumerate(train_items + test_items):
        bounds = dp_bounds_for(it["features"].detach().cpu().numpy(), K)
        store[str(K)][it["id"]] = bounds
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  dp {i + 1}/{len(train_items) + len(test_items)} "
                  f"({el:.0f}s, ~{el / (i + 1) * (len(train_items) + len(test_items) - i - 1):.0f}s left)",
                  flush=True)
    cache.write_text(json.dumps(store))
    print(f"  cached -> {cache} ({time.time() - t0:.0f}s)", flush=True)
    return store


def utt_bits(T: int, arm: str, K: int, codebook: int) -> tuple[float, float]:
    """(exact bits incl. side info, content-only bits) per utterance."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment F: mel rate-distortion (B2 protocol)")
    parser.add_argument("--k", type=int, default=32)
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
    parser.add_argument("--no-phase", action="store_true",
                        help="disable the u (phase) decoder channel")
    parser.add_argument("--no-duration", action="store_true",
                        help="disable the d (duration) decoder channel")
    parser.add_argument("--timing-mode", type=str, default="density",
                        choices=["density", "billed", "none"],
                        help="density: u/d from the learned density (un-billed "
                             "per-frame info); billed: u/d derived from the hard "
                             "partition only; none: no timing channels")
    parser.add_argument("--dead-code-reset", action="store_true",
                        help="reset unused VQ codes (codebook-collapse control)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=RESEARCH / "experiments" / "results_experiment_f.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"device={device}", flush=True)
    K = args.k

    # Data: standardize mel with TRAIN stats; apply to all splits.
    train_items, stats = load_mel_items("train", device)
    val_items, _ = load_mel_items("val", device, stats)
    test_items, _ = load_mel_items("test", device, stats)
    eval_items = test_items[: args.eval_size]
    eval_batches = [list(range(i, min(i + args.batch_size, len(eval_items))))
                    for i in range(0, len(eval_items), args.batch_size)]

    # DP bounds on mel (cached).
    cache = RESEARCH / "experiments" / f"dp_bounds_mel_K{K}.json"
    dp_store = get_or_compute_dp_bounds(train_items, test_items, K, cache)
    for it in train_items + test_items:
        it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}

    codebooks = [int(c) for c in args.codebooks.split(",")]
    results = {
        "config": {
            "k": K, "seed": args.seed, "latent_dim": args.latent_dim,
            "codebooks": codebooks, "warmup_steps": args.warmup_steps,
            "anneal_steps": args.anneal_steps, "lr": args.lr,
            "vq_weight": args.vq_weight, "eval_size": args.eval_size,
            "mel_rate_hz": MEL_RATE, "n_mels": 80,
            "standardization": "per-dimension z-score (train stats), all arms",
            "dp_warm_start": "recomputed on standardized mel (cached)",
            "decision_rule": (
                "F-NULL: frame < adaptive at every N (gap ~ WavLM, frame/adaptive ~0.5) "
                "-> negative result is representation-class independent; "
                "F-POSITIVE: adaptive <= frame at some N or frame/adaptive > 0.9 "
                "-> segmentation competitive on acoustic features; "
                "PARTIAL: frame/adaptive in (0.7, 0.9). "
                "Sanity: adaptive must beat uniform (as on WavLM in B)."),
            "output": str(args.output),
        },
        "points": [],
    }

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2, default=str),
                               encoding="utf-8")

    total_runs = len(codebooks) * 3
    done = 0
    for codebook in codebooks:
        batches = [list(range(i, min(i + args.batch_size, len(train_items))))
                   for i in range(0, len(train_items), args.batch_size)]
        for arm in ("adaptive", "uniform", "frame"):
            done += 1
            print(f"[{done}/{total_runs}] arm={arm} K={K} codebook={codebook}",
                  flush=True)
            t0 = time.time()
            if arm == "frame":
                model = FrameVQ(80, args.latent_dim, codebook).to(device)
                out = train_frame_vq(model, train_items, batches, eval_items,
                                     eval_batches, args)
            else:
                model = PhaseDurationSegmenter(
                    80, K, latent_dim=args.latent_dim, codebook_size=codebook,
                    fixed_uniform=(arm == "uniform"), tau=args.tau_start,
                    timing_mode=args.timing_mode,
                    use_phase=not args.no_phase,
                    use_duration=not args.no_duration,
                ).to(device)
                out = train_segmenter_vq(model, train_items, batches, eval_items,
                                         eval_batches, args, K,
                                         fixed_uniform=(arm == "uniform"))
            mean_t = float(np.mean([it["T"] for it in eval_items]))
            bits_exact, bits_adj = zip(*[
                utt_bits(it["T"], arm, K, codebook) for it in eval_items
            ])
            point = {
                "arm": arm, "K": K, "codebook": codebook,
                "dims": args.latent_dim,
                "mse": out["mse"],
                "perplexity": out["perplexity"],
                "bits_per_utt_exact": float(np.mean(bits_exact)),
                "bits_per_utt_adj": float(np.mean(bits_adj)),
                "bits_per_sec_exact": float(np.mean(bits_exact)) / (mean_t / MEL_RATE),
                "train_seconds": out["train_seconds"],
                "wall_seconds": round(time.time() - t0, 1),
            }
            results["points"].append(point)
            save()
            print(f"    mse={point['mse']:.5f} bits/s={point['bits_per_sec_exact']:.0f} "
                  f"perplexity={point['perplexity']:.1f} ({out['train_seconds']}s)",
                  flush=True)

    # Summary + decision-rule evaluation.
    print("\n=== Experiment F: mel RD table (exact bits/s) ===", flush=True)
    by_key = {(p["arm"], p["codebook"]): p for p in results["points"]}
    print("N     frame      adaptive   uniform    frame/adaptive", flush=True)
    for codebook in codebooks:
        f_ = by_key[("frame", codebook)]
        a = by_key[("adaptive", codebook)]
        u = by_key[("uniform", codebook)]
        ratio = f_["mse"] / max(a["mse"], 1e-9)
        print(f"{codebook:5d} {f_['mse']:.5f}    {a['mse']:.5f}    {u['mse']:.5f}    "
              f"{ratio:.3f}", flush=True)
    print(f"\nsaved={args.output}", flush=True)


if __name__ == "__main__":
    main()
