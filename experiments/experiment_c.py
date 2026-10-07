"""Experiment C: do learned boundaries correspond to phoneme structure?

Compares warp-exact boundaries from the trained adaptive segmenters (K=16, 32,
identical Experiment B recipe, seed 7) against MAS phone alignments extracted
from the trained v6 FastSpeech2 checkpoint (extract_alignments_c.py).

References and controls, all mapped to the 50 Hz SSL frame grid:
  - MAS phone edges: every phone-token edge; two variants: all tokens and
    content tokens (excluding silence/pause markers).
  - uniform@K edges, DP@K edges (cached oracle), random K-1 subsets of frames.

Metrics per condition: boundary F1 within +/-2 SSL frames (40 ms), mean
distance to nearest reference edge, and the fraction of reference edges
covered within tolerance.

Outputs experiments/results_experiment_c.json.
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
    apply_pca,
    fit_pca,
    load_items,
    make_batches,
    train_arm,
    warp_boundaries,
)
import experiment_b as expb

RESEARCH = Path(__file__).resolve().parent.parent
ALIGN_DIR = RESEARCH / "experiments" / "alignments_c"
PROJECT = Path("C:/Users/Lenovo/Desktop/fastspeech2.0")
SSL_RATE = 50.0
MEL_RATE = 22050.0 / 256.0
MEL_TO_SSL = SSL_RATE / MEL_RATE

# Token IDs treated as silence/pause/structural (not phonetic content).
STRUCTURAL_IDS = {0, 1, 2, 3, 4, 5, 21, 51, 52, 53}


def phone_edges_ssl(alignment: dict) -> tuple[list[int], list[int]]:
    """MAS phone edges in SSL frames: (all tokens, content-only tokens)."""
    durations = alignment["durations"]
    tokens = alignment["tokens"]
    edges_all = []
    edges_content = []
    cum = 0
    for tok, d in zip(tokens[:-1], durations[:-1]):
        cum += d
        edges_all.append(cum)
        if int(tok) not in STRUCTURAL_IDS:
            edges_content.append(cum)
    scale = MEL_TO_SSL
    return (
        [int(round(e * scale)) for e in edges_all],
        [int(round(e * scale)) for e in edges_content],
    )


def match_f1(pred: list[int], ref: list[int], tol: int) -> dict:
    """Greedy one-to-one matching within tolerance."""
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": len(pred), "n_ref": len(ref)}
    used = set()
    hits = 0
    dists = []
    for p in pred:
        cand = [(abs(p - r), r) for r in ref if r not in used]
        if not cand:
            break
        d, r = min(cand)
        if d <= tol:
            hits += 1
            used.add(r)
            dists.append(d)
    precision = hits / len(pred)
    recall = hits / len(ref)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    nearest = [min(abs(p - r) for r in ref) for p in pred] if ref else []
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_dist_frames": round(float(np.mean(nearest)), 2) if nearest else None,
        "median_dist_frames": round(float(np.median(nearest)), 2) if nearest else None,
        "n_pred": len(pred),
        "n_ref": len(ref),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment C: boundary vs phone alignment")
    parser.add_argument("--ks", type=str, default="16,32")
    parser.add_argument("--tolerance", type=int, default=2, help="SSL frames (2 = 40 ms)")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--anneal-steps", type=int, default=1700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tau-start", type=float, default=0.35)
    parser.add_argument("--tau-end", type=float, default=0.08)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=RESEARCH / "experiments" / "results_experiment_c.json")
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

    ks = [int(k) for k in args.ks.split(",")]
    dp_store = json.loads((FEATURE_DIR / f"dp_bounds_pca{args.pca_dim}.json").read_text())
    for it in train_items + test_items:
        it["dp_bounds"] = {k: dp_store[str(k)][it["id"]] for k in ks if str(k) in dp_store}

    # Train the adaptive segmenters with the exact Experiment B recipe.
    models: dict[int, object] = {}
    results: dict = {"config": {**vars(args), "output": str(args.output)}, "arms": {}}
    for k in ks:
        print(f"training adaptive K={k}...", flush=True)
        model, out = train_arm(train_items, make_batches(train_items, args.batch_size),
                               eval_items, eval_batches, k, args, pca)
        models[k] = model
        results["arms"][f"K{k}_final_mse"] = out["final"]["learned_mse"]
        print(f"  K={k} final soft MSE {out['final']['learned_mse']:.4f}", flush=True)

    # Reference edges per eval utterance.
    alignments: dict[str, dict] = {}
    for it in eval_items:
        path = ALIGN_DIR / f"{it['id']}.json"
        if not path.exists():
            raise SystemExit(f"missing alignment for {it['id']}; run extract_alignments_c.py first")
        alignments[it["id"]] = json.loads(path.read_text())

    rng = random.Random(args.seed)
    per_condition: dict[str, list] = {f"learned_K{k}": [] for k in ks}
    per_condition["uniform"] = []
    per_condition["random"] = []
    per_condition_dp = {f"dp_K{k}": [] for k in ks}

    with torch.no_grad():
        for it in eval_items:
            t = it["T"]
            feats_dev = it["features"].unsqueeze(0)
            mask = torch.ones(1, t, device=device)
            ref_all, ref_content = phone_edges_ssl(alignments[it["id"]])

            for k in ks:
                model = models[k]
                _, masks, edge_c, _, _ = model.forward_full(feats_dev, mask)
                learned = warp_boundaries(edge_c[0], t, k)
                per_condition[f"learned_K{k}"].append(match_f1(learned, ref_all, args.tolerance))
                per_condition_dp[f"dp_K{k}"].append(match_f1(it["dp_bounds"][k], ref_all, args.tolerance))

            # Controls use the mean K across the sweep for comparability.
            k_bar = ks[len(ks) // 2]
            uniform_b = expb.uniform_bounds(t, k_bar)
            per_condition["uniform"].append(match_f1(uniform_b, ref_all, args.tolerance))
            random_b = sorted(rng.sample(range(1, t), k_bar - 1))
            per_condition["random"].append(match_f1(random_b, ref_all, args.tolerance))

    summary = {}
    for cond, rows in per_condition.items():
        summary[cond] = {
            metric: round(float(np.mean([r[metric] for r in rows])), 4)
            for metric in ("precision", "recall", "f1", "mean_dist_frames", "median_dist_frames")
        }
    dp_summary = {}
    for cond, rows in per_condition_dp.items():
        dp_summary[cond] = {
            metric: round(float(np.mean([r[metric] for r in rows])), 4)
            for metric in ("precision", "recall", "f1", "mean_dist_frames", "median_dist_frames")
        }
    # Content-only reference variant (excludes silence/pause edges).
    content_summary = {}
    per_condition_content: dict[str, list] = {}
    with torch.no_grad():
        for k in ks:
            rows = []
            model = models[k]
            for it in eval_items:
                t = it["T"]
                _, masks, edge_c, _, _ = model.forward_full(it["features"].unsqueeze(0),
                                                            torch.ones(1, t, device=device))
                learned = warp_boundaries(edge_c[0], t, k)
                rows.append(match_f1(learned, ref_content_of(alignments[it["id"]]), args.tolerance))
            per_condition_content[f"learned_K{k}_vs_content_edges"] = rows
            content_summary[f"learned_K{k}_vs_content_edges"] = {
                metric: round(float(np.mean([r[metric] for r in rows])), 4)
                for metric in ("precision", "recall", "f1", "mean_dist_frames")
            }

    results["summary"] = summary
    results["summary_dp"] = dp_summary
    results["summary_content_edges"] = content_summary
    results["per_utterance"] = {
        **per_condition,
        **per_condition_dp,
        **per_condition_content,
    }
    results["tolerance_ssl_frames"] = args.tolerance
    results["tolerance_ms"] = round(1000 * args.tolerance / SSL_RATE)
    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n=== Experiment C summary (reference: all MAS phone edges) ===", flush=True)
    for cond, s in summary.items():
        print(f"{cond:16s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
              f"meanDist={s['mean_dist_frames']} frames", flush=True)
    for cond, s in dp_summary.items():
        print(f"{cond:16s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
              f"meanDist={s['mean_dist_frames']} frames", flush=True)
    for cond, s in content_summary.items():
        print(f"{cond:36s} F1={s['f1']:.3f}", flush=True)
    print(f"saved={args.output}", flush=True)


def ref_content_of(alignment: dict) -> list[int]:
    _, content = phone_edges_ssl(alignment)
    return content


if __name__ == "__main__":
    main()
