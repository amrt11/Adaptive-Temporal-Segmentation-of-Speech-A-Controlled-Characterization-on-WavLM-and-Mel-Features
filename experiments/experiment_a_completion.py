"""Experiment A completion: robustness, K-sweep, and boundary-perturbation causality.

Continues from experiment_a_continuation.py after the successful
supervised-to-unsupervised result (warm start -> remove boundary loss ->
anneal tau). This implements the remaining Experiment A checklist from the
research notes:

1. Robustness grid: the recipe across clean/noisy conditions, K in {4, 6},
   and multiple training seeds, with boundary F1 at +/-1 frame added to the
   metrics.
2. K-sweep: does boundary recovery hold as K changes?
3. Boundary-perturbation causal test (research notes section 11): move each
   learned boundary by a range of offsets, re-pool segment means under the
   hard piecewise-constant reconstruction family shared with the uniform and
   DP baselines, and measure Delta-MSE. If boundaries are causally useful,
   cost should be minimal at the learned (near-true) positions and grow as a
   boundary moves away into a homogeneous region.

The synthetic decoder has no phase/duration channel, so the perturbation test
cannot be confounded by hidden timing inputs; the u/d ablation from the notes
becomes relevant in Experiment B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiment_a import (
    Batch,
    DensityWarpSegmenter,
    boundary_mae,
    evaluate_classical,
    sample_signals,
    segment_mse,
    set_seed,
    soft_boundary_estimates,
)
from experiment_a_continuation import run_stage2, train_warm_start


def boundary_f1_at_1(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Boundary F1 where a predicted boundary matches if within 1 frame of truth."""
    p = pred.cpu().numpy()
    t = truth.cpu().numpy()
    predicted_hits = sum(
        1
        for i in range(p.shape[0])
        for b in p[i]
        if np.min(np.abs(t[i] - b)) <= 1
    )
    truth_hits = sum(
        1
        for i in range(t.shape[0])
        for b in t[i]
        if np.min(np.abs(p[i] - b)) <= 1
    )
    precision = predicted_hits / max(p.size, 1)
    recall = truth_hits / max(t.size, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def train_and_evaluate(
    args: argparse.Namespace, device: torch.device, test: Batch
) -> tuple[DensityWarpSegmenter, dict[str, float]]:
    """Warm start, anneal, and evaluate on the held-out test set."""
    set_seed(args.seed)
    model = DensityWarpSegmenter(args.dims, args.segments, tau=args.tau_start).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_warm_start(model, opt, args, device, [], test)
    final = run_stage2(model, opt, args, device, [], test, anneal=True, label="anneal")
    with torch.no_grad():
        masks, _, _ = model.masks_and_reconstruction(test.x)
        pred = soft_boundary_estimates(masks)
    final["boundary_f1_at_1"] = boundary_f1_at_1(pred, test.boundaries)
    return model, final


def make_run_args(
    base: argparse.Namespace, noise: float, segments: int, seed: int
) -> argparse.Namespace:
    data = vars(base).copy()
    data.update({"noise": noise, "segments": segments, "seed": seed})
    return argparse.Namespace(**data)


def perturbation_test(
    model: DensityWarpSegmenter,
    test: Batch,
    segments: int,
    t_steps: int,
    deltas: list[int],
) -> dict:
    """Causal boundary test in the hard piecewise-constant reconstruction family.

    Moves each learned boundary by each offset and re-pools segment means from
    the perturbed boundary set using hard frame assignment -- the same
    reconstruction family as the uniform and DP baselines -- so Delta-MSE
    isolates boundary position from soft-edge width and decoder effects. The
    synthetic decoder has no phase/duration inputs, so no hidden timing
    channel can absorb the perturbation. Cost is reported against the
    distance of the moved boundary from its own true transition.
    """
    model.eval()
    x = test.x
    with torch.no_grad():
        masks, recon_native, _ = model.masks_and_reconstruction(x)
    native_mse = float(((x - recon_native) ** 2).mean().item())
    pred = soft_boundary_estimates(masks)
    base_per_signal = [segment_mse(x[i], pred[i].tolist())[0] for i in range(x.shape[0])]
    true_per_signal = [
        segment_mse(x[i], test.boundaries[i].tolist())[0] for i in range(x.shape[0])
    ]
    truth_np = test.boundaries.cpu().numpy()
    records = []
    for k in range(segments - 1):
        for delta in deltas:
            for i in range(x.shape[0]):
                b = pred[i].tolist()
                lo = (b[k - 1] + 1) if k > 0 else 1
                hi = (b[k + 1] - 1) if k < segments - 2 else t_steps - 1
                moved = int(np.clip(b[k] + delta, lo, hi))
                if moved == b[k]:
                    continue
                moved_list = list(b)
                moved_list[k] = moved
                mse_i = segment_mse(x[i], moved_list)[0]
                records.append(
                    {
                        "boundary": k,
                        "requested_delta": int(delta),
                        "actual_delta": int(moved - b[k]),
                        "distance_to_true_boundary": int(abs(moved - truth_np[i][k])),
                        "delta_mse": round(float(mse_i - base_per_signal[i]), 8),
                    }
                )

    def bucket(distance: int) -> str:
        if distance <= 1:
            return "0-1"
        if distance <= 4:
            return "2-4"
        if distance <= 8:
            return "5-8"
        return "9+"

    by_delta: dict[int, list[float]] = {}
    by_distance: dict[str, list[float]] = {}
    by_boundary: dict[int, list[float]] = {}
    for r in records:
        by_delta.setdefault(r["actual_delta"], []).append(r["delta_mse"])
        by_distance.setdefault(bucket(r["distance_to_true_boundary"]), []).append(r["delta_mse"])
        by_boundary.setdefault(r["boundary"], []).append(r["delta_mse"])

    def stats(values: list[float]) -> dict:
        return {
            "mean": round(float(np.mean(values)), 8) if values else None,
            "count": len(values),
        }

    return {
        "native_model_mse": round(native_mse, 8),
        "hard_mse_at_learned_boundaries": round(float(np.mean(base_per_signal)), 8),
        "hard_mse_at_true_boundaries": round(float(np.mean(true_per_signal)), 8),
        "cost_by_actual_delta": {
            str(d): stats(v) for d, v in sorted(by_delta.items())
        },
        "cost_by_distance_to_true_boundary": {
            b: stats(v) for b, v in sorted(by_distance.items())
        },
        "cost_by_boundary_index": {
            str(k): stats(v) for k, v in sorted(by_boundary.items())
        },
        "_records": records,
    }


def summarize_grid(grid: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    by_condition: dict[str, list[dict]] = {}
    for row in grid:
        by_condition.setdefault(row["condition"], []).append(row)
    for condition, rows in by_condition.items():
        mse = [r["mse"] for r in rows]
        mae = [r["boundary_mae_frames"] for r in rows]
        f1 = [r["boundary_f1_at_1"] for r in rows]
        g = [r["gap_closed"] for r in rows]
        summary[condition] = {
            "seeds": [r["seed"] for r in rows],
            "mse_mean": round(float(np.mean(mse)), 6),
            "mse_std": round(float(np.std(mse)), 6),
            "boundary_mae_mean": round(float(np.mean(mae)), 4),
            "boundary_mae_std": round(float(np.std(mae)), 4),
            "f1_at_1_mean": round(float(np.mean(f1)), 4),
            "gap_closed_mean": round(float(np.mean(g)), 4),
            "gap_closed_std": round(float(np.std(g)), 4),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A completion suite")
    parser.add_argument("--warmup-steps", type=int, default=800)
    parser.add_argument("--anneal-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tau-start", type=float, default=0.35)
    parser.add_argument("--tau-end", type=float, default=0.08)
    parser.add_argument("--boundary-weight", type=float, default=0.35)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--seeds", type=str, default="21,42")
    parser.add_argument("--deltas", type=str, default="-16,-12,-8,-4,-2,2,4,8,12,16")
    parser.add_argument("--sweep-ks", type=str, default="2,3,8")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-ksweep", action="store_true")
    parser.add_argument("--skip-perturbation", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results_experiment_a_completion.json")
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    deltas = [int(d) for d in args.deltas.split(",")]
    sweep_ks = [int(k) for k in args.sweep_ks.split(",")]
    print(f"device={device} seeds={seeds}", flush=True)

    conditions = [
        (0.05, 4, "clean_K4"),
        (0.30, 4, "noisy_K4"),
        (0.05, 6, "clean_K6"),
        (0.30, 6, "noisy_K6"),
    ]

    # Held-out test sets and classical baselines are shared across seeds.
    test_sets: dict[tuple[float, int], tuple[Batch, dict[str, float]]] = {}
    for noise, segments, _ in conditions:
        set_seed(9000 + segments * 100 + int(noise * 1000))
        test = sample_signals(
            args.test_size, args.t_steps, args.dims, segments, noise, args.min_len, device
        )
        classical = evaluate_classical(test, segments)
        test_sets[(noise, segments)] = (test, classical)
        print(
            f"condition noise={noise} K={segments}: uniform={classical['uniform_mse']:.4f} "
            f"dp={classical['dp_mse']:.4f}",
            flush=True,
        )

    result: dict = {
        "device": str(device),
        "config": {**vars(args), "output": str(args.output)},
        "grid": [],
        "grid_summary": {},
        "k_sweep": [],
        "perturbation": {},
    }
    if args.output.exists():
        try:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
        if args.skip_grid and previous.get("grid"):
            result["grid"] = previous["grid"]
            print(f"loaded previous grid rows: {len(result['grid'])}", flush=True)
        if args.skip_ksweep and previous.get("k_sweep"):
            result["k_sweep"] = previous["k_sweep"]
            print(f"loaded previous k_sweep rows: {len(result['k_sweep'])}", flush=True)
    models: dict[tuple[float, int, int], DensityWarpSegmenter] = {}

    def save() -> None:
        result["grid_summary"] = summarize_grid(result["grid"])
        args.output.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

    if not args.skip_grid:
        for noise, segments, label in conditions:
            test, classical = test_sets[(noise, segments)]
            for seed in seeds:
                run_args = make_run_args(args, noise, segments, seed)
                model, final = train_and_evaluate(run_args, device, test)
                row = {
                    "condition": label,
                    "noise": noise,
                    "segments": segments,
                    "seed": seed,
                    **classical,
                    **final,
                }
                gap = classical["uniform_mse"] - classical["dp_mse"]
                row["gap_closed"] = (
                    (classical["uniform_mse"] - final["mse"]) / gap if gap > 1e-12 else None
                )
                result["grid"].append(row)
                if seed == seeds[0]:
                    models[(noise, segments, seed)] = model
                save()
                print(
                    f"[grid] {label} seed={seed}: mse={final['mse']:.4f} "
                    f"mae={final['boundary_mae_frames']:.2f} "
                    f"f1@1={final['boundary_f1_at_1']:.3f} G={row['gap_closed']:.3f}",
                    flush=True,
                )

    if not args.skip_ksweep:
        noise = 0.05
        for segments in sweep_ks:
            if (noise, segments) not in test_sets:
                set_seed(9000 + segments * 100 + int(noise * 1000))
                test = sample_signals(
                    args.test_size, args.t_steps, args.dims, segments, noise, args.min_len, device
                )
                classical = evaluate_classical(test, segments)
                test_sets[(noise, segments)] = (test, classical)
            test, classical = test_sets[(noise, segments)]
            model_key = (noise, segments, seeds[0])
            if model_key in models:
                model = models[model_key]
                trained = False
                with torch.no_grad():
                    masks, recon, _ = model.masks_and_reconstruction(test.x)
                    pred = soft_boundary_estimates(masks)
                final = {
                    "mse": float(((test.x - recon) ** 2).mean().item()),
                    "boundary_mae_frames": boundary_mae(pred, test.boundaries),
                    "boundary_f1_at_1": boundary_f1_at_1(pred, test.boundaries),
                }
            else:
                run_args = make_run_args(args, noise, segments, seeds[0])
                model, final = train_and_evaluate(run_args, device, test)
                models[model_key] = model
                trained = True
            gap = classical["uniform_mse"] - classical["dp_mse"]
            row = {
                "condition": f"clean_K{segments}",
                "noise": noise,
                "segments": segments,
                "seed": seeds[0],
                "reused_from_grid": not trained,
                **classical,
                **final,
                "gap_closed": (
                    (classical["uniform_mse"] - final["mse"]) / gap if gap > 1e-12 else None
                ),
            }
            result["k_sweep"].append(row)
            save()
            print(
                f"[k_sweep] K={segments}: mse={final['mse']:.4f} "
                f"mae={final['boundary_mae_frames']:.2f} "
                f"f1@1={final['boundary_f1_at_1']:.3f} G={row['gap_closed']:.3f}",
                flush=True,
            )

    if not args.skip_perturbation:
        for noise, segments, label in conditions:
            if segments != 4:
                continue
            model = models.get((noise, segments, seeds[0]))
            if model is None:
                run_args = make_run_args(args, noise, segments, seeds[0])
                model, _ = train_and_evaluate(run_args, device, test_sets[(noise, segments)][0])
                models[(noise, segments, seeds[0])] = model
            test, _ = test_sets[(noise, segments)]
            pert = perturbation_test(model, test, segments, args.t_steps, deltas)
            records = pert.pop("_records", [])
            if label == "clean_K4":
                result["perturbation"]["records_clean_K4"] = records
            result["perturbation"][label] = pert
            save()
            dist = pert["cost_by_distance_to_true_boundary"]
            print(
                f"[perturb] {label}: hard@learned={pert['hard_mse_at_learned_boundaries']:.4f} "
                f"hard@true={pert['hard_mse_at_true_boundaries']:.4f} "
                f"cost@0-1={dist.get('0-1', {}).get('mean')} "
                f"cost@2-4={dist.get('2-4', {}).get('mean')} "
                f"cost@5-8={dist.get('5-8', {}).get('mean')} "
                f"cost@9+={dist.get('9+', {}).get('mean')}",
                flush=True,
            )

    save()
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
