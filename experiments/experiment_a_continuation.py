"""Supervised-to-unsupervised continuation for Experiment A.

Stage 1 trains with a temporary boundary loss (warm start). Stage 2 removes
the boundary loss and anneals the mask temperature while continuing the
reconstruction objective. The question is whether the learned partition
survives without direct boundary supervision.

A fixed-temperature control arm repeats Stage 2 without annealing, starting
from the same warm-started weights and optimizer state, so the effect of
annealing can be separated from the effect of removing the boundary loss.

Outputs experiments/results_experiment_a_continuation.json with the metric
trajectory, final metrics for both arms, uniform/DP baselines, gap-closed
values, and a small dump of predicted vs. true boundaries.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

from experiment_a import (
    Batch,
    DensityWarpSegmenter,
    boundary_mae,
    evaluate_classical,
    sample_signals,
    set_seed,
    soft_boundary_estimates,
)


def boundary_targets(edge_c: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """C evaluated at the true boundary edges; each should equal its index k."""
    return torch.stack([edge_c[i, boundaries[i]] for i in range(boundaries.shape[0])], dim=0)


def evaluate(model: DensityWarpSegmenter, test: Batch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        masks, recon, _ = model.masks_and_reconstruction(test.x)
        pred = soft_boundary_estimates(masks)
    return {
        "mse": float(((test.x - recon) ** 2).mean().item()),
        "boundary_mae_frames": boundary_mae(pred, test.boundaries),
        "partition_max_error": float((masks.sum(dim=1) - 1).abs().max().item()),
    }


def log_eval(
    history: list[dict],
    model: DensityWarpSegmenter,
    test: Batch,
    stage: str,
    step: int,
    train_recon: float,
    extra: dict[str, float],
) -> None:
    metrics = evaluate(model, test)
    metrics.update(
        {
            "stage": stage,
            "step": step,
            "tau": model.tau,
            "train_recon_loss": train_recon,
            **extra,
        }
    )
    history.append(metrics)
    extras = " ".join(f"{k}={v:.4f}" for k, v in extra.items())
    print(
        f"[{stage} step={step:5d}] tau={model.tau:.3f} "
        f"test_mse={metrics['mse']:.4f} test_mae={metrics['boundary_mae_frames']:.2f} "
        f"train_recon={train_recon:.4f} {extras}"
    )


def train_warm_start(
    model: DensityWarpSegmenter,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    history: list[dict],
    test: Batch,
) -> None:
    """Stage 1: reconstruction plus a temporary boundary loss on C(b_k)=k."""
    for step in range(1, args.warmup_steps + 1):
        model.train()
        batch = sample_signals(
            args.batch_size, args.t_steps, args.dims, args.segments, args.noise, args.min_len, device
        )
        masks, recon, edge_c = model.masks_and_reconstruction(batch.x)
        targets = torch.arange(1, args.segments, device=device).float()[None, :].expand(
            batch.x.shape[0], -1
        )
        c_boundary = boundary_targets(edge_c, batch.boundaries)
        boundary_loss = ((c_boundary - targets) ** 2).mean()
        recon_loss = ((batch.x - recon) ** 2).mean()
        loss = recon_loss + args.boundary_weight * boundary_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.warmup_steps:
            log_eval(
                history, model, test, "supervised", step,
                float(recon_loss.item()), {"boundary_loss": float(boundary_loss.item())},
            )


def run_stage2(
    model: DensityWarpSegmenter,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    history: list[dict],
    test: Batch,
    anneal: bool,
    label: str,
) -> dict[str, float]:
    """Stage 2: reconstruction only. Anneal tau if requested, else keep it fixed."""
    set_seed(args.seed + 1)  # identical batch stream across arms
    for step in range(1, args.anneal_steps + 1):
        model.train()
        if anneal and args.anneal_steps > 0:
            progress = step / args.anneal_steps
            model.tau = args.tau_start * (args.tau_end / args.tau_start) ** progress
        batch = sample_signals(
            args.batch_size, args.t_steps, args.dims, args.segments, args.noise, args.min_len, device
        )
        recon, masks = model(batch.x)
        recon_loss = ((batch.x - recon) ** 2).mean()
        entropy = -(masks.clamp_min(1e-8) * masks.clamp_min(1e-8).log()).mean()
        loss = recon_loss + args.entropy_weight * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.anneal_steps:
            log_eval(
                history, model, test, label, step,
                float(recon_loss.item()), {"entropy": float(entropy.item())},
            )
    return evaluate(model, test)


def gap_closed(mse: float, uniform_mse: float, dp_mse: float) -> float:
    gap = uniform_mse - dp_mse
    return (uniform_mse - mse) / gap if gap > 1e-12 else math.nan


def dump_samples(model: DensityWarpSegmenter, test: Batch, count: int = 8) -> list[dict]:
    """Predicted vs. true boundaries for a few test signals; masks for the first."""
    model.eval()
    with torch.no_grad():
        masks, _, _ = model.masks_and_reconstruction(test.x)
        pred = soft_boundary_estimates(masks)
    samples = []
    for i in range(min(count, test.x.shape[0])):
        entry = {
            "index": i,
            "true_boundaries": test.boundaries[i].tolist(),
            "predicted_boundaries": pred[i].tolist(),
        }
        if i == 0:
            entry["masks"] = [[round(float(v), 4) for v in row] for row in masks[i]]
        samples.append(entry)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised-to-unsupervised continuation for Experiment A")
    parser.add_argument("--warmup-steps", type=int, default=800)
    parser.add_argument("--anneal-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tau-start", type=float, default=0.35)
    parser.add_argument("--tau-end", type=float, default=0.08)
    parser.add_argument("--boundary-weight", type=float, default=0.35)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--skip-control", action="store_true", help="skip the fixed-tau control arm")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output", type=Path, default=Path("experiments/results_experiment_a_continuation.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Reproducible held-out test set, generated before any training RNG use.
    set_seed(args.seed + 9999)
    test = sample_signals(
        args.test_size, args.t_steps, args.dims, args.segments, args.noise, args.min_len, device
    )
    classical = evaluate_classical(test, args.segments)
    print(
        f"classical: uniform_mse={classical['uniform_mse']:.4f} "
        f"dp_mse={classical['dp_mse']:.4f}"
    )

    history: list[dict] = []

    # Stage 1: boundary-supervised warm start.
    set_seed(args.seed)
    model = DensityWarpSegmenter(args.dims, args.segments, tau=args.tau_start).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_warm_start(model, opt, args, device, history, test)
    supervised_metrics = evaluate(model, test)
    warm_model_state = copy.deepcopy(model.state_dict())
    warm_opt_state = copy.deepcopy(opt.state_dict())

    # Stage 2 main arm: remove the boundary loss and anneal tau.
    final_metrics = run_stage2(
        model, opt, args, device, history, test, anneal=True, label="unsupervised_anneal"
    )

    # Control arm: identical warm start, identical batch stream, fixed tau.
    control_final: dict[str, float] | None = None
    if not args.skip_control:
        set_seed(args.seed)
        cmodel = DensityWarpSegmenter(args.dims, args.segments, tau=args.tau_start).to(device)
        cmodel.load_state_dict(warm_model_state)
        copt = torch.optim.Adam(cmodel.parameters(), lr=args.lr)
        copt.load_state_dict(copy.deepcopy(warm_opt_state))
        control_final = run_stage2(
            cmodel, copt, args, device, history, test, anneal=False, label="unsupervised_fixed_tau"
        )

    def g(mse: float) -> float:
        return gap_closed(mse, classical["uniform_mse"], classical["dp_mse"])

    result = {
        "device": str(device),
        "config": {**vars(args), "output": str(args.output)},
        "classical": classical,
        "supervised_metrics": supervised_metrics,
        "final_metrics": final_metrics,
        "control_final_metrics": control_final,
        "gap_closed": {
            "supervised": g(supervised_metrics["mse"]),
            "final_annealed": g(final_metrics["mse"]),
            **({"control_fixed_tau": g(control_final["mse"])} if control_final else {}),
        },
        "history": history,
        "samples": dump_samples(model, test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    print(
        f"uniform_mse={classical['uniform_mse']:.4f}  dp_mse={classical['dp_mse']:.4f}"
    )
    print(
        f"warm start (supervised):   mse={supervised_metrics['mse']:.4f} "
        f"mae={supervised_metrics['boundary_mae_frames']:.2f}  G={g(supervised_metrics['mse']):.3f}"
    )
    print(
        f"after removal (annealed):  mse={final_metrics['mse']:.4f} "
        f"mae={final_metrics['boundary_mae_frames']:.2f}  G={g(final_metrics['mse']):.3f}"
    )
    if control_final is not None:
        print(
            f"control (fixed tau):       mse={control_final['mse']:.4f} "
            f"mae={control_final['boundary_mae_frames']:.2f}  G={g(control_final['mse']):.3f}"
        )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
