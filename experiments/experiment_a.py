"""Experiment A: synthetic sanity check for adaptive monotonic segmentation.

The model predicts a positive per-frame density, warps time with its cumulative
sum, and creates exact-partition soft masks from sigmoid transitions in warped
coordinates. Segment means reconstruct piecewise-constant synthetic signals.

Baselines:
  * uniform K-segment reconstruction
  * dynamic-programming optimal K-segment reconstruction
  * learned adaptive density-warp segmentation

This intentionally excludes VQ, TTS, and a powerful decoder. It tests only the
segmentation parameterization and its reconstruction/boundary behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class Batch:
    x: torch.Tensor
    boundaries: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_signals(
    batch_size: int,
    t_steps: int,
    dims: int,
    segments: int,
    noise_std: float,
    min_len: int,
    device: torch.device,
) -> Batch:
    """Sample piecewise-constant signals with random valid boundaries."""
    xs = []
    all_boundaries = []
    for _ in range(batch_size):
        # Sample positive random lengths, then repair rounding while preserving
        # a minimum segment length.
        raw = np.random.dirichlet(np.ones(segments))
        lengths = np.maximum(min_len, np.floor(raw * (t_steps - min_len * segments)).astype(int) + min_len)
        while lengths.sum() > t_steps:
            j = int(np.argmax(lengths))
            if lengths[j] > min_len:
                lengths[j] -= 1
        while lengths.sum() < t_steps:
            lengths[int(np.argmin(lengths))] += 1
        boundaries = np.cumsum(lengths)[:-1]
        values = np.random.normal(size=(segments, dims)).astype(np.float32)
        # Keep neighboring regions distinguishable but not unrealistically far apart.
        values = values / np.maximum(values.std(axis=0, keepdims=True), 1e-3)
        x = np.repeat(values, lengths, axis=0)
        x += np.random.normal(scale=noise_std, size=x.shape).astype(np.float32)
        xs.append(x)
        all_boundaries.append(boundaries.astype(np.int64))
    return Batch(
        x=torch.tensor(np.stack(xs), dtype=torch.float32, device=device),
        boundaries=torch.tensor(np.stack(all_boundaries), dtype=torch.long, device=device),
    )


def segment_mse(x: torch.Tensor, boundaries: list[int]) -> tuple[float, torch.Tensor]:
    """Piecewise-constant reconstruction from explicit boundaries."""
    starts = [0] + boundaries
    ends = boundaries + [x.shape[0]]
    out = torch.zeros_like(x)
    for start, end in zip(starts, ends):
        out[start:end] = x[start:end].mean(dim=0, keepdim=True)
    return float(((x - out) ** 2).mean().item()), out


def uniform_reconstruction(x: torch.Tensor, segments: int) -> tuple[float, list[int]]:
    t_steps = x.shape[0]
    boundaries = [round(i * t_steps / segments) for i in range(1, segments)]
    mse, _ = segment_mse(x, boundaries)
    return mse, boundaries


def dp_reconstruction(x: torch.Tensor, segments: int) -> tuple[float, list[int]]:
    """Exact optimal piecewise-constant segmentation for squared error."""
    # The oracle is evaluation-only. Compute all segment costs once on CPU with
    # NumPy; the DP itself then needs only O(K*T^2) scalar operations.
    arr = x.detach().cpu().numpy().astype(np.float64)
    t_steps = arr.shape[0]
    prefix = np.vstack([np.zeros((1, arr.shape[1])), np.cumsum(arr, axis=0)])
    prefix2 = np.concatenate([[0.0], np.cumsum((arr * arr).sum(axis=1))])
    idx = np.arange(t_steps + 1)
    lengths = idx[None, :] - idx[:, None]
    sums = prefix[None, :, :] - prefix[:, None, :]
    sse = prefix2[None, :] - prefix2[:, None] - (sums * sums).sum(axis=2) / np.maximum(lengths, 1)
    sse[lengths <= 0] = np.inf

    dp = np.full((segments + 1, t_steps + 1), np.inf, dtype=np.float64)
    prev = np.full((segments + 1, t_steps + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, segments + 1):
        for end in range(k, t_steps + 1):
            starts = np.arange(k - 1, end)
            vals = dp[k - 1, starts] + sse[starts, end]
            best_offset = int(np.argmin(vals))
            best = int(starts[best_offset])
            dp[k, end] = vals[best_offset]
            prev[k, end] = best
    boundaries = []
    end = t_steps
    for k in range(segments, 1, -1):
        end = int(prev[k, end])
        boundaries.append(end)
    boundaries.reverse()
    mse, _ = segment_mse(x, boundaries)
    return mse, boundaries


class DensityWarpSegmenter(nn.Module):
    def __init__(self, dims: int, segments: int, rho_min: float = 0.05, tau: float = 0.15):
        super().__init__()
        self.segments = segments
        self.rho_min = rho_min
        self.tau = tau
        # Small temporal-context predictor. It sees local values and first
        # differences, but has no reconstruction shortcut or segment labels.
        self.density = nn.Sequential(
            nn.Conv1d(dims * 2, 16, kernel_size=7, padding=3),
            nn.Tanh(),
            nn.Conv1d(16, 16, kernel_size=7, padding=3),
            nn.Tanh(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def masks_and_reconstruction(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,T,D]
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        features = torch.cat([x, delta.abs()], dim=-1).transpose(1, 2)
        rho = self.rho_min + torch.nn.functional.softplus(self.density(features).squeeze(1))
        # Construct C on frame edges, then evaluate it at frame centers.
        cumulative = torch.cumsum(
            torch.cat([torch.zeros_like(rho[:, :1]), rho], dim=1), dim=1
        )
        edge_c = self.segments * cumulative / cumulative[:, -1:].clamp_min(1e-6)
        u = 0.5 * (edge_c[:, :-1] + edge_c[:, 1:])
        transitions = [torch.ones_like(u[:, :1])]
        for k in range(1, self.segments):
            transitions.append(torch.sigmoid((u - float(k)) / self.tau))
        transitions.append(torch.zeros_like(u[:, :1]))
        masks = torch.stack(
            [transitions[k] - transitions[k + 1] for k in range(self.segments)], dim=1
        )
        # Segment means and reconstruction.
        mass = masks.sum(dim=2, keepdim=True).clamp_min(1e-6)
        z = torch.einsum("bkt,btd->bkd", masks, x) / mass
        recon = torch.einsum("bkt,bkd->btd", masks, z)
        return masks, recon, edge_c

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks, recon, _ = self.masks_and_reconstruction(x)
        return recon, masks


def soft_boundary_estimates(masks: torch.Tensor) -> torch.Tensor:
    """Estimate internal boundaries from mask crossings."""
    # masks: [B,K,T]; boundary k is where cumulative mass reaches 0.5.
    cumulative = masks.cumsum(dim=1)
    estimates = []
    for k in range(1, masks.shape[1]):
        score = (cumulative[:, k - 1] - 0.5).abs()
        estimates.append(score.argmin(dim=1))
    return torch.stack(estimates, dim=1)


def boundary_mae(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred.float() - truth.float()).abs().mean().item())


def evaluate_model(model: DensityWarpSegmenter, batch: Batch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        recon, masks = model(batch.x)
        pred_b = soft_boundary_estimates(masks)
        mse = float(((batch.x - recon) ** 2).mean().item())
        mae = boundary_mae(pred_b, batch.boundaries)
        exact_partition = float((masks.sum(dim=1) - 1.0).abs().max().item())
    return {"mse": mse, "boundary_mae_frames": mae, "partition_max_error": exact_partition}


def evaluate_classical(batch: Batch, segments: int) -> dict[str, float]:
    uniform_mse = []
    dp_mse = []
    dp_boundary_mae = []
    for x, truth in zip(batch.x, batch.boundaries):
        umse, _ = uniform_reconstruction(x, segments)
        dmse, db = dp_reconstruction(x, segments)
        uniform_mse.append(umse)
        dp_mse.append(dmse)
        dp_boundary_mae.append(float(np.mean(np.abs(np.asarray(db) - truth.cpu().numpy()))))
    return {
        "uniform_mse": float(np.mean(uniform_mse)),
        "dp_mse": float(np.mean(dp_mse)),
        "dp_boundary_mae_frames": float(np.mean(dp_boundary_mae)),
    }


def train_model(args: argparse.Namespace, device: torch.device) -> DensityWarpSegmenter:
    model = DensityWarpSegmenter(args.dims, args.segments, tau=args.tau).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        model.train()
        batch = sample_signals(args.batch_size, args.t_steps, args.dims, args.segments, args.noise, args.min_len, device)
        recon, masks = model(batch.x)
        loss = ((batch.x - recon) ** 2).mean()
        # Encourage sharp but do not force hard masks prematurely.
        entropy = -(masks.clamp_min(1e-8) * masks.clamp_min(1e-8).log()).mean()
        total = loss + args.entropy_weight * entropy
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step={step:5d} recon={loss.item():.6f} entropy={entropy.item():.6f}")
    return model


def run_condition(args: argparse.Namespace, noise: float, segments: int, label: str, device: torch.device) -> dict:
    local = argparse.Namespace(**vars(args))
    local.noise = noise
    local.segments = segments
    set_seed(args.seed + segments * 100 + int(noise * 1000))
    model = train_model(local, device)
    test = sample_signals(args.test_size, args.t_steps, args.dims, segments, noise, args.min_len, device)
    learned = evaluate_model(model, test)
    classical = evaluate_classical(test, segments)
    result = {"condition": label, "noise_std": noise, "segments": segments, **classical, **learned}
    uniform_gap = result["uniform_mse"] - result["dp_mse"]
    result["learned_gap_closed"] = ((result["uniform_mse"] - result["mse"]) / uniform_gap) if uniform_gap > 1e-12 else math.nan
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("experiments/results_experiment_a.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    results = []
    for noise, segments, label in [
        (0.05, 4, "clean_unequal_4_segments"),
        (0.30, 4, "noisy_unequal_4_segments"),
        (0.05, 6, "clean_unequal_6_segments"),
    ]:
        results.append(run_condition(args, noise, segments, label, device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": vars(args), "results": results}, indent=2, default=str), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
