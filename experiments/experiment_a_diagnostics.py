"""Follow-up diagnostics for Experiment A.

This separates parameterization correctness from unsupervised identifiability:
  1. oracle density/warp
  2. fixed true-boundary reconstruction
  3. temporary boundary-supervised warm start
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiment_a import (
    DensityWarpSegmenter,
    boundary_mae,
    evaluate_classical,
    sample_signals,
    set_seed,
    soft_boundary_estimates,
)


def masks_from_density(rho: torch.Tensor, segments: int, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    cumulative = torch.cumsum(
        torch.cat([torch.zeros_like(rho[:, :1]), rho], dim=1), dim=1
    )
    edge_c = segments * cumulative / cumulative[:, -1:].clamp_min(1e-6)
    u = 0.5 * (edge_c[:, :-1] + edge_c[:, 1:])
    transitions = [torch.ones_like(u[:, :1])]
    for k in range(1, segments):
        transitions.append(torch.sigmoid((u - float(k)) / tau))
    transitions.append(torch.zeros_like(u[:, :1]))
    masks = torch.stack([transitions[k] - transitions[k + 1] for k in range(segments)], dim=1)
    return masks, edge_c


def reconstruct_from_masks(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    mass = masks.sum(dim=2, keepdim=True).clamp_min(1e-6)
    z = torch.einsum("bkt,btd->bkd", masks, x) / mass
    return torch.einsum("bkt,bkd->btd", masks, z)


def oracle_test(batch: object, segments: int, tau: float) -> dict[str, float]:
    x = batch.x
    endpoints = torch.cat([
        torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device),
        batch.boundaries,
        torch.full((x.shape[0], 1), x.shape[1], dtype=torch.long, device=x.device),
    ], dim=1)
    lengths = torch.diff(endpoints, dim=1).float()
    # Each true segment receives equal cumulative mass, so C(b_k)=k.
    rho = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
    for i in range(x.shape[0]):
        for k in range(segments):
            rho[i, endpoints[i, k] : endpoints[i, k + 1]] = 1.0 / lengths[i, k]
    masks, edge_c = masks_from_density(rho, segments, tau)
    recon = reconstruct_from_masks(x, masks)
    pred = soft_boundary_estimates(masks)
    return {
        "oracle_mse": float(((x - recon) ** 2).mean().item()),
        "oracle_boundary_mae_frames": boundary_mae(pred, batch.boundaries),
        "oracle_partition_error": float((masks.sum(dim=1) - 1).abs().max().item()),
        "oracle_warp_error": float(torch.stack([edge_c[i, batch.boundaries[i]] for i in range(x.shape[0])], dim=0).sub(torch.arange(1, segments, device=x.device)).abs().mean().item()),
    }


def supervised_warm_start(steps: int, segments: int, device: torch.device) -> dict[str, float]:
    model = DensityWarpSegmenter(4, segments, tau=0.35).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(steps):
        batch = sample_signals(64, 128, 4, segments, 0.05, 8, device)
        masks, recon, edge_c = model.masks_and_reconstruction(batch.x)
        targets = torch.arange(1, segments, device=device).float()[None, :].expand(batch.x.shape[0], -1)
        # Boundary positions are edges: C(b_k) should equal k.
        c_at_boundary = torch.stack([edge_c[i, batch.boundaries[i]] for i in range(batch.x.shape[0])], dim=0)
        boundary_loss = ((c_at_boundary - targets) ** 2).mean()
        recon_loss = ((batch.x - recon) ** 2).mean()
        loss = recon_loss + 0.25 * boundary_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    test = sample_signals(256, 128, 4, segments, 0.05, 8, device)
    model.eval()
    with torch.no_grad():
        masks, recon, _ = model.masks_and_reconstruction(test.x)
        pred = soft_boundary_estimates(masks)
    return {
        "warm_mse": float(((test.x - recon) ** 2).mean().item()),
        "warm_boundary_mae_frames": boundary_mae(pred, test.boundaries),
        "warm_partition_error": float((masks.sum(dim=1) - 1).abs().max().item()),
    }


def main() -> None:
    set_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segments = 4
    batch = sample_signals(256, 128, 4, segments, 0.05, 8, device)
    result = {"device": str(device), "oracle": oracle_test(batch, segments, 0.15)}
    result["classical"] = evaluate_classical(batch, segments)
    result["supervised_warm_start"] = supervised_warm_start(1200, segments, device)
    out = Path("experiments/results_experiment_a_diagnostics.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
