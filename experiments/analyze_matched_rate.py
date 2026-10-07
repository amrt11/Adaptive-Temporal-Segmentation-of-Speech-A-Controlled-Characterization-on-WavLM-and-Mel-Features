"""Analyze matched-rate RD results and generate matched-bitrate comparisons.

Method (aligned with the review's P0 requirement):
1. Load experiment points (per arm, per seed).
2. Pareto-filter each arm's measured (rate, MSE) points into a monotone
   lower envelope: keep points not dominated by a cheaper point with
   lower-or-equal MSE. For segment arms the envelope is over all (K, N)
   configurations, i.e. "best segment system at rate R".
3. Evaluate each envelope on a common bitrate grid via linear interpolation,
   ONLY inside the arm's measured rate range (no extrapolation; outside is
   NaN and flagged).
4. Matched-rate comparison is reported only at grid points inside ALL
   compared arms' measured ranges (the genuine overlap window).
5. With multiple seeds, per-seed paired comparisons are aggregated
   (mean +/- std across seeds) instead of interpolating seed-averaged curves.

Output: experiments/analysis_matched_rate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def load_results(filepath: Path) -> dict:
    return json.loads(filepath.read_text(encoding="utf-8"))


def pareto_envelope(rates: np.ndarray, mses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower envelope of an RD scatter: increasing rate, strictly decreasing MSE."""
    order = np.argsort(rates, kind="stable")
    rates, mses = rates[order], mses[order]
    keep_r, keep_d, best = [], [], np.inf
    for r, d in zip(rates, mses):
        if d < best:  # strictly better than anything cheaper
            keep_r.append(r)
            keep_d.append(d)
            best = d
    return np.array(keep_r), np.array(keep_d)


def interp_envelope(points: list[dict], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Interpolate one seed's/arm's points to the grid; NaN outside measured range."""
    rates = np.array([p["bits_per_sec_exact"] for p in points], dtype=float)
    mses = np.array([p["mse"] for p in points], dtype=float)
    env_r, env_d = pareto_envelope(rates, mses)
    out = np.full(grid.shape, np.nan)
    if len(env_r) >= 2:
        inside = (grid >= env_r[0]) & (grid <= env_r[-1])
        out[inside] = np.interp(grid[inside], env_r, env_d)
    elif len(env_r) == 1:
        out[grid == env_r[0]] = env_d[0]
    return out, env_d, float(env_r.min()) if len(env_r) else np.nan, float(env_r.max()) if len(env_r) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched-rate RD results")
    parser.add_argument("--input", type=Path,
                        default=Path("experiments/results_matched_rate_multiseed.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/analysis_matched_rate.json"))
    parser.add_argument("--rate-grid", type=str,
                        default="25,30,35,40,45,50,55,60,65,70,75,80,90,100,110,120,130,150,175,200,250,300",
                        help="Bitrate grid for comparison (bits/s)")
    parser.add_argument("--baseline", type=str, default="frame",
                        help="Arm treated as the baseline in comparisons")
    parser.add_argument("--compare", type=str, default="adaptive,uniform",
                        help="Arms compared against the baseline")
    args = parser.parse_args()

    results = load_results(args.input)
    points = results["points"]
    print(f"Loaded {len(points)} points from {args.input}", flush=True)

    grid = np.array([float(r) for r in args.rate_grid.split(",")])

    # Group points: arm -> seed -> [points]
    by_arm_seed: dict[str, dict[int, list[dict]]] = {}
    for p in points:
        by_arm_seed.setdefault(p["arm"], {}).setdefault(int(p["seed"]), []).append(p)

    seeds = sorted({s for m in by_arm_seed.values() for s in m})
    arms = sorted(by_arm_seed.keys())
    print(f"Arms: {arms}  Seeds: {seeds}", flush=True)

    # Per arm: per-seed grid interpolants + measured range
    interp_by_arm: dict[str, dict[int, np.ndarray]] = {}
    range_by_arm: dict[str, tuple[float, float]] = {}
    for arm in arms:
        interp_by_arm[arm] = {}
        lo, hi = np.inf, -np.inf
        for seed in seeds:
            pts = by_arm_seed[arm].get(seed, [])
            if not pts:
                continue
            vals, _, rmin, rmax = interp_envelope(pts, grid)
            interp_by_arm[arm][seed] = vals
            lo, hi = min(lo, rmin), max(hi, rmax)
        range_by_arm[arm] = (lo, hi)
        print(f"arm={arm:9s} measured rate range: {lo:.1f} .. {hi:.1f} b/s", flush=True)

    # Overlap windows: one per compared pair (a global all-arms window is
    # over-restricted when an arm's Pareto envelope is non-monotone, e.g. the
    # uniform arm gets worse with rate), plus the global window for reference.
    base = args.baseline
    others = [a for a in args.compare.split(",") if a in by_arm_seed and a != base]
    pair_windows = {}
    for arm in others:
        lo = max(range_by_arm[base][0], range_by_arm[arm][0])
        hi = min(range_by_arm[base][1], range_by_arm[arm][1])
        pair_windows[arm] = (lo, hi)
        print(f"\nPair window {base} vs {arm}: {lo:.1f} - {hi:.1f} b/s", flush=True)
    ov_lo = max(range_by_arm[a][0] for a in arms)
    ov_hi = min(range_by_arm[a][1] for a in arms)
    print(f"Global all-arms overlap window: {ov_lo:.1f} - {ov_hi:.1f} b/s", flush=True)

    # Matched-rate comparisons, per seed, inside each pair's window
    comparisons = []
    for i, rate in enumerate(grid):
        active_pairs = [arm for arm in others
                        if pair_windows[arm][0] <= rate <= pair_windows[arm][1]]
        if not active_pairs:
            continue
        comp = {"rate": float(rate), "pairs": active_pairs}
        for arm in [base] + active_pairs:
            vals = np.array([interp_by_arm[arm][s][i] for s in seeds
                             if s in interp_by_arm[arm] and not np.isnan(interp_by_arm[arm][s][i])])
            if len(vals) == 0:
                comp[f"{arm}_mse_mean"] = None
                continue
            comp[f"{arm}_mse_mean"] = float(np.mean(vals))
            comp[f"{arm}_mse_std"] = float(np.std(vals))
            comp[f"{arm}_n_seeds"] = int(len(vals))
        for arm in active_pairs:
            diffs, ratios = [], []
            for s in seeds:
                if s not in interp_by_arm[base] or s not in interp_by_arm[arm]:
                    continue
                b, c = interp_by_arm[base][s][i], interp_by_arm[arm][s][i]
                if np.isnan(b) or np.isnan(c):
                    continue
                diffs.append(b - c)          # >0 means baseline (frame) worse
                ratios.append(b / c if c > 0 else np.nan)
            if diffs:
                comp[f"{base}_minus_{arm}_mean"] = float(np.mean(diffs))
                comp[f"{base}_minus_{arm}_std"] = float(np.std(diffs))
                comp[f"{base}_over_{arm}_ratio_mean"] = float(np.nanmean(ratios))
                comp[f"{base}_worse_count"] = int(sum(d > 0 for d in diffs))
                comp[f"{base}_better_count"] = int(sum(d < 0 for d in diffs))
                comp["n_seeds_paired"] = len(diffs)
        comparisons.append(comp)

    print("\n=== Matched-rate comparison (per-pair windows only) ===", flush=True)
    for comp in comparisons:
        line = f"{comp['rate']:6.0f} | frame={comp.get('frame_mse_mean', float('nan')):.5f}"
        for arm in comp["pairs"]:
            ratio = comp.get(f"frame_over_{arm}_ratio_mean")
            nb = comp.get("frame_better_count", 0)
            ns = comp.get("n_seeds_paired", 0)
            line += (f"  {arm}={comp.get(f'{arm}_mse_mean', float('nan')):.5f}"
                     f" (frame/{arm}={ratio:.3f}, frame better {nb}/{ns} seeds)")
        print(line, flush=True)

    # AUC over the primary pair's window (per seed, trapezoid on the grid)
    auc_by_arm: dict[str, dict[str, float]] = {}
    primary = others[0] if others else None
    if primary is not None:
        p_lo, p_hi = pair_windows[primary]
        grid_ov = np.array([c["rate"] for c in comparisons if c["pairs"] == [primary] or primary in c["pairs"]])
        grid_ov = grid_ov[(grid_ov >= p_lo) & (grid_ov <= p_hi)]
        if len(grid_ov) >= 2:
            idx_of = {float(r): i for i, r in enumerate(grid)}
            for arm in [base, primary]:
                per_seed = {}
                for s in seeds:
                    if s not in interp_by_arm[arm]:
                        continue
                    vals = np.array([interp_by_arm[arm][s][idx_of[float(r)]] for r in grid_ov])
                    if np.isnan(vals).any() or len(vals) < 2:
                        continue
                    per_seed[str(s)] = float(trapz(vals, grid_ov))
                if per_seed:
                    vals = np.array(list(per_seed.values()))
                    auc_by_arm[arm] = {"auc_mean": float(np.mean(vals)),
                                       "auc_std": float(np.std(vals)),
                                       "per_seed": per_seed}
            if base in auc_by_arm and primary in auc_by_arm:
                print(f"\nAUC over {base}-vs-{primary} window ({p_lo:.0f}-{p_hi:.0f} b/s; lower is better):", flush=True)
                for arm in [base, primary]:
                    print(f"  {arm:9s} {auc_by_arm[arm]['auc_mean']:.4f} "
                          f"+/- {auc_by_arm[arm]['auc_std']:.4f}", flush=True)

    # Conservative matched-config-pair comparison: pair each segment config with
    # the frame config of closest rate (within tolerance), compare config MEANS
    # (no per-seed envelope selection) with per-seed paired win counts.
    print("\n=== Matched config pairs (|rate diff| <= 10%; config means, per-seed paired) ===", flush=True)
    matched_pairs = []
    if "summary" in results:
        summary = results["summary"]
        frame_rows = [r for r in summary.values() if r["arm"] == "frame"]
        for arm in others:
            for r in summary.values():
                if r["arm"] != arm:
                    continue
                if not frame_rows:
                    continue
                f_best = min(frame_rows, key=lambda fr: abs(fr["bits_per_sec"] - r["bits_per_sec"]))
                dr = abs(f_best["bits_per_sec"] - r["bits_per_sec"]) / r["bits_per_sec"]
                if dr > 0.10:
                    continue
                # per-seed paired comparison from raw points
                pts_f = {(p["seed"]): p["mse"] for p in points
                         if p["arm"] == "frame" and p["codebook"] == f_best["codebook"]}
                pts_a = {(p["seed"]): p["mse"] for p in points
                         if p["arm"] == arm and p["K"] == r["K"] and p["codebook"] == r["codebook"]}
                common = sorted(set(pts_f) & set(pts_a))
                if not common:
                    continue
                diffs = [pts_f[s_] - pts_a[s_] for s_ in common]
                wins_a = int(sum(dd > 0 for dd in diffs))
                rec = {
                    "segment_config": f"{arm} K={r['K']} N={r['codebook']}",
                    "frame_config": f"frame N={f_best['codebook']}",
                    "rate_segment": r["bits_per_sec"], "rate_frame": f_best["bits_per_sec"],
                    "rate_diff_frac": dr,
                    "mse_segment_mean": r["mse_mean"], "mse_segment_std": r["mse_std"],
                    "mse_frame_mean": f_best["mse_mean"], "mse_frame_std": f_best["mse_std"],
                    "mean_diff_frame_minus_segment": float(np.mean(diffs)),
                    "segment_wins": wins_a, "n_seeds": len(common),
                }
                matched_pairs.append(rec)
                print(f"  {rec['segment_config']:>22s} ({rec['rate_segment']:5.1f} b/s, "
                      f"{rec['mse_segment_mean']:.4f}+/-{rec['mse_segment_std']:.4f})  vs  "
                      f"{rec['frame_config']:>12s} ({rec['rate_frame']:5.1f} b/s, "
                      f"{rec['mse_frame_mean']:.4f}+/-{rec['mse_frame_std']:.4f})  "
                      f"-> segment better {rec['segment_wins']}/{rec['n_seeds']} seeds", flush=True)

    analysis = {
        "config": {**vars(args), "input": str(args.input), "output": str(args.output)},
        "rate_grid": grid.tolist(),
        "measured_ranges": {a: {"min": range_by_arm[a][0], "max": range_by_arm[a][1]}
                            for a in arms},
        "overlap_window": {"min": ov_lo, "max": ov_hi},
        "pair_overlap_windows": {a: {"min": pair_windows[a][0], "max": pair_windows[a][1]}
                                 for a in others},
        "matched_rate_comparisons": comparisons,
        "matched_config_pairs": matched_pairs,
        "auc_overlap_window": auc_by_arm,
        "note": ("Comparisons are restricted to the measured-rate overlap window; "
                 "grid points outside an arm's measured range are excluded rather "
                 "than extrapolated. Segment-arm envelopes are Pareto lower "
                 "envelopes over (K, N) configurations per seed."),
    }
    args.output.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
