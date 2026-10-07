"""Equivalence testing for learned vs DP boundaries using paired TOST.

The paper must not claim "statistically tied" from a non-significant difference
test: failing to reject H0 does not establish equivalence. This script runs
Two One-Sided Tests (TOST) on PER-UTTERANCE boundary-F1 values from
Experiment C, using the paired design (the same 48 utterances are scored for
both partitions).

Tests, for margin delta:
  H0: |mu_learned - mu_ref| >= delta      H1: |mu_learned - mu_ref| < delta
Equivalence is claimed only if BOTH one-sided tests reject at level alpha.

Implemented as:
  1. Paired bootstrap TOST: percentile CI of the mean paired difference over
     utterance resamples; equivalent if the CI lies inside (-delta, +delta).
  2. Parametric paired TOST: two one-sided one-sample t-tests on the paired
     differences (df = n - 1).

Comparisons:
  - learned K=32 vs DP K=32   (primary; the paper's "tie" claim)
  - learned K=32 vs uniform   (control; expected NON-equivalent)
  - learned K=16 vs DP K=16   (secondary)
plus a sensitivity analysis across margins delta.

Output: experiments/analysis_equivalence_tost.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def t_sf(t: float, df: int) -> float:
    """Survival function of Student t via the regularized incomplete beta."""
    try:
        from scipy import stats
        return float(stats.t.sf(t, df))
    except ImportError:
        # Continued-fraction incomplete beta (Lentz), no scipy needed.
        def betacf(a: float, b: float, x: float) -> float:
            tiny, eps = 1e-300, 3e-12
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c, d = 1.0, 1.0 - qab * x / qap
            if abs(d) < tiny:
                d = tiny
            d = 1.0 / d
            h = d
            for m in range(1, 300):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                if abs(d) < tiny:
                    d = tiny
                c = 1.0 + aa / c
                if abs(c) < tiny:
                    c = tiny
                d = 1.0 / d
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                if abs(d) < tiny:
                    d = tiny
                c = 1.0 + aa / c
                if abs(c) < tiny:
                    c = tiny
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return h

        def betainc(a: float, b: float, x: float) -> float:
            if x <= 0:
                return 0.0
            if x >= 1:
                return 1.0
            lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
            front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
            if x < (a + 1) / (a + b + 2):
                return front * betacf(a, b, x)
            return 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) / b * betacf(b, a, 1 - x)

        return betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_parametric_tost(diff: np.ndarray, delta: float, alpha: float = 0.05) -> dict:
    """Two one-sided one-sample t-tests on paired differences."""
    d = np.asarray(diff, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    # T1: H0: mu <= -delta  vs  H1: mu > -delta
    t1 = (mean + delta) / se
    p1 = t_sf(t1, df)  # P(T >= t1)
    # T2: H0: mu >= +delta  vs  H1: mu < +delta
    t2 = (mean - delta) / se
    p2 = t_sf(-t2, df)  # P(T <= t2) = 1 - sf(t2)
    p_tost = max(p1, p2)
    t_crit = _t_ppf(1 - alpha / 2, df)
    return {
        "n": int(n),
        "mean_diff": mean,
        "sd_diff": sd,
        "se": se,
        "df": df,
        "t_lower": float(t1),
        "p_lower": float(p1),
        "t_upper": float(t2),
        "p_upper": float(p2),
        "p_tost": float(p_tost),
        "alpha": alpha,
        "ci95_lower": mean - t_crit * se,
        "ci95_upper": mean + t_crit * se,
        "is_equivalent": bool(p_tost < alpha),
    }


def _t_ppf(p: float, df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(p, df))
    except ImportError:
        # Bisection on the survival function (accurate to 1e-6).
        lo, hi = -50.0, 50.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if t_sf(mid, df) > p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def paired_bootstrap_tost(diff: np.ndarray, delta: float, n_boot: int = 10000,
                          seed: int = 42, alpha: float = 0.05) -> dict:
    """Bootstrap percentile CI of the mean paired difference (utterance resamples)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    # Two one-sided bootstrap p-values (fraction of resamples beyond each margin).
    p_lower = float((means <= -delta).mean())   # H0: mu <= -delta
    p_upper = float((means >= delta).mean())    # H0: mu >= +delta
    p_tost = max(p_lower, p_upper)
    return {
        "n": int(n),
        "mean_diff": float(d.mean()),
        "n_boot": int(n_boot),
        "ci_lower": lo,
        "ci_upper": hi,
        "p_tost": float(p_tost),
        "alpha": alpha,
        "is_equivalent": bool(p_tost < alpha and lo > -delta and hi < delta),
    }


def load_c_results(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    pu = data["per_utterance"]
    out = {}
    for arm, rows in pu.items():
        out[arm] = np.array([r["f1"] for r in rows], dtype=float)
    return out


def paired_difference_test(diff: np.ndarray) -> dict:
    """Paired t-test on differences (for context only - NOT an equivalence test)."""
    d = np.asarray(diff, float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se
    p = 2 * t_sf(abs(t), n - 1)
    return {"t": t, "p_two_sided": float(min(p, 1.0)), "mean_diff": mean}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired TOST equivalence testing for learned vs DP boundary F1")
    parser.add_argument("--results", type=Path,
                        default=Path("experiments/results_experiment_c.json"))
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Equivalence margin in F1 units (default 0.01)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/analysis_equivalence_tost.json"))
    args = parser.parse_args()

    f1 = load_c_results(args.results)
    print(f"Loaded per-utterance F1 for arms: {sorted(f1.keys())}", flush=True)

    comparisons = [
        ("learned_K32", "dp_K32", "primary: paper's learned-vs-DP tie claim"),
        ("learned_K32", "uniform", "control: expected NOT equivalent"),
        ("learned_K16", "dp_K16", "secondary: K=16 comparison"),
    ]

    report = {"config": vars(args), "comparisons": {}, "sensitivity": {}}
    for arm_a, arm_b, note in comparisons:
        if arm_a not in f1 or arm_b not in f1:
            print(f"  skipping {arm_a} vs {arm_b}: missing arm", flush=True)
            continue
        a, b = f1[arm_a], f1[arm_b]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        diff = a - b
        print(f"\n=== {arm_a} vs {arm_b} ({note}) ===", flush=True)
        print(f"  n={n}  mean F1: {arm_a}={a.mean():.4f}  {arm_b}={b.mean():.4f}  "
              f"paired diff={diff.mean():+.4f} +/- {diff.std(ddof=1) / math.sqrt(n):.4f} SE",
              flush=True)
        dt = paired_difference_test(diff)
        print(f"  paired difference test (context only): t={dt['t']:.3f}, "
              f"p={dt['p_two_sided']:.4f}", flush=True)
        para = paired_parametric_tost(diff, args.delta, args.alpha)
        print(f"  parametric TOST (delta={args.delta}): p={para['p_tost']:.4f} -> "
              f"{'EQUIVALENT' if para['is_equivalent'] else 'NOT equivalent'}",
              flush=True)
        boot = paired_bootstrap_tost(diff, args.delta, args.n_bootstrap, args.seed, args.alpha)
        print(f"  bootstrap TOST (delta={args.delta}): CI=[{boot['ci_lower']:+.4f}, "
              f"{boot['ci_upper']:+.4f}] p={boot['p_tost']:.4f} -> "
              f"{'EQUIVALENT' if boot['is_equivalent'] else 'NOT equivalent'}", flush=True)
        report["comparisons"][f"{arm_a}_vs_{arm_b}"] = {
            "note": note,
            "n": int(n),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "difference_test": dt,
            "parametric_tost": para,
            "bootstrap_tost": boot,
        }

        # Sensitivity across margins for the primary comparison.
        sens = []
        for delta in (0.005, 0.01, 0.02, 0.03, 0.05):
            para_d = paired_parametric_tost(diff, delta, args.alpha)
            boot_d = paired_bootstrap_tost(diff, delta, args.n_bootstrap, args.seed, args.alpha)
            sens.append({
                "delta": delta,
                "parametric_equivalent": para_d["is_equivalent"],
                "parametric_p_tost": para_d["p_tost"],
                "bootstrap_equivalent": boot_d["is_equivalent"],
                "bootstrap_ci": [boot_d["ci_lower"], boot_d["ci_upper"]],
            })
            print(f"    delta={delta:.3f}: parametric={'EQUIV' if para_d['is_equivalent'] else 'not equiv'} "
                  f"(p={para_d['p_tost']:.4f}), bootstrap="
                  f"{'EQUIV' if boot_d['is_equivalent'] else 'not equiv'} "
                  f"CI=[{boot_d['ci_lower']:+.4f},{boot_d['ci_upper']:+.4f}]", flush=True)
        report["sensitivity"][f"{arm_a}_vs_{arm_b}"] = sens

    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved TOST analysis to {args.output}", flush=True)
    print("Reminder: an equivalence claim requires a pre-registered margin delta; "
          "report the margin alongside the result.", flush=True)


if __name__ == "__main__":
    main()
