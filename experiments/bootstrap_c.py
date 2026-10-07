"""Paired bootstrap for Experiment C: is learned-K32 F1 above controls?

Resamples utterances (paired design: every condition is scored on the same 48
test utterances from the same run, seed 7, tol 2), 10,000 bootstrap replicates,
percentile CIs and two-sided p-values for the mean F1 difference.

Run from the research root with the fastspeech2.0 venv python.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results_experiment_c.json"
RNG = np.random.default_rng(0)
N_BOOT = 10_000

data = json.loads(RES.read_text())
per = data["per_utterance"]
conds = sorted(per.keys())
f1 = {c: np.array([r["f1"] for r in per[c]], dtype=np.float64) for c in conds}
n = len(f1[conds[0]])
assert all(len(v) == n for v in f1.values()), "condition row counts differ"
print(f"n_utterances={n}, conditions={conds}\n")

means = {c: f1[c].mean() for c in conds}


def paired_test(a: str, b: str) -> None:
    d = f1[a] - f1[b]
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    p = min(1.0, p + 1.0 / N_BOOT)
    print(f"{a:28s} vs {b:12s} dF1={d.mean():+.4f} "
          f"95%CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} wins={int((d > 0).sum())}/{n}")


print("=== paired bootstrap (10k resamples) ===")
paired_test("learned_K32", "uniform")
paired_test("learned_K32", "random")
paired_test("learned_K32", "dp_K32")
paired_test("learned_K16", "random")
paired_test("learned_K16", "uniform")

print("\n=== mean F1 per condition (this run) ===")
for c in conds:
    print(f"{c:28s} {means[c]:.4f}")

print("\n=== seed replicates of learned_K32 (summary F1) ===")
seeds = {}
for s in ("", "_s3", "_s11"):
    p = Path(__file__).resolve().parent / f"results_experiment_c{s}.json"
    if p.exists():
        r = json.loads(p.read_text())
        tag = s.lstrip("_") or "7"
        seeds[tag] = r["summary"]["learned_K32"]["f1"]
        print(f"  seed {tag}: {seeds[tag]:.4f}")
if len(seeds) >= 2:
    v = np.array(list(seeds.values()))
    print(f"  mean={v.mean():.4f} sd={v.std(ddof=1):.4f}")
