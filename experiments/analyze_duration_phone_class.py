"""Duration coupling vs phone-class confounding (residualization analysis).

The paper reports R^2 ~ 0.72 for adaptive segment vectors predicting segment
duration (Experiment E, pooled-vector ridge probe, utterance-level 5-fold CV).
The open confound: segment duration may be predictable from the dominant phone
class alone (silence -> long, vowels -> long, transitions -> short), so the
vectors may merely re-encode phone identity.

This script quantifies the confound on the SAME records that produced the
published R^2 (segment_records_e_seed7.npz, dumped by experiment_e.py
--dump-records), using the same estimator and folds:

  1. r2_full        ridge(duration | z)                       (reproduces the paper)
  2. r2_norm        ridge(duration | ||z||)                   (geometric-artifact control)
  3. r2_phone       ridge(duration | one-hot(dominant phone)) (confound strength)
  4. r2_additive    duration | phone-mean + ridge(z on residual)
                    - leakage-free additive partial model:
                      per CV fold, class means are estimated on TRAIN folds only
  5. incremental    r2_additive - r2_phone
                    (duration information in z BEYOND phone class)

Run for every condition in the records (adaptive, adaptive_hard, dp, uniform),
so the question "is the coupling specific to the learned segmentation?" is
answered alongside the confound question.

Output: experiments/analysis_duration_phone_class.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Estimators (mirroring experiment_e.ridge_probe_r2 exactly)
# ---------------------------------------------------------------------------

def ridge_probe_r2(vecs: np.ndarray, values: np.ndarray, utt_ids: np.ndarray,
                   fold_ids: np.ndarray, alpha: float = 10.0) -> dict:
    """Within-utterance-centered ridge R^2, utterance-level CV (experiment_e)."""
    X = np.asarray(vecs, float)
    y = np.asarray(values, float)
    ok = ~np.isnan(y)
    X, y, utt_ids, fold_ids = X[ok], y[ok], utt_ids[ok], fold_ids[ok]
    Xc = np.empty_like(X)
    yc = np.empty_like(y)
    for u in np.unique(utt_ids):
        m = utt_ids == u
        if m.sum() > 1:
            Xc[m] = X[m] - X[m].mean(axis=0, keepdims=True)
            yc[m] = y[m] - y[m].mean()
        else:
            Xc[m] = 0.0
            yc[m] = 0.0
    preds = np.full(len(y), np.nan)
    for f in np.unique(fold_ids):
        tr, te = fold_ids != f, fold_ids == f
        if tr.sum() < 10 or te.sum() == 0:
            continue
        sd = Xc[tr].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xtr = Xc[tr] / sd
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
        beta = np.linalg.solve(A, Xtr.T @ yc[tr])
        preds[te] = (Xc[te] / sd) @ beta
    valid = ~np.isnan(preds)
    if valid.sum() < 2 or yc[valid].var() <= 0:
        return {"r2": 0.0, "n_eval": int(valid.sum())}
    ss_res = ((yc[valid] - preds[valid]) ** 2).sum()
    ss_tot = ((yc[valid] - yc[valid].mean()) ** 2).sum()
    return {"r2": float(1.0 - ss_res / ss_tot), "n_eval": int(valid.sum())}


def ridge_r2_norm_control(vecs: np.ndarray, values: np.ndarray, utt_ids: np.ndarray,
                          fold_ids: np.ndarray) -> dict:
    """Artifact control: predict target from ||Xc|| alone (experiment_e)."""
    X = np.asarray(vecs, float)
    y = np.asarray(values, float)
    ok = ~np.isnan(y)
    X, y, utt_ids, fold_ids = X[ok], y[ok], utt_ids[ok], fold_ids[ok]
    norm = np.zeros(len(y))
    yc = np.zeros(len(y))
    for u in np.unique(utt_ids):
        m = utt_ids == u
        if m.sum() > 1:
            Xc = X[m] - X[m].mean(axis=0, keepdims=True)
            norm[m] = np.linalg.norm(Xc, axis=1)
            yc[m] = y[m] - y[m].mean()
    preds = np.full(len(y), np.nan)
    for f in np.unique(fold_ids):
        tr, te = fold_ids != f, fold_ids == f
        if tr.sum() < 10 or te.sum() == 0:
            continue
        sd = norm[tr].std()
        phi_tr = np.stack([norm[tr] / max(sd, 1e-8), np.ones(int(tr.sum()))], axis=1)
        phi_te = np.stack([norm[te] / max(sd, 1e-8), np.ones(int(te.sum()))], axis=1)
        beta, *_ = np.linalg.lstsq(phi_tr, yc[tr], rcond=None)
        preds[te] = phi_te @ beta
    valid = ~np.isnan(preds)
    if valid.sum() < 2 or yc[valid].var() <= 0:
        return {"r2": 0.0}
    ss_res = ((yc[valid] - preds[valid]) ** 2).sum()
    ss_tot = ((yc[valid] - yc[valid].mean()) ** 2).sum()
    return {"r2": float(1.0 - ss_res / ss_tot)}


def ridge_r2_additive_with_phone(vecs: np.ndarray, values: np.ndarray, phones: np.ndarray,
                                 utt_ids: np.ndarray, fold_ids: np.ndarray,
                                 alpha: float = 10.0) -> dict:
    """Leakage-free additive partial model: duration | phone + z.

    Per CV fold:
      - class means E[yc | phone] estimated on TRAIN folds only
      - ridge on z predicts the residual (yc - class mean), train folds only
      - test prediction = train-fold class mean + predicted residual
    R^2 is computed against the within-utterance-centered duration yc.
    """
    X = np.asarray(vecs, float)
    y = np.asarray(values, float)
    ph = np.asarray(phones)
    ok = ~np.isnan(y)
    X, y, ph, utt_ids, fold_ids = X[ok], y[ok], ph[ok], utt_ids[ok], fold_ids[ok]
    Xc = np.empty_like(X)
    yc = np.empty_like(y)
    for u in np.unique(utt_ids):
        m = utt_ids == u
        if m.sum() > 1:
            Xc[m] = X[m] - X[m].mean(axis=0, keepdims=True)
            yc[m] = y[m] - y[m].mean()
        else:
            Xc[m] = 0.0
            yc[m] = 0.0
    preds = np.full(len(y), np.nan)
    for f in np.unique(fold_ids):
        tr, te = fold_ids != f, fold_ids == f
        if tr.sum() < 10 or te.sum() == 0:
            continue
        # Train-fold class means of the centered duration (global mean = 0).
        class_means = {}
        for c in np.unique(ph[tr]):
            m = tr & (ph == c)
            class_means[c] = float(yc[m].mean())
        base_tr = np.array([class_means.get(c, 0.0) for c in ph[tr]])
        base_te = np.array([class_means.get(c, 0.0) for c in ph[te]])
        resid_tr = yc[tr] - base_tr
        sd = Xc[tr].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xtr = Xc[tr] / sd
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
        beta = np.linalg.solve(A, Xtr.T @ resid_tr)
        preds[te] = base_te + (Xc[te] / sd) @ beta
    valid = ~np.isnan(preds)
    if valid.sum() < 2 or yc[valid].var() <= 0:
        return {"r2": 0.0, "n_eval": int(valid.sum())}
    ss_res = ((yc[valid] - preds[valid]) ** 2).sum()
    ss_tot = ((yc[valid] - yc[valid].mean()) ** 2).sum()
    return {"r2": float(1.0 - ss_res / ss_tot), "n_eval": int(valid.sum())}


def one_hot(labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    idx = np.searchsorted(classes, labels)
    out = np.zeros((len(labels), len(classes)), dtype=float)
    out[np.arange(len(labels)), idx] = 1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Duration coupling vs phone-class confounding (residualization)")
    parser.add_argument("--records", type=Path,
                        default=Path("experiments/segment_records_e_seed7.npz"),
                        help="npz dumped by experiment_e.py --dump-records")
    parser.add_argument("--reference-r2", type=float, default=None,
                        help="Published r2_duration for the adaptive condition (sanity check)")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/analysis_duration_phone_class.json"))
    args = parser.parse_args()

    if not args.records.exists():
        raise SystemExit(
            f"records file not found: {args.records}\n"
            "Generate it with: experiment_e.py --seed 7 --output <scratch.json> "
            "--dump-records experiments/segment_records_e_seed7.npz")

    dump = np.load(args.records)
    conds = sorted({k.split("__")[0] for k in dump.files})
    print(f"Loaded records for conditions: {conds}", flush=True)

    # Rebuild experiment_e's fold assignment exactly:
    #   utt_folds = np.random.default_rng(1234).integers(0, 5, size=n_utts)
    n_utts = int(max(dump[f"{c}__utt"].max() for c in conds)) + 1
    utt_folds = np.random.default_rng(1234).integers(0, 5, size=n_utts)

    results = {"config": vars(args), "n_utts": n_utts, "conditions": {}}
    header = (f"{'condition':>14} {'r2_full':>8} {'r2_norm':>8} {'r2_phone':>9} "
              f"{'r2_additive':>12} {'incremental':>12} {'n':>6}")
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)

    for cond in conds:
        vecs = dump[f"{cond}__vecs"].astype(float)
        dur = dump[f"{cond}__dur"].astype(float)
        phone = dump[f"{cond}__phone"]
        utt = dump[f"{cond}__utt"].astype(int)
        folds = utt_folds[utt]

        r2_full = ridge_probe_r2(vecs, dur, utt, folds)
        r2_norm = ridge_r2_norm_control(vecs, dur, utt, folds)
        onehot = one_hot(phone)
        r2_phone = ridge_probe_r2(onehot, dur, utt, folds)
        r2_add = ridge_r2_additive_with_phone(vecs, dur, phone, utt, folds)
        incremental = r2_add["r2"] - r2_phone["r2"]

        row = {
            "r2_full": round(r2_full["r2"], 4),
            "r2_norm_control": round(r2_norm["r2"], 4),
            "r2_phone_only": round(r2_phone["r2"], 4),
            "r2_additive_phone_plus_z": round(r2_add["r2"], 4),
            "incremental_r2_beyond_phone": round(incremental, 4),
            "n_segments_eval": int(r2_full["n_eval"]),
        }
        results["conditions"][cond] = row
        print(f"{cond:>14} {row['r2_full']:>8.4f} {row['r2_norm_control']:>8.4f} "
              f"{row['r2_phone_only']:>9.4f} {row['r2_additive_phone_plus_z']:>12.4f} "
              f"{row['incremental_r2_beyond_phone']:>12.4f} {row['n_segments_eval']:>6d}",
              flush=True)

    # Interpretation guidance printed for the adaptive condition.
    a = results["conditions"].get("adaptive", {})
    if a:
        print("\nInterpretation (adaptive):", flush=True)
        print(f"  r2_full should reproduce the published r2_duration"
              + (f" ({args.reference_r2})" if args.reference_r2 else "") + ".", flush=True)
        print(f"  r2_phone_only = {a['r2_phone_only']:.4f}: how much of duration the "
              f"dominant phone class alone explains (the confound).", flush=True)
        print(f"  incremental_r2_beyond_phone = {a['incremental_r2_beyond_phone']:.4f}: "
              f"duration information in z BEYOND phone class. If ~0, the coupling is "
              f"phone-class confounded; if clearly > 0, the vectors carry genuine "
              f"duration-relevant content.", flush=True)

    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
