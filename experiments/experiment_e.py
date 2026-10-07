"""
Experiment E — What information does the adaptive mask decomposition carry?

Per the post-D direction (professor review): stop asking "where are the
boundaries?" (D answered: acoustic change points, indistinguishable from
reconstruction-optimal DP) and ask instead whether adaptive segmentation
ORGANIZES INFORMATION DIFFERENTLY than reference partitions, even though it
does not compress better (B/B2).

Design (fixed before running):

  Partitions at K=32, per utterance:
    - adaptive  : learned soft masks m_k(t) (the actual mechanism output)
    - adaptive_hard : hard segmentation at the learned boundaries (isolates
      the soft-pooling effect from the boundary-placement effect)
    - dp        : reconstruction-optimal hard segmentation (cached)
    - uniform   : equal-length hard segmentation

  Information channels measured per utterance, then paired-bootstrap over
  utterances (10k resamples):

    - H(segment)  : entropy of the segment variable. For a complete hard
      partition this is exactly log2(K) minus a small correction, so it is
      nearly identical by construction; it is reported as a sanity check,
      NOT as a discriminating metric.
    - I(segment; phoneme)      : plug-in MI, cluster-count shrinkage.
    - I(segment; pitch)        : weighted plug-in MI over log-F0 bins.
    - I(segment; energy)       : weighted plug-in MI over per-utt energy
      quantile bins (natural discrete variable).
    - eta2(segment; duration)  : one-way ANOVA effect size of segment id on
      segment duration.
    - Pooled-vector probes (the z_k test): nearest-centroid phoneme accuracy
      and ridge R^2 for energy-bin and pitch-bin prediction from the pooled
      segment vector, 5-fold per-utterance CV.

  Speaker channel: structurally impossible here — the corpus is single-speaker
  (features_wavlm_l6/index.json carries no speaker metadata; the dataset is
  one Nepali speaker). Dropped by design, not silently.

Estimator caveats (also in docs/experiment_e.md): plug-in MI is biased
upward; all hard conditions have exactly K=32 clusters so the bias is
comparable across conditions. Soft-mask "H" is NOT comparable to hard
partitions (it is not a partition) and is excluded from MI claims; soft-mask
geometric statistics are reported separately.

Outputs: experiments/results_experiment_e.json

Run from the research root with the fastspeech2.0 venv python:
  C:/Users/Lenovo/Desktop/fastspeech2.0/venv/Scripts/python.exe experiments/experiment_e.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    uniform_bounds,
    warp_boundaries,
)
from experiment_c import STRUCTURAL_IDS  # same structural token set as C/D

# Reuse D's acoustic extraction (module import has no side effects: main is guarded).
from experiment_d import (
    MEL_TO_SSL,
    SSL_RATE,
    BoundaryAnalyzer,
    load_wav,
    rms_energy,
)

RESEARCH = Path(__file__).resolve().parent.parent
ALIGN_DIR = RESEARCH / "experiments" / "alignments_c"

F0_MIN, F0_MAX = 65.0, 400.0
N_F0_BINS = 24
N_ENERGY_BINS = 8
SHRINK_K = 5.0  # prior strength for cluster-count entropy shrinkage


# ---------------------------------------------------------------------------
# Alignment-derived frame labels
# ---------------------------------------------------------------------------

def phone_labels_from_align(align: dict, T: int) -> np.ndarray:
    """Per-SSL-frame phone token id from MAS durations (mel rate -> SSL)."""
    durations = np.asarray(align["durations"], dtype=float)
    tokens = np.asarray(align["tokens"], dtype=int)
    edges_mel = np.concatenate([[0.0], np.cumsum(durations)])
    edges_ssl = edges_mel * MEL_TO_SSL
    edges_ssl = np.clip(edges_ssl, 0, T)
    t = np.arange(T)
    return tokens[np.searchsorted(edges_ssl, t, side="right") - 1].clip(min=0)


def energy_labels(feats_ssl_T: int, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-SSL-frame energy quantile-bin label (1) and log-RMS value."""
    mel_energy = rms_energy(wav)
    mel_rate = 22050.0 / 256.0
    mel_times = (np.arange(len(mel_energy)) + 0.5) / mel_rate
    ssl_times = (np.arange(feats_ssl_T) + 0.5) / SSL_RATE
    e_ssl = np.interp(ssl_times, mel_times, mel_energy)
    log_e = np.log(np.clip(e_ssl, 1e-9, None))
    # Per-utterance standardized log-energy -> quantile bins.
    z = (log_e - log_e.mean()) / max(log_e.std(), 1e-9)
    edges = np.quantile(z, np.linspace(0, 1, N_ENERGY_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, z, side="right")
    return bins.astype(int), log_e


def f0_labels(wav: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-SSL-frame log-F0 bin label and voicing weight.

    Returns (bins, weights); bins are meaningless where weight ~ 0.
    """
    import librosa

    f0, _, vprob = librosa.pyin(
        wav.astype(np.float64), fmin=F0_MIN, fmax=F0_MAX,
        frame_length=1024, sr=16000,
    )
    mel_rate = 22050.0 / 256.0
    mel_times = (np.arange(len(f0)) + 0.5) / mel_rate
    ssl_times = (np.arange(T) + 0.5) / SSL_RATE
    idx = np.clip(np.round(np.interp(ssl_times, mel_times, np.arange(len(f0)))).astype(int),
                  0, len(f0) - 1)
    f0_ssl = f0[idx]
    w_ssl = np.nan_to_num(vprob[idx], nan=0.0)
    log_f0 = np.log2(np.clip(f0_ssl, F0_MIN, F0_MAX))
    edges = np.linspace(math.log2(F0_MIN), math.log2(F0_MAX), N_F0_BINS + 1)[1:-1]
    bins = np.searchsorted(edges, log_f0, side="right")
    return bins.astype(int), w_ssl.astype(np.float32), log_f0.astype(np.float32)


def segment_assignments(bounds: list[int], T: int) -> np.ndarray:
    """Per-frame segment id from hard boundaries."""
    a = np.zeros(T, dtype=int)
    starts = [0] + list(bounds)
    ends = list(bounds) + [T]
    for k, (s, e) in enumerate(zip(starts, ends)):
        a[s:e] = k
    return a


# ---------------------------------------------------------------------------
# Entropy / MI estimators (plug-in with cluster-count shrinkage)
# ---------------------------------------------------------------------------

def _kish_neff(w: np.ndarray) -> float:
    s1 = w.sum()
    s2 = (w * w).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def _H_shrunk(weights: np.ndarray, codes: np.ndarray) -> float:
    """Shrunk plug-in Shannon entropy of a discrete code with per-row weights."""
    w = np.asarray(weights, float)
    keep = w > 0
    w, codes = w[keep], np.asarray(codes)[keep]
    if len(w) == 0:
        return 0.0
    uniq, counts_w = np.unique(codes, return_counts=True)
    # Weighted mass per code.
    mass = np.zeros(len(uniq))
    np.add.at(mass, np.searchsorted(uniq, codes), w)
    p = mass / mass.sum()
    n_outcomes = len(uniq)
    h = float(-(p * np.log2(p)).sum())
    h_unif = math.log2(n_outcomes) if n_outcomes > 1 else 0.0
    neff = _kish_neff(w)
    return float((neff * h + SHRINK_K * h_unif) / (neff + SHRINK_K))


def weighted_mi(cluster: np.ndarray, attr: np.ndarray, w: np.ndarray) -> float:
    """I(cluster; attr) via shrunk plug-in entropies."""
    keep = w > 0
    c, a, w = np.asarray(cluster)[keep], np.asarray(attr)[keep], np.asarray(w, float)[keep]
    if len(w) == 0:
        return 0.0
    h_c = _H_shrunk(w, c)
    h_a = _H_shrunk(w, a)
    h_j = _H_shrunk(w, c.astype(np.int64) * 100000 + a)
    return float(max(h_c + h_a - h_j, 0.0))


def eta2_duration(cluster: np.ndarray, durations: np.ndarray) -> float:
    """One-way ANOVA eta^2: variance of segment duration explained by cluster id."""
    y = np.asarray(durations, float)
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    uniq = np.unique(cluster)
    ss_bet = 0.0
    for c in uniq:
        m = y[cluster == c]
        if len(m):
            ss_bet += len(m) * (m.mean() - grand) ** 2
    return float(ss_bet / ss_tot)


# ---------------------------------------------------------------------------
# Pooled-vector probes
# ---------------------------------------------------------------------------

def pooled_vectors_hard(feats: np.ndarray, cluster: np.ndarray, K: int) -> np.ndarray:
    """Mean feature vector per hard segment (zero rows for empty segments)."""
    out = np.zeros((K, feats.shape[1]), dtype=np.float32)
    for k in range(K):
        m = cluster == k
        if m.any():
            out[k] = feats[m].mean(axis=0)
    return out


def knn1_centroid_probe(vecs: np.ndarray, labels: np.ndarray,
                        test_fold_of: np.ndarray, fold_ids: np.ndarray) -> dict:
    """Nearest-centroid phone probe over ALL pooled segment vectors.

    Centroids per phone class are built from training folds' segments;
    accuracy is evaluated on test-fold segments. Returns overall accuracy,
    the majority-class baseline on the same rows, and per-utterance mean
    accuracy for paired bootstrap (fold assignment is shared across
    conditions, so per-utterance differences are paired).
    """
    labels = np.asarray(labels)
    uniq_labels = np.unique(labels[labels >= 0])
    label_to_int = {l: i for i, l in enumerate(uniq_labels)}
    keep = np.isin(labels, uniq_labels)
    y = np.array([label_to_int[l] for l in labels[keep]])
    X = vecs[keep]
    utts = test_fold_of[keep]
    folds = fold_ids[keep]

    preds = np.full(len(y), -1)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        centroids = np.zeros((len(uniq_labels), X.shape[1]), dtype=np.float64)
        for li in range(len(uniq_labels)):
            m = tr & (y == li)
            if m.any():
                centroids[li] = X[m].mean(axis=0)
        d = ((X[te][:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        preds[te] = d.argmin(axis=1)
    ok = preds >= 0
    acc = float((preds[ok] == y[ok]).mean()) if ok.any() else 0.0
    vals, cnts = np.unique(y[ok], return_counts=True)
    majority = float(cnts.max() / cnts.sum()) if len(cnts) else 0.0
    # Per-utterance accuracy on evaluated rows (for paired bootstrap).
    per_utt_acc = {}
    for u in np.unique(utts[ok]):
        m = ok & (utts == u)
        per_utt_acc[int(u)] = float((preds[m] == y[m]).mean())
    return {"acc": acc, "majority_baseline": majority,
            "n_eval": int(ok.sum()), "per_utt_acc": per_utt_acc}


def ridge_probe_r2(vecs: np.ndarray, values: np.ndarray, utt_ids: np.ndarray,
                   fold_ids: np.ndarray, alpha: float = 10.0) -> dict:
    """Within-utterance-centered ridge regression R^2 over all pooled segment
    vectors, utterance-level CV (no leakage: a segment's utterance is either
    fully in train or fully in test for a given fold)."""
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
        return {"r2": 0.0, "n_eval": int(valid.sum()), "per_utt_r2": {}}
    ss_res = ((yc[valid] - preds[valid]) ** 2).sum()
    ss_tot = ((yc[valid] - yc[valid].mean()) ** 2).sum()
    # Per-utterance held-out R^2 contribution (for paired bootstrap).
    per_utt_r2 = {}
    for u in np.unique(utt_ids[valid]):
        m = valid & (utt_ids == u)
        if m.sum() >= 2 and yc[m].var() > 0:
            r_u = 1.0 - ((yc[m] - preds[m]) ** 2).sum() / ((yc[m] - yc[m].mean()) ** 2).sum()
            per_utt_r2[int(u)] = float(r_u)
    return {"r2": float(1.0 - ss_res / ss_tot), "n_eval": int(valid.sum()),
            "per_utt_r2": per_utt_r2}


# ---------------------------------------------------------------------------
# Soft-mask geometry (adaptive only)
# ---------------------------------------------------------------------------

def ridge_r2_norm_control(vecs: np.ndarray, values: np.ndarray, utt_ids: np.ndarray,
                          fold_ids: np.ndarray) -> dict:
    """Artifact control: predict the target from ||Xc|| alone (1 feature).

    Under within-utterance centering, long segments have pooled vectors pulled
    toward the utterance mean (small ||Xc||). If a 1-feature model on ||Xc||
    matches the full 64-dim R^2, the 'duration information' is a
    regression-to-the-mean geometric artifact, not segment content.
    """
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
        # Simple linear fit on [norm, 1] (norm standardized per fold).
        sd = norm[tr].std()
        phi_tr = np.stack([norm[tr] / max(sd, 1e-8), np.ones(tr.sum())], axis=1)
        phi_te = np.stack([norm[te] / max(sd, 1e-8), np.ones(te.sum())], axis=1)
        beta, *_ = np.linalg.lstsq(phi_tr, yc[tr], rcond=None)
        preds[te] = phi_te @ beta
    valid = ~np.isnan(preds)
    if valid.sum() < 2 or yc[valid].var() <= 0:
        return {"r2": 0.0}
    ss_res = ((yc[valid] - preds[valid]) ** 2).sum()
    ss_tot = ((yc[valid] - yc[valid].mean()) ** 2).sum()
    return {"r2": float(1.0 - ss_res / ss_tot)}


def soft_mask_stats(masks_np: np.ndarray, cluster: np.ndarray, K: int) -> dict:
    """Geometric statistics of the soft masks. Not comparable to hard stats."""
    conc, own_mass, peak = [], [], []
    for k in range(K):
        m = masks_np[k]
        s = m.sum()
        if s <= 1e-8:
            continue
        conc.append(float((m * m).sum() / s))
        own_mass.append(float(m[cluster == k].sum() / s))
        peak.append(float(m.max()))
    return {
        "mass_concentration_mean": round(float(np.mean(conc)), 4),
        "mass_in_own_hard_segment_frac": round(float(np.mean(own_mass)), 4),
        "peak_value_mean": round(float(np.mean(peak)), 4),
    }


# ---------------------------------------------------------------------------
# Paired bootstrap over utterances
# ---------------------------------------------------------------------------

def paired_bootstrap(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return {"delta_mean": None, "ci": None, "p": None, "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "delta_mean": round(float(d.mean()), 6),
        "ci": [round(float(lo), 6), round(float(hi), 6)],
        "p": round(float(min(p, 1.0)), 4),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment E: information in adaptive masks")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--output", type=Path,
                        default=RESEARCH / "experiments" / "results_experiment_e.json")
    parser.add_argument("--dump-records", type=Path, default=None,
                        help="Optionally save per-segment probe records (vecs/dur/phone/"
                             "utt per condition) to an .npz for offline analyses")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    K = args.k

    # Data: PCA on BOTH splits (train_arm reads torch tensors off train items),
    # DP bounds attached to BOTH splits (boundary loss reads them).
    train_items = load_items("train", device)
    pca = fit_pca(train_items)
    test_items = load_items("test", device)
    apply_pca(train_items, pca, args.pca_dim, device)
    apply_pca(test_items, pca, args.pca_dim, device)
    eval_items = test_items[: args.eval_size]
    print(f"eval utterances: {len(eval_items)}", flush=True)

    dp_store = json.loads((FEATURE_DIR / f"dp_bounds_pca{args.pca_dim}.json").read_text())
    for it in train_items + eval_items:
        it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}

    # Train the K=32 adaptive segmenter with the identical B/C/D recipe.
    print(f"training K={K} adaptive segmenter (seed {args.seed})...", flush=True)
    batches = make_batches(train_items, 8)
    eval_batches = make_batches(eval_items, 8)
    model, out = train_arm(train_items, batches, eval_items, eval_batches, K,
                           argparse.Namespace(
                               latent_dim=32, warmup_steps=300, anneal_steps=1700,
                               batch_size=8, lr=2e-3, tau_start=0.35, tau_end=0.08,
                               boundary_weight=1.0, entropy_weight=1e-3,
                               eval_every=1000, pca_dim=args.pca_dim, seed=args.seed,
                               vq_weight=0.0,
                           ), pca)
    print(f"  K={K} final soft MSE {out['final']['learned_mse']:.4f}", flush=True)

    # Waveforms and acoustic tracks.
    wav_cache, energy_cache, f0_cache = {}, {}, {}
    for it in eval_items:
        uid = it["id"]
        wav = load_wav(uid)
        wav_cache[uid] = wav
        e_bins, log_e = energy_labels(it["T"], wav)
        energy_cache[uid] = (e_bins, log_e)
        f0_cache[uid] = f0_labels(wav, it["T"])

    # Per-utterance measurement.
    hard_conditions = ("adaptive", "dp", "uniform")
    per_utt = {c: [] for c in hard_conditions}
    soft_rows = []

    # Pooled-vector probe storage: per condition, collect segment vectors with
    # labels/targets; probes run once per condition at the end with
    # utterance-level CV folds shared across conditions.
    probe_store = {c: {"vecs": [], "phone": [], "energy": [], "pitch": [],
                       "utt": [], "dur": []}
                   for c in ("adaptive", "adaptive_hard", "dp", "uniform")}

    with torch.no_grad():
        for idx, it in enumerate(eval_items):
            uid = it["id"]
            T = it["T"]
            feats_np = it["features"].detach().cpu().numpy()

            x = it["features"].unsqueeze(0)
            mask = torch.ones(1, T, device=device)
            _, masks, edge_c, _, _ = model.forward_full(x, mask)
            masks_np = masks[0].detach().cpu().numpy()  # [K, T]
            learned_bounds = warp_boundaries(edge_c[0], T, K)
            bounds_by_cond = {
                "adaptive": learned_bounds,
                "dp": list(it["dp_bounds"][K]),
                "uniform": uniform_bounds(T, K),
            }
            clusters = {c: segment_assignments(b, T) for c, b in bounds_by_cond.items()}

            align = json.loads((ALIGN_DIR / f"{uid}.json").read_text())
            phones = phone_labels_from_align(align, T)
            e_bins, log_e = energy_cache[uid]
            f0_bins, f0_w, log_f0 = f0_cache[uid]

            for cond in hard_conditions:
                cl = clusters[cond]
                w = np.ones(T, dtype=float)
                content_masked = np.where(np.isin(phones, list(STRUCTURAL_IDS)), -1, phones)
                row = {
                    "id": uid,
                    "h_segment": _H_shrunk(w, cl),
                    "mi_phone": weighted_mi(cl, phones, w),
                    "mi_phone_content": weighted_mi(cl, content_masked, w),
                    "mi_energy": weighted_mi(cl, e_bins, w),
                    "mi_pitch": weighted_mi(cl, f0_bins, f0_w),
                }
                # eta^2 over segment durations: per-frame label = duration of
                # the segment containing the frame (this condition's bounds).
                bounds = bounds_by_cond[cond]
                seg_dur = np.zeros(T, dtype=float)
                for k, (s, e) in enumerate(zip([0] + list(bounds), list(bounds) + [T])):
                    seg_dur[s:e] = e - s
                row["eta2_duration"] = eta2_duration(cl, seg_dur)
                per_utt[cond].append(row)

                # Soft-mask geometry (adaptive only).
                if cond == "adaptive":
                    soft_rows.append({"id": uid,
                                      **soft_mask_stats(masks_np, cl, K)})

                # Pooled segment vectors for the z_k probes.
                if cond == "adaptive":
                    mass = masks_np.sum(axis=1, keepdims=True).clip(min=1e-8)
                    pooled = (masks_np @ feats_np) / mass  # [K, D] exact z_k
                else:
                    pooled = pooled_vectors_hard(feats_np, cl, K)

                # Dominant phoneme label per segment (-1 for empty).
                seg_labels = []
                for k in range(K):
                    m = cl == k
                    if m.any():
                        vals, cnts = np.unique(phones[m], return_counts=True)
                        seg_labels.append(int(vals[cnts.argmax()]))
                    else:
                        seg_labels.append(-1)

                # Per-segment ridge targets: frame-level tracks reduced to
                # per-segment means, NaN where a segment has no voiced frames.
                energy_seg = np.zeros(K, dtype=float)
                pitch_seg = np.full(K, np.nan, dtype=float)
                for k in range(K):
                    m = cl == k
                    if m.any():
                        energy_seg[k] = float(log_e[m].mean())
                        w_k = f0_w[m]
                        if w_k.sum() > 0:
                            pitch_seg[k] = float((log_f0[m] * w_k).sum() / w_k.sum())
                # Per-segment duration (ridge target).
                seg_dur_k = np.array([e - s for s, e in
                                      zip([0] + list(bounds), list(bounds) + [T])],
                                     dtype=float)
                store = probe_store[cond]
                store["vecs"].append(pooled)
                store["phone"].append(np.array(seg_labels))
                store["energy"].append(energy_seg)
                store["pitch"].append(pitch_seg)
                store["utt"].append(np.full(K, idx))
                store["dur"].append(seg_dur_k)

            # adaptive_hard probes: hard means at the LEARNED boundaries
            # (isolates soft pooling from boundary placement).
            cl_a = clusters["adaptive"]
            pooled_hard = pooled_vectors_hard(feats_np, cl_a, K)
            seg_labels_a = []
            for k in range(K):
                m = cl_a == k
                if m.any():
                    vals, cnts = np.unique(phones[m], return_counts=True)
                    seg_labels_a.append(int(vals[cnts.argmax()]))
                else:
                    seg_labels_a.append(-1)
            seg_dur_a = np.array([e - s for s, e in
                                  zip([0] + list(learned_bounds), list(learned_bounds) + [T])],
                                 dtype=float)
            energy_seg_a = np.zeros(K, dtype=float)
            pitch_seg_a = np.full(K, np.nan, dtype=float)
            for k in range(K):
                m = cl_a == k
                if m.any():
                    energy_seg_a[k] = float(log_e[m].mean())
                    w_k = f0_w[m]
                    if w_k.sum() > 0:
                        pitch_seg_a[k] = float((log_f0[m] * w_k).sum() / w_k.sum())
            store = probe_store["adaptive_hard"]
            store["vecs"].append(pooled_hard)
            store["phone"].append(np.array(seg_labels_a))
            store["energy"].append(energy_seg_a)
            store["pitch"].append(pitch_seg_a)
            store["utt"].append(np.full(K, idx))
            store["dur"].append(seg_dur_a)

            if idx % 10 == 0:
                print(f"  analyzed {idx + 1}/{len(eval_items)}: {uid} T={T}", flush=True)

    # Aggregates.
    METRIC_KEYS = ["h_segment", "mi_phone", "mi_phone_content", "mi_energy",
                   "mi_pitch", "eta2_duration"]
    aggregates = {}
    for cond in hard_conditions:
        aggregates[cond] = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in per_utt[cond] if r[key] is not None]
            aggregates[cond][key] = round(float(np.mean(vals)), 4)
    aggregates["adaptive_soft_geometry"] = {
        key: round(float(np.mean([r[key] for r in soft_rows])), 4)
        for key in soft_rows[0] if key != "id"
    }

    # ------------------------------------------------------------------
    # Pooled-vector probes: run once per condition, utterance-level CV with
    # folds shared across conditions (paired comparisons).
    # ------------------------------------------------------------------
    n_utts = len(eval_items)
    utt_folds = np.random.default_rng(1234).integers(0, 5, size=n_utts)

    def stack(cond: str, key: str) -> np.ndarray:
        return np.concatenate(probe_store[cond][key])

    probe_results, probe_per_utt = {}, {}
    for cond in probe_store:
        vecs = np.concatenate(probe_store[cond]["vecs"], axis=0)
        utt_ids = stack(cond, "utt")
        fold_ids = utt_folds[utt_ids]
        acc = knn1_centroid_probe(vecs, stack(cond, "phone"), utt_ids, fold_ids)
        r2e = ridge_probe_r2(vecs, stack(cond, "energy"), utt_ids, fold_ids)
        r2p = ridge_probe_r2(vecs, stack(cond, "pitch"), utt_ids, fold_ids)
        r2d = ridge_probe_r2(vecs, stack(cond, "dur"), utt_ids, fold_ids)
        # Norm-only controls (geometric-artifact diagnostic).
        cn_e = ridge_r2_norm_control(vecs, stack(cond, "energy"), utt_ids, fold_ids)
        cn_p = ridge_r2_norm_control(vecs, stack(cond, "pitch"), utt_ids, fold_ids)
        cn_d = ridge_r2_norm_control(vecs, stack(cond, "dur"), utt_ids, fold_ids)
        probe_results[cond] = {
            "acc_phoneme": round(acc["acc"], 4),
            "majority_baseline": round(acc["majority_baseline"], 4),
            "n_eval": acc["n_eval"],
            "r2_energy": round(r2e["r2"], 4),
            "r2_pitch": round(r2p["r2"], 4),
            "r2_duration": round(r2d["r2"], 4),
            "r2_energy_norm_control": round(cn_e["r2"], 4),
            "r2_pitch_norm_control": round(cn_p["r2"], 4),
            "r2_duration_norm_control": round(cn_d["r2"], 4),
        }
        probe_per_utt[cond] = {
            "acc_phoneme": acc["per_utt_acc"],
            "r2_energy": r2e["per_utt_r2"],
            "r2_pitch": r2p["per_utt_r2"],
        }

    # Permutation control for the phone probe: labels shuffled across segments
    # (destroys segment→phone information, keeps class distribution + folds).
    perm_results = {}
    for cond in probe_store:
        accs = []
        for rep in range(5):
            rng_p = np.random.default_rng(500 + rep)
            lab = stack(cond, "phone").copy()
            lab = lab[rng_p.permutation(len(lab))]
            vecs = np.concatenate(probe_store[cond]["vecs"], axis=0)
            utt_ids = stack(cond, "utt")
            accs.append(knn1_centroid_probe(vecs, lab, utt_ids, utt_folds[utt_ids])["acc"])
        perm_results[cond] = round(float(np.mean(accs)), 4)

    # Paired bootstrap: condition differences per metric (partition channels
    # per utterance; probes on shared utterance subsets).
    boot = {}
    rng = np.random.default_rng(args.seed)
    comparisons = [("adaptive", "uniform"), ("adaptive", "dp"), ("dp", "uniform")]
    for a, b in comparisons:
        key = f"{a}_minus_{b}"
        boot[key] = {}
        for mk in METRIC_KEYS:
            da = np.array([r[mk] for r in per_utt[a]], float)
            db = np.array([r[mk] for r in per_utt[b]], float)
            boot[key][mk] = paired_bootstrap(da - db, args.n_boot, rng)
        for pk in ("acc_phoneme", "r2_energy", "r2_pitch"):
            ua = probe_per_utt[a].get(pk, {})
            ub = probe_per_utt[b].get(pk, {})
            common = sorted(set(ua) & set(ub))
            if common:
                da = np.array([ua[u] for u in common], float)
                db = np.array([ub[u] for u in common], float)
                boot[key][f"probe_{pk}"] = paired_bootstrap(da - db, args.n_boot, rng)

    if args.dump_records:
        dump = {}
        for cond, store in probe_store.items():
            for key in ("vecs", "phone", "energy", "pitch", "utt", "dur"):
                dump[f"{cond}__{key}"] = np.concatenate(store[key], axis=0)
        np.savez_compressed(args.dump_records, **dump)
        print(f"segment records dumped -> {args.dump_records}", flush=True)

    results = {
        "config": {
            "k": K, "seed": args.seed, "pca_dim": args.pca_dim,
            "eval_size": args.eval_size, "n_boot": args.n_boot,
            "f0_bounds": [F0_MIN, F0_MAX], "n_f0_bins": N_F0_BINS,
            "n_energy_bins": N_ENERGY_BINS,
            "soft_mse_final": out["final"]["learned_mse"],
            "speaker_channel": "NOT MEASURED — single-speaker corpus "
                               "(no speaker metadata in features index; dataset is one Nepali speaker)",
            "estimator": "plug-in MI with cluster-count shrinkage (SHRINK_K=5, "
                         "Kish effective N); identical K=32 cluster count across "
                         "hard conditions so plug-in bias is comparable",
        },
        "aggregates": aggregates,
        "bootstrap": boot,
        "per_utterance": {c: per_utt[c] for c in hard_conditions},
        "probe_results": probe_results,
        "probe_permutation": perm_results,
        "soft_rows": soft_rows,
    }
    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Summary.
    print("\n=== Experiment E: Information in adaptive masks (K=32) ===", flush=True)
    print("\n--- Partition information channels (means over 48 utts) ---", flush=True)
    header = f"{'metric':>18} | " + " | ".join(f"{c:>9}" for c in hard_conditions)
    print(header, flush=True)
    for mk in METRIC_KEYS:
        vals = [aggregates[c][mk] for c in hard_conditions]
        print(f"{mk:>18} | " + " | ".join(f"{v:>9.4f}" for v in vals), flush=True)
    print("\n--- Pooled-vector probes (utterance-level 5-fold CV) ---", flush=True)
    for cond in probe_results:
        a = probe_results[cond]
        print(f"{cond:>13}: acc_phoneme={a['acc_phoneme']:.4f} "
              f"(majority {a['majority_baseline']:.4f}, permuted {perm_results[cond]:.4f}) "
              f"r2_energy={a['r2_energy']:.4f} r2_pitch={a['r2_pitch']:.4f} "
              f"r2_duration={a['r2_duration']:.4f}", flush=True)
        print(f"{'':>13}  norm controls: energy={a['r2_energy_norm_control']:.4f} "
              f"pitch={a['r2_pitch_norm_control']:.4f} "
              f"duration={a['r2_duration_norm_control']:.4f}", flush=True)
    print("\n--- Soft-mask geometry (adaptive only) ---", flush=True)
    for k, v in aggregates["adaptive_soft_geometry"].items():
        print(f"  {k}: {v}", flush=True)
    print("\n--- Paired bootstrap (adaptive - uniform / adaptive - dp) ---", flush=True)
    for key in ("adaptive_minus_uniform", "adaptive_minus_dp", "dp_minus_uniform"):
        print(f"  [{key}]", flush=True)
        for mk, r in boot[key].items():
            if r["delta_mean"] is None:
                continue
            ci = r["ci"]
            print(f"    {mk:>18}: d={r['delta_mean']:+.4f} "
                  f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] p={r['p']}", flush=True)
    print(f"\nsaved={args.output}", flush=True)


if __name__ == "__main__":
    main()
