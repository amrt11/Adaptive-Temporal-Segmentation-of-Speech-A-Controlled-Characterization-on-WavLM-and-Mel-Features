"""
Experiment D — What does adaptive segmentation actually discover?

Takes the existing K=32 segmenter and performs a serious acoustic analysis of
every learned boundary. For each boundary we measure:

  boundary
    ├── spectral change (frame-difference magnitude at boundary vs within segment)
    ├── energy change (RMS discontinuity across boundary)
    ├── silence/pause proximity (distance to nearest low-energy region)
    ├── segment duration statistics
    ├── phone transition (from MAS alignments)
    └── position within utterance (relative)

Then compares learned boundaries against FOUR reference placements on the SAME
acoustic statistics:

  1. Learned boundaries (K=32 adaptive segmenter, seed 7)
  2. DP reconstruction-optimal boundaries (K=32)
  3. Uniform boundaries (K=32)
  4. Random boundaries (K=32, resampled per utterance)

The critical question: if learned ≈ DP for every acoustic statistic, then the
mechanism is discovering acoustic change points. If learned has some property
that DP does NOT have, that's where novelty might hide.

Outputs: experiments/results_experiment_d.json

Run from research root with the fastspeech2.0 venv python.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiment_a import set_seed
from experiment_b import (
    FEATURE_DIR,
    apply_pca,
    fit_pca,
    load_items,
    pad_batch,
    warp_boundaries,
)
from experiment_b import PhaseDurationSegmenter as Segmenter

RESEARCH = Path(__file__).resolve().parent.parent
ALIGN_DIR = RESEARCH / "experiments" / "alignments_c"
WAV_DIR = Path("C:/Users/Lenovo/Desktop/fastspeech2.0/dataset/raw/wavs")
SSL_RATE = 50.0
MEL_RATE = 22050.0 / 256.0
MEL_TO_SSL = SSL_RATE / MEL_RATE

# Token IDs treated as silence/pause/structural (from experiment_c.py).
STRUCTURAL_IDS = {0, 1, 2, 3, 4, 5, 21, 51, 52, 53}

# Tolerances (in SSL frames, 1 frame = 20 ms) for phone-edge hit-rate sweep.
PHONE_TOLERANCES = (0, 1, 2, 3, 5, 8, 13)


# ---------------------------------------------------------------------------
# Acoustic feature extraction from waveforms
# ---------------------------------------------------------------------------

def load_wav(utt_id: str) -> np.ndarray:
    """Load 16 kHz mono waveform for an utterance."""
    import soundfile as sf
    wav_path = WAV_DIR / f"{utt_id}.wav"
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def rms_energy(wav: np.ndarray, frame_len: int = 256, hop: int = 128) -> np.ndarray:
    """Per-frame RMS energy (parallel to mel frame rate ~22050/256 Hz)."""
    wav = np.pad(wav, (0, frame_len))
    frames = []
    for i in range(0, len(wav) - frame_len + 1, hop):
        frames.append(wav[i:i + frame_len])
    if not frames:
        return np.array([0.0])
    frames = np.stack(frames)
    return np.sqrt((frames ** 2).mean(axis=1))


def spectral_change(feats: np.ndarray) -> np.ndarray:
    """Frame-to-frame spectral change magnitude (L2 norm of difference)."""
    if feats.shape[0] < 2:
        return np.zeros(feats.shape[0])
    diff = np.diff(feats, axis=0)
    return np.sqrt((diff ** 2).sum(axis=1))


def voicing_probability(wav: np.ndarray, sr: int = 16000,
                        frame_len: int = 256, hop: int = 128,
                        f0_min: float = 75.0, f0_max: float = 400.0) -> np.ndarray:
    """
    Simple voicing measure: normalized autocorrelation peak at expected F0 lag.

    Returns a per-frame voicing score in [0, 1]. Not a proper F0 extractor;
    just a quick acoustic proxy for "is this frame periodic?".
    """
    wav = np.pad(wav, (0, frame_len))
    frames = []
    for i in range(0, len(wav) - frame_len + 1, hop):
        frames.append(wav[i:i + frame_len])
    if not frames:
        return np.zeros(1)
    frames = np.stack(frames)
    # Expected lag range for F0 in samples at 16 kHz.
    lag_min = int(sr / f0_max)
    lag_max = int(sr / f0_min)
    lag_min = max(1, lag_min)
    lag_max = min(frame_len // 2, lag_max)
    if lag_max <= lag_min:
        return np.zeros(frames.shape[0])
    lags = np.arange(lag_min, lag_max + 1)
    frames_centered = frames - frames.mean(axis=1, keepdims=True)
    n_frames, width = frames_centered.shape
    # Prefix sums of squared samples for cheap energy normalization.
    csum = np.concatenate(
        [np.zeros((n_frames, 1)), np.cumsum(frames_centered ** 2, axis=1)], axis=1
    )  # [N, W+1]
    # Normalized short-time autocorrelation at each lag:
    #   ac(f, L) = sum_t f[t] f[t+L] / (||f[:W-L]|| * ||f[L:]||)
    ac = np.empty((len(lags), n_frames))
    for li, lag in enumerate(lags):
        num = (frames_centered[:, :-lag] * frames_centered[:, lag:]).sum(axis=1)
        den = np.sqrt(
            (csum[:, width - lag] * (csum[:, width] - csum[:, lag])).clip(min=1e-12)
        )
        ac[li] = num / den
    # Take max over lags as voicing proxy.
    voicing = np.abs(ac).max(axis=0)
    # Normalize to roughly [0,1] via tanh scaling.
    voicing = np.tanh(voicing * 3.0)
    return voicing


# ---------------------------------------------------------------------------
# Boundary analysis
# ---------------------------------------------------------------------------

class BoundaryAnalyzer:
    """
    Computes acoustic statistics for a set of boundaries on a single utterance.

    For each boundary b_k (between segment k and k+1), we measure:

    - spectral_change_at: mean spectral change magnitude in +/-2 frames around the boundary.
    - spectral_change_nearby: mean spectral change in the surrounding segment interiors
      (excluding the boundary zone).
    - energy_jump: |log(energy_{b+1}) - log(energy_{b-1})|
    - energy_min_proximity: distance (in SSL frames) to nearest low-energy frame
      (energy below 20th percentile of utterance), capturing silence/pause adjacency.
    - segment_duration_k: duration of segment k (frames).
    - segment_duration_k1: duration of segment k+1 (frames).
    - relative_position: b_k / T (where in utterance the boundary sits).
    - phone_transition: 1 if a MAS phone edge falls within +/-2 SSL frames, else 0.
    - content_phone_transition: same but excluding structural tokens.
    - structural_boundary: 1 if boundary is within +/-2 frames of a structural token edge.
    """

    def __init__(self, feats: np.ndarray, align: dict, wav: np.ndarray,
                 ssl_to_mel: float = MEL_TO_SSL):
        self.feats = feats  # [T_ssl, D] PCA features
        self.align = align
        self.wav = wav
        self.ssl_to_mel = ssl_to_mel
        self.T = len(feats)

        # Precompute per-frame acoustic features at SSL rate.
        # We need features aligned to SSL frames (50 Hz).
        # Mel frames are at ~86.13 Hz; we'll resample.
        self.mel_energy = rms_energy(wav)
        self.ssl_energy = self._resample_to_ssl(self.mel_energy)
        self.spectral_change = spectral_change(feats)

        # Voicing at mel rate, resample to SSL.
        self.mel_voicing = voicing_probability(wav)
        self.ssl_voicing = self._resample_to_ssl(self.mel_voicing)

        # Phone edges in SSL frames.
        self.phone_edges_all, self.phone_edges_content = self._phone_edges_ssl()
        self.phone_edges_arr = np.asarray(self.phone_edges_all, dtype=int)
        self.phone_edges_content_arr = np.asarray(self.phone_edges_content, dtype=int)

        # Utterance-level statistics of spectral change, for z-scoring.
        self.sc_mean = float(self.spectral_change.mean()) if self.T > 1 else 0.0
        self.sc_std = float(self.spectral_change.std()) if self.T > 1 else 1.0
        self.sc_std = max(self.sc_std, 1e-6)

        # Low-energy threshold (20th percentile).
        self.energy_threshold = np.percentile(self.ssl_energy, 20)
        self.low_energy_frames = np.where(self.ssl_energy < self.energy_threshold)[0]

    def _resample_to_ssl(self, mel_frame_values: np.ndarray) -> np.ndarray:
        """Resample mel-rate (~22050/256 Hz) values to SSL 50 Hz grid."""
        if len(mel_frame_values) == 0:
            return np.zeros(self.T)
        # mel frame i centers at (i + 0.5) / mel_rate seconds.
        # ssl frame j centers at (j + 0.5) / ssl_rate seconds.
        mel_rate = 22050.0 / 256.0
        ssl_times = (np.arange(self.T) + 0.5) / SSL_RATE
        mel_times = (np.arange(len(mel_frame_values)) + 0.5) / mel_rate
        # Linear interpolation.
        from scipy.interpolate import interp1d
        f = interp1d(mel_times, mel_frame_values, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        return f(ssl_times)

    def _phone_edges_ssl(self) -> tuple[list[int], list[int]]:
        """MAS phone edges in SSL frames."""
        durations = self.align["durations"]
        tokens = self.align["tokens"]
        edges_all = []
        edges_content = []
        cum = 0
        for tok, d in zip(tokens[:-1], durations[:-1]):
            cum += d
            edges_all.append(int(round(cum * self.ssl_to_mel)))
            if int(tok) not in STRUCTURAL_IDS:
                edges_content.append(int(round(cum * self.ssl_to_mel)))
        return edges_all, edges_content

    def analyze_boundary(self, b: int, k: int, T: int) -> dict:
        """Compute acoustic statistics for boundary at frame b, between seg k and k+1."""
        b = int(b)
        T = int(T)

        # Clamp for safety.
        if b <= 0 or b >= T - 1:
            return {
                "valid": False,
                "spectral_change_at": None,
                "spectral_change_ratio": None,
                "energy_jump": None,
                "energy_min_proximity": None,
                "segment_duration_k": None,
                "segment_duration_k1": None,
                "relative_position": None,
                "phone_transition": 0,
                "content_phone_transition": 0,
                "structural_boundary": 0,
            }

        # Spectral change at boundary (mean of +/-2 frames around b).
        lo = max(0, b - 2)
        hi = min(T, b + 3)
        spectral_at = float(np.mean(self.spectral_change[lo:hi])) if hi > lo else 0.0

        # Spectral change in surrounding segment interiors (avoid boundary zone).
        # Segment k interior: [prev_boundary+1, b-3]
        # Segment k+1 interior: [b+3, next_boundary-1]
        # We'll use wider context: the 10 frames before b (not too close) and
        # 10 frames after b.
        before_lo = max(0, b - 10)
        before_hi = max(0, b - 3)
        after_lo = min(T, b + 3)
        after_hi = min(T, b + 10)
        before_vals = self.spectral_change[before_lo:before_hi] if before_hi > before_lo else np.array([0.0])
        after_vals = self.spectral_change[after_lo:after_hi] if after_hi > after_lo else np.array([0.0])
        nearby_mean = float(np.mean(np.concatenate([before_vals, after_vals]))) if len(before_vals) + len(after_vals) > 0 else 0.0

        # Ratio: spectral_change_at / nearby_mean (how much does the boundary stand out?).
        spectral_ratio = spectral_at / max(nearby_mean, 1e-9)

        # Energy jump: log energy difference.
        e_before = self.ssl_energy[max(0, b - 1)]
        e_after = self.ssl_energy[min(T - 1, b + 1)]
        energy_jump = float(abs(math.log(max(e_after, 1e-9)) - math.log(max(e_before, 1e-9))))

        # Distance to nearest low-energy frame.
        if len(self.low_energy_frames) > 0:
            dists = np.abs(self.low_energy_frames - b)
            energy_min_proximity = float(np.min(dists))
        else:
            energy_min_proximity = float('inf')

        # Segment durations — we need the full boundary list to compute this.
        # We'll pass segment durations separately.

        # Relative position.
        relative_position = b / T

        # Phone transition: tolerance sweep. A boundary is a "hit" if a phone        # edge falls within +/-tol frames. We record hits at several tolerances
        # because a fixed +/-2 frames (~40 ms) saturates when mean phone duration is short.
        phone_hits = {tol: 0 for tol in PHONE_TOLERANCES}
        content_hits = {tol: 0 for tol in PHONE_TOLERANCES}
        if len(self.phone_edges_arr):
            d_all = np.abs(self.phone_edges_arr - b)
            for tol in PHONE_TOLERANCES:
                phone_hits[tol] = int((d_all <= tol).any())
        if len(self.phone_edges_content_arr):
            d_c = np.abs(self.phone_edges_content_arr - b)
            for tol in PHONE_TOLERANCES:
                content_hits[tol] = int((d_c <= tol).any())

        # Structural boundary: edge adjacent to a structural token within +/-2
        # SSL frames (kept at the original fixed tolerance on purpose).
        structural = 0
        for i, (tok, d) in enumerate(zip(self.align["tokens"][:-1], self.align["durations"][:-1])):
            edge_frame = int(round(sum(self.align["durations"][:i+1]) * self.ssl_to_mel))
            if abs(edge_frame - b) <= 2 and int(tok) in STRUCTURAL_IDS:
                structural = 1
                break

        return {
            "valid": True,
            "spectral_change_at": round(spectral_at, 6),
            "spectral_change_ratio": round(spectral_ratio, 4),
            "spectral_change_z": round((spectral_at - self.sc_mean) / self.sc_std, 4),
            "energy_jump": round(energy_jump, 6),
            "energy_min_proximity": round(energy_min_proximity, 2),
            "relative_position": round(relative_position, 4),
            "phone_hits": phone_hits,
            "content_hits": content_hits,
            "phone_transition": phone_hits[2],
            "content_phone_transition": content_hits[2],
            "structural_boundary": structural,
        }


def analyze_boundaries(analyzer: BoundaryAnalyzer, boundaries: list[int],
                       segment_durations: list[int]) -> list[dict]:
    """Analyze all boundaries with segment duration context."""
    results = []
    T = analyzer.T
    for k, b in enumerate(boundaries):
        stats = analyzer.analyze_boundary(b, k, T)
        if stats["valid"]:
            stats["segment_duration_k"] = int(segment_durations[k]) if k < len(segment_durations) else None
            stats["segment_duration_k1"] = int(segment_durations[k + 1]) if (k + 1) < len(segment_durations) else None
        results.append(stats)
    return results


def segment_durations_from_boundaries(boundaries: list[int], T: int) -> list[int]:
    """Compute segment durations from boundary list."""
    starts = [0] + list(boundaries)
    ends = list(boundaries) + [T]
    return [e - s for s, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
# Random boundary generation (matched to K)
# ---------------------------------------------------------------------------

def random_boundaries(t: int, k: int, rng: random.Random) -> list[int]:
    """Random K-1 distinct boundaries in [1, t-1]."""
    if k <= 1:
        return []
    return sorted(rng.sample(range(1, t), k - 1))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment D: acoustic analysis of learned boundaries")
    parser.add_argument("--ks", type=str, default="32")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--output", type=Path,
                        default=RESEARCH / "experiments" / "results_experiment_d.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Load data.
    train_items = load_items("train", device)
    pca = fit_pca(train_items)
    test_items = load_items("test", device)
    # PCA must be applied to BOTH splits: train_arm reads torch tensors
    # (device + dim) off the train items via pad_batch.
    apply_pca(train_items, pca, args.pca_dim, device)
    apply_pca(test_items, pca, args.pca_dim, device)

    eval_items = test_items[:args.eval_size]
    print(f"eval utterances: {len(eval_items)}", flush=True)

    # Load DP boundaries (cached from Experiment B/C) — needed by BOTH the
    # boundary loss during training (train items) and the analysis (eval items).
    dp_store = json.loads((FEATURE_DIR / f"dp_bounds_pca{args.pca_dim}.json").read_text())
    K = args.k
    for it in train_items:
        it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}
    for it in eval_items:
        it["dp_bounds"] = {K: dp_store[str(K)][it["id"]]}

    # Load the trained K=32 segmenter from Experiment C.
    # We need to retrain it with the exact same config. The experiment_c.py
    # script trains K=16 and K=32 arms. We'll extract just the K=32 arm.
    #
    # For efficiency, we'll reload the model using the same training recipe
    # but we can also just train it here. Since we have the cached features,
    # this is fast (~30 seconds per arm).
    print(f"training K={K} adaptive segmenter (seed {args.seed})...", flush=True)

    from experiment_c import train_arm as train_arm_c
    # Re-implement minimal training here to avoid coupling to experiment_c's CLI.

    # Actually, let's just use experiment_b's train_arm since it's the same model.
    from experiment_b import train_arm

    batches = [list(range(i, min(i + 8, len(train_items)))) for i in range(0, len(train_items), 8)]
    eval_batches = [list(range(i, min(i + 8, len(eval_items)))) for i in range(0, len(eval_items), 8)]

    model, out = train_arm(train_items, batches, eval_items, eval_batches, K, argparse.Namespace(
        latent_dim=32, warmup_steps=300, anneal_steps=1700, batch_size=8, lr=2e-3,
        tau_start=0.35, tau_end=0.08, boundary_weight=1.0, entropy_weight=1e-3,
        eval_every=1000, pca_dim=args.pca_dim, seed=args.seed, vq_weight=0.0,
    ), pca)

    print(f"  K={K} final soft MSE {out['final']['learned_mse']:.4f}", flush=True)

    # Load alignments.
    alignments = {}
    for it in eval_items:
        path = ALIGN_DIR / f"{it['id']}.json"
        if not path.exists():
            raise SystemExit(f"missing alignment for {it['id']}")
        alignments[it["id"]] = json.loads(path.read_text())

    # Load waveforms (lazy, only for eval set).
    wav_cache = {}
    for it in eval_items:
        wav_cache[it["id"]] = load_wav(it["id"])

    # Analyze each utterance.
    rng = random.Random(args.seed)
    all_analysis = {
        "learned": [],
        "dp": [],
        "uniform": [],
        "random": [],
    }

    per_utterance_stats = []

    with torch.no_grad():
        for idx, it in enumerate(eval_items):
            uid = it["id"]
            T = it["T"]
            feats_np = it["features"].detach().cpu().numpy()

            # Get learned boundaries.
            x = it["features"].unsqueeze(0)
            mask = torch.ones(1, T, device=device)
            _, masks, edge_c, _, _ = model.forward_full(x, mask)
            learned_bounds = warp_boundaries(edge_c[0], T, K)
            learned_durs = segment_durations_from_boundaries(learned_bounds, T)

            # DP boundaries.
            dp_bounds = it["dp_bounds"][K]
            dp_durs = segment_durations_from_boundaries(dp_bounds, T)

            # Uniform boundaries.
            uniform_bounds = [round(i * T / K) for i in range(1, K)]

            # Random boundaries.
            random_bounds = random_boundaries(T, K, rng)

            # Create analyzer for this utterance.
            analyzer = BoundaryAnalyzer(feats_np, alignments[uid], wav_cache[uid])

            # Analyze each set of boundaries.
            for name, bounds, durs in [
                ("learned", learned_bounds, learned_durs),
                ("dp", dp_bounds, dp_durs),
                ("uniform", uniform_bounds, segment_durations_from_boundaries(uniform_bounds, T)),
                ("random", random_bounds, segment_durations_from_boundaries(random_bounds, T)),
            ]:
                stats = analyze_boundaries(analyzer, bounds, durs)
                all_analysis[name].extend(stats)

            # Store per-utterance summary.
            per_utterance_stats.append({
                "id": uid,
                "T": T,
                "n_phones": len(alignments[uid]["tokens"]),
                "learned_bounds": learned_bounds,
                "dp_bounds": dp_bounds,
                "uniform_bounds": uniform_bounds,
                "random_bounds": random_bounds,
                "learned phone_hits": sum(1 for s in analyze_boundaries(analyzer, learned_bounds, learned_durs) if s.get("phone_transition", 0)),
                "dp phone_hits": sum(1 for s in analyze_boundaries(analyzer, dp_bounds, dp_durs) if s.get("phone_transition", 0)),
                "uniform phone_hits": sum(1 for s in analyze_boundaries(analyzer, uniform_bounds,
                                       segment_durations_from_boundaries(uniform_bounds, T)) if s.get("phone_transition", 0)),
                "random phone_hits": sum(1 for s in analyze_boundaries(analyzer, random_bounds,
                                       segment_durations_from_boundaries(random_bounds, T)) if s.get("phone_transition", 0)),
                "learned structural_hits": sum(1 for s in analyze_boundaries(analyzer, learned_bounds, learned_durs) if s.get("structural_boundary", 0)),
                "dp structural_hits": sum(1 for s in analyze_boundaries(analyzer, dp_bounds, dp_durs) if s.get("structural_boundary", 0)),
            })

            if idx % 10 == 0:
                print(f"  analyzed {idx+1}/{len(eval_items)}: {uid} T={T}", flush=True)

    # Explode per-boundary tolerance dicts into flat numeric keys.
    for stats in all_analysis.values():
        for s in stats:
            if not s.get("valid", False):
                continue
            for tol, hit in s.pop("phone_hits", {}).items():
                s[f"_hit{tol}"] = hit
            for tol, hit in s.pop("content_hits", {}).items():
                s[f"_chit{tol}"] = hit

    # Aggregate statistics per condition.
    def aggregate(stats: list[dict], key: str):
        vals = [s[key] for s in stats if s.get("valid", False) and s[key] is not None]
        if not vals:
            return None
        return {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    def fraction(stats: list[dict], key: str):
        vals = [s[key] for s in stats if s.get("valid", False)]
        if not vals:
            return None
        return round(float(np.mean(vals)), 4)

    conditions = {}
    for name in ("learned", "dp", "uniform", "random"):
        stats = all_analysis[name]
        conditions[name] = {
            "n_boundaries": len(stats),
            "n_valid": sum(1 for s in stats if s.get("valid", False)),
            "spectral_change_at": aggregate(stats, "spectral_change_at"),
            "spectral_change_ratio": aggregate(stats, "spectral_change_ratio"),
            "spectral_change_z": aggregate(stats, "spectral_change_z"),
            "energy_jump": aggregate(stats, "energy_jump"),
            "energy_min_proximity": aggregate(stats, "energy_min_proximity"),
            "relative_position": aggregate(stats, "relative_position"),
            "segment_duration_k": aggregate(stats, "segment_duration_k"),
            "segment_duration_k1": aggregate(stats, "segment_duration_k1"),
            "phone_transition_fraction": fraction(stats, "phone_transition"),
            "content_phone_transition_fraction": fraction(stats, "content_phone_transition"),
            "structural_boundary_fraction": fraction(stats, "structural_boundary"),
            "phone_hit_rate_by_tolerance": {
                str(tol): fraction(stats, f"_hit{tol}") for tol in PHONE_TOLERANCES
            },
            "content_hit_rate_by_tolerance": {
                str(tol): fraction(stats, f"_chit{tol}") for tol in PHONE_TOLERANCES
            },
        }

    # Comparison table: learned vs dp vs uniform vs random on key metrics.
    comparison = {}
    for metric_name, key in [
        ("spectral_change_at", "spectral_change_at"),
        ("spectral_change_ratio", "spectral_change_ratio"),
        ("spectral_change_z", "spectral_change_z"),
        ("energy_jump", "energy_jump"),
        ("energy_min_proximity", "energy_min_proximity"),
        ("phone_transition_fraction", "phone_transition_fraction"),
        ("structural_boundary_fraction", "structural_boundary_fraction"),
    ]:
        comparison[metric_name] = {}
        for cond in ("learned", "dp", "uniform", "random"):
            val = conditions[cond].get(key)
            if isinstance(val, dict):
                comparison[metric_name][cond] = val["mean"]
            else:
                comparison[metric_name][cond] = val

    # Pairwise comparisons: learned vs each baseline.
    pairwise = {}
    for baseline in ("dp", "uniform", "random"):
        pair = {}
        for metric_name in comparison:
            learned_val = comparison[metric_name].get("learned")
            baseline_val = comparison[metric_name].get(baseline)
            if learned_val is not None and baseline_val is not None:
                pair[metric_name] = {
                    "learned": learned_val,
                    "baseline": baseline_val,
                    "ratio": round(learned_val / max(baseline_val, 1e-9), 3),
                    "diff": round(learned_val - baseline_val, 4),
                }
        pairwise[baseline] = pair

    # Perturbation analysis: for learned AND DP boundaries, how much does moving
    # them change the acoustic statistics? Symmetric by design: if learned and DP
    # respond identically to displacement, the learned mechanism is basically
    # acoustic change-point detection; any divergence is where novelty could hide.
    perturbation_stats = []
    deltas = [-16, -8, -4, -2, 2, 4, 8, 16]
    for it in eval_items[:16]:  # subset for speed
        uid = it["id"]
        T = it["T"]
        feats_np = it["features"].detach().cpu().numpy()
        analyzer = BoundaryAnalyzer(feats_np, alignments[uid], wav_cache[uid])

        x = it["features"].unsqueeze(0)
        mask = torch.ones(1, T, device=device)
        _, _, edge_c, _, _ = model.forward_full(x, mask)
        learned_bounds = warp_boundaries(edge_c[0], T, K)

        for cond_name, base_bounds in (
            ("learned", learned_bounds),
            ("dp", list(it["dp_bounds"][K])),
        ):
            base_durs = segment_durations_from_boundaries(base_bounds, T)
            base_stats = analyze_boundaries(analyzer, base_bounds, base_durs)

            for k, b in enumerate(base_bounds):
                if b <= 0 or b >= T - 1:
                    continue
                lo = (base_bounds[k - 1] + 1) if k > 0 else 1
                hi = (base_bounds[k + 1] - 1) if k < K - 2 else T - 1
                for delta in deltas:
                    moved = int(np.clip(b + delta, lo, hi))
                    if moved == b:
                        continue
                    moved_bounds = list(base_bounds)
                    moved_bounds[k] = moved
                    moved_durs = segment_durations_from_boundaries(moved_bounds, T)
                    moved_stats = analyze_boundaries(analyzer, moved_bounds, moved_durs)
                    if moved_stats[k].get("valid", False) and base_stats[k].get("valid", False):
                        perturbation_stats.append({
                            "condition": cond_name,
                            "utt": uid,
                            "boundary_idx": k,
                            "original_pos": b,
                            "moved_pos": moved,
                            "delta": delta,
                            "spectral_change_at_diff": round(
                                moved_stats[k]["spectral_change_at"] - base_stats[k]["spectral_change_at"], 6
                            ),
                            "energy_jump_diff": round(
                                moved_stats[k]["energy_jump"] - base_stats[k]["energy_jump"], 6
                            ),
                            "phone_transition_changed": int(moved_stats[k]["phone_transition"]) - int(base_stats[k]["phone_transition"]),
                        })

    # Aggregate perturbation results by condition and delta magnitude.
    perturb_agg = {}
    for cond_name in ("learned", "dp"):
        cond_ps = [p for p in perturbation_stats if p["condition"] == cond_name]
        perturb_agg[cond_name] = {}
        for abs_delta in sorted(set(abs(p["delta"]) for p in cond_ps)):
            subset = [p for p in cond_ps if abs(p["delta"]) == abs_delta]
            if subset:
                perturb_agg[cond_name][str(abs_delta)] = {
                    "n": len(subset),
                    "spectral_change_at_mean_abs_diff": round(float(np.mean([abs(p["spectral_change_at_diff"]) for p in subset])), 6),
                    "energy_jump_mean_abs_diff": round(float(np.mean([abs(p["energy_jump_diff"]) for p in subset])), 6),
                    "phone_transition_flips": sum(1 for p in subset if p["phone_transition_changed"] != 0),
                }

    results = {
        "config": {
            "k": K,
            "seed": args.seed,
            "pca_dim": args.pca_dim,
            "eval_size": args.eval_size,
            "ssl_rate": SSL_RATE,
            "wav_rate": 16000,
            "output": str(args.output),
        },
        "conditions": conditions,
        "comparison": comparison,
        "pairwise_learned_vs": pairwise,
        "perturbation": {
            "n_boundaries_perturbed": len(perturbation_stats),
            "by_condition": perturb_agg,
        },
        "per_utterance_summary": per_utterance_stats,
    }

    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Print summary.
    print("\n=== Experiment D: Acoustic Analysis of Learned Boundaries (K=32) ===", flush=True)
    print("\n--- Condition summary (means) ---", flush=True)
    for name in ("learned", "dp", "uniform", "random"):
        c = conditions[name]
        print(f"\n{name} (n={c['n_valid']} valid boundaries):", flush=True)
        print(f"  spectral_change_at mean: {c['spectral_change_at']['mean'] if c['spectral_change_at'] else 'n/a'}", flush=True)
        print(f"  spectral_change_ratio mean: {c['spectral_change_ratio']['mean'] if c['spectral_change_ratio'] else 'n/a'}", flush=True)
        print(f"  spectral_change_z mean: {c['spectral_change_z']['mean'] if c['spectral_change_z'] else 'n/a'}", flush=True)
        print(f"  energy_jump mean: {c['energy_jump']['mean'] if c['energy_jump'] else 'n/a'}", flush=True)
        print(f"  energy_min_proximity mean: {c['energy_min_proximity']['mean'] if c['energy_min_proximity'] else 'n/a'}", flush=True)
        print(f"  phone_transition fraction: {c['phone_transition_fraction']}", flush=True)
        print("  phone hit-rate by tolerance (frames): " + ", ".join(
            f"+/-{tol}={c['phone_hit_rate_by_tolerance'][str(tol)]}" for tol in PHONE_TOLERANCES
        ), flush=True)
        print(f"  structural_boundary fraction: {c['structural_boundary_fraction']}", flush=True)

    print("\n--- Learn vs baselines (means) ---", flush=True)
    for metric_name, vals in comparison.items():
        print(f"\n{metric_name}:", flush=True)
        for cond, val in vals.items():
            print(f"  {cond}: {val}", flush=True)

    print("\n--- Perturbation analysis (learned vs dp) ---", flush=True)
    print(f"  boundaries perturbed: {len(perturbation_stats)}", flush=True)
    for cond_name, by_delta in perturb_agg.items():
        print(f"  [{cond_name}]", flush=True)
        for delta, stats in by_delta.items():
            print(f"    |delta|={delta}: n={stats['n']}, "
                  f"spectral_change_abs_diff={stats['spectral_change_at_mean_abs_diff']:.4f}, "
                  f"energy_jump_abs_diff={stats['energy_jump_mean_abs_diff']:.4f}, "
                  f"phone_flips={stats['phone_transition_flips']}", flush=True)

    print(f"\nsaved={args.output}", flush=True)


if __name__ == "__main__":
    main()
