# Experiment F — Does Segmentation Become Useful on Acoustic Representations?

Status: **complete** (B2 protocol transplanted to mel; K=32, codebooks
{64, 256, 1024}, seed 7, 48 test utterances; mel DP warm-start bounds
recomputed and cached).

## Question

B2's negative result was scoped to one representation class: smooth,
content-biased WavLM L6 features. The pre-registered question (professor's
Direction 4):

> If frame-VQ wins again on mel, the general negative result holds across
> semantic **and** acoustic representations. If segment-VQ becomes
> competitive on mel, temporal segmentation is useful exactly where acoustic
> structure is explicit — a finding about *where temporal abstraction
> happens* in speech encoders.

## Decision rule (fixed before running)

- **F-NULL**: frame MSE < adaptive at every codebook N, gap similar to
  WavLM (frame/adaptive ≈ 0.5–0.66) → negative result is
  representation-class-independent.
- **F-POSITIVE**: adaptive ≤ frame at some N, or frame/adaptive > 0.9.
- **PARTIAL**: frame/adaptive ∈ (0.7, 0.9).
- **Sanity**: adaptive must beat uniform on mel (it did on WavLM in B);
  otherwise the segmenter failed to train (confound).

## Setup

- Features: **native 80-dim log-mel** from `dataset/processed/mel/`
  (22050 Hz, hop 256, n_fft 1024, fmax 8000 — the exact pipeline the MAS
  alignments were built on; sum(durations) == mel T verified). NO PCA —
  discarding acoustic detail would defeat the test. Per-dimension z-score
  standardization (train stats), identical for all arms.
- Frame rate 86.13 Hz (vs 50 Hz for WavLM features).
- Arms (identical recipe to B2: latent 32, warmup 300 + anneal 1700 steps,
  lr 2e-3, vq_weight 1.0, batch 8, seed 7): **frame** (per-frame VQ),
  **uniform** segment-VQ (fixed uniform partition), **adaptive** segment-VQ
  (density-warp segmenter).
- Representation-specific: **DP warm-start bounds recomputed on standardized
  mel** (cached `experiments/dp_bounds_mel_K32.json`, 1,180 utterances) —
  the cached WavLM DP bounds are meaningless for mel.
- Bit accounting identical to B2: frame T·log2(N); segment arms K·log2(N);
  adaptive adds log2 C(T, K−1) boundary side information.

## Results — rate-distortion table, K=32 (lower MSE = better)

| N | frame MSE | adaptive MSE | uniform MSE | frame/adaptive | adaptive bits/s | frame bits/s |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.1588 | 0.2214 | 0.5014 | 0.717 | 59 | 517 |
| 256 | 0.1339 | 0.2221 | 0.5194 | 0.603 | 70 | 689 |
| 1024 | 0.1203 | 0.2214 | 0.5526 | 0.543 | 81 | 861 |

Comparison with the WavLM panel (B2, same protocol/seed):

| N | WavLM frame/adaptive | mel frame/adaptive |
|---:|---:|---:|
| 64 | 0.605 | 0.717 |
| 256 | 0.509 | 0.603 |
| 1024 | 0.449 | 0.543 |

Code usage (perplexity): frame 57/209/661 (healthy); adaptive 17.8/13.5/21.7
— comparable to its WavLM values (22.3/18.5/25.7); uniform degenerate
(10–38). Not a codebook-collapse artifact.

## Findings

1. **F-NULL: frame-VQ wins at every codebook size on mel too.** Adaptive
   never gets within a factor 0.7 of frame, and the gap *widens* with N
   (0.717 → 0.543) because adaptive MSE is flat in N (~0.222 across
   64→1024) exactly as on WavLM (~0.184). The segment bottleneck — not the
   codebook — binds, on both representation classes.
2. **The WavLM gap is replicated, not exaggerated.** Mel is somewhat kinder
   to segments at small N (0.717 vs 0.605 at N=64) — consistent with mel's
   stronger local temporal structure giving segment means slightly more
   information per segment — but the effect is a marginal PARTIAL at one
   point, not a regime change. Nowhere near the >0.9 F-POSITIVE band, and
   the WavLM gap is actually larger at N=1024 (0.449 vs 0.543).
3. **Sanity check passes: the mechanism transfers.** Adaptive beats uniform
   2.2× on mel (0.221 vs 0.501), replicating B's WavLM result (0.183 vs
   0.420). The segmenter trains and places informative boundaries on this
   representation; the negative comparison is not a training failure.
4. **The negative result is now representation-class-independent.**
   Temporal segmentation does not provide an efficient discrete
   representation of speech on either semantic (WavLM L6) or acoustic
   (log-mel) features, under matched bits, identical decoders, and a
   reconstruction-optimal warm start for the segment arms.

## Verdict

**F-NULL (with a marginal PARTIAL at N=64 only).** The professor's framing
is confirmed: "Temporal segmentation does not provide an efficient discrete
representation across both semantic and acoustic speech representations" —
and the design is now rigorous enough (matched bits, matched decoders,
DP warm start, representation-matched preprocessing, sanity controls) that
this negative is publishable as a characterization result.

With B2 (WavLM) and F (mel), the compression question is closed on both
representation classes. Remaining open directions are multi-scale
segmentation (Direction 2) and downstream-usefulness questions (Direction
3), which are new research programs rather than further falsification of
this one.

## Caveats

- MSE at the mel frame rate is not directly comparable in magnitude to the
  WavLM panel (86 vs 50 Hz, different feature scales); the controlled
  comparison is frame/adaptive *within* each representation.
- Single seed for the mel panel (B2 was also single-seed per point); the
  cross-seed variance measured in C/E (sd ≈ 0.002 soft MSE) is an order of
  magnitude smaller than the gaps reported here.
- The adaptive arm's flat MSE in N means codebook size is not the binding
  constraint; the segment-mean bottleneck is. A codebook-size sweep beyond
  1024 is therefore not expected to change the verdict (B2's own caveat,
  now tested on a second representation).
- Mel DP bounds are reconstruction-optimal on standardized mel; the adaptive
  arm consumes them only as a warm-start target (300 steps), as in B2.

## Artifacts

- `experiments/experiment_f.py` — mel data loading, mel DP cache, B2 arms
  reused via `experiment_b_vq.py`
- `experiments/results_experiment_f.json` — 9 RD points with exact bit
  accounting
- `experiments/dp_bounds_mel_K32.json` — cached mel DP bounds (1,180 utts)
- Reuses: `dataset/processed/mel/` (fastspeech2.0), `experiment_b_vq.py`
  models/training loops
