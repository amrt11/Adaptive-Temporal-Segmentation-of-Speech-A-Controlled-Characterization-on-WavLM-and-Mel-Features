# Experiment E — What Information Does the Adaptive Mask Decomposition Carry?

Status: **complete** (K=32 arm, seed 7, 48 test utterances, 1,536 segments per
condition; partition-level MI channels, pooled-vector probes with
utterance-level CV, permutation + geometric controls, paired bootstrap 10k).

## Question

D closed boundary forensics: learned boundaries are acoustic change points,
statistically indistinguishable from reconstruction-optimal DP on every
measured statistic. The pre-registered question for E (per the post-D review)
shifts the object of study from *where the boundaries are* to *what the
decomposition organizes*:

> **Does adaptive segmentation organize information differently than uniform
> or DP partitions — per phoneme, pitch, energy, voicing, and duration — even
> though it does not compress better (B/B2)?**

Decision rule fixed before running:

- If adaptive ≈ DP ≈ uniform on every channel → adaptive masks carry **no
  unique information**; the decomposition idea dies with the compression idea.
- If adaptive ≠ DP on some channel → that channel defines what the mechanism
  actually does, and is the only place novelty can live.

Speaker channel: **structurally impossible** — the corpus is single-speaker
(no speaker metadata in the features index; one Nepali speaker). Dropped by
design, not silently. Voicing is measured inside the pitch channel
(voicing-weighted log-F0); the discrete voicing on/off contrast is carried by
energy/pitch coverage.

## Data & setup

- Features: cached WavLM L6, PCA-64, test split first 48 utterances; K=32
  adaptive segmenter retrained with the identical B/C/D recipe (soft MSE
  0.0931 — matches C/D, faithful arm).
- Conditions: **adaptive** (soft masks, hard bounds derived via
  `warp_boundaries`), **adaptive_hard** (hard means at the learned bounds —
  isolates soft pooling from placement), **dp** (cached reconstruction-optimal
  bounds), **uniform** (equal-length bounds). All at exactly K=32.
- Labels per frame: MAS phone tokens (`experiments/alignments_c/`), per-utt
  standardized log-RMS energy quantile bins (8), pyin log-F0 (24 bins over
  65–400 Hz, voicing-probability weighted), segment durations.

## Method

1. **Partition channels** (per utterance, plug-in estimators): H(segment);
   I(segment; phone) — all edges and content-only; I(segment; energy);
   I(segment; pitch). Entropies use cluster-count shrinkage (prior strength 5,
   Kish effective N) and voicing weights for pitch. All hard conditions have
   exactly 32 clusters, so plug-in bias is comparable; H(segment) is a sanity
   check, not a discriminator.
2. **Pooled-vector probes** (the z_k test): each segment contributes one
   vector — the exact soft-mask pooling z_k = Σ m_k(t)x_t / Σ m_k(t) for
   adaptive, the segment mean otherwise. Probes: nearest-centroid phone
   accuracy (dominant phone per segment) and within-utterance-centered ridge
   R² for energy, pitch, and duration, with **utterance-level 5-fold CV**
   (folds shared across conditions → paired comparisons).
3. **Controls**: label permutation (phone probe ceiling check) and a
   **norm-only control** — predicting each target from ||z_k − mean_z^utt||
   alone, detecting the regression-to-the-mean artifact where long segments
   have pooled vectors pulled toward the utterance mean.
4. **Soft-mask geometry** (adaptive only): mass concentration Σm²/Σm, mass in
   own hard segment, peak value.
5. **Paired bootstrap** over utterances (10k resamples, 95% CI).

## Results

### Partition information channels (means; higher = more shared information)

| metric | adaptive | dp | uniform |
|---|---:|---:|---:|
| H(segment) | 4.948 | 4.794 | 4.999 |
| I(segment; phone) | 2.880 | 2.843 | 2.894 |
| I(segment; phone, content-only) | 2.770 | 2.736 | 2.784 |
| I(segment; energy) | 1.302 | 1.243 | 1.327 |
| I(segment; pitch) | 1.877 | 1.802 | 1.904 |
| η²(segment; duration) | 1.000 | 1.000 | 0.979 |

**The raw ordering (uniform > adaptive > DP on every channel) exactly tracks
H(segment)** (5.00 / 4.95 / 4.79): MI between a K-way partition and any
attribute is bounded by H(segment), and uniform maximizes H by keeping
occupancy uniform. Normalized by H(segment), the conditions are
near-identical (phone: 0.582 / 0.593 / 0.579; energy: 0.263 / 0.259 /
0.265; pitch: 0.379 / 0.376 / 0.381 — spreads of 0.5–1.4 points).
**No partition condition shows a meaningful information-organization
difference.**

### Pooled-vector probes (utterance-level 5-fold CV)

| condition | phone acc (majority / permuted) | R² energy | R² pitch | R² duration | norm-only duration |
|---|---|---:|---:|---:|---:|
| adaptive | 0.265 (0.124 / 0.011) | 0.081 | **0.291** | **0.704** | −0.007 |
| adaptive_hard | 0.279 (0.124 / 0.013) | 0.075 | 0.276 | 0.721 | 0.032 |
| dp | **0.302** (0.106 / 0.012) | **0.185** | 0.171 | 0.139 | **0.436** |
| uniform | 0.268 (0.117 / 0.012) | 0.067 | 0.206 | −0.037 | 0.006 |

- Phone probe: all conditions ≈ 0.27–0.30 vs majority 0.11 and permuted
  ≈ 0.01 — real but condition-independent segment phone purity. DP
  significantly highest (+0.034 vs adaptive, p=0.04; +0.034 vs uniform,
  p=0.026); adaptive vs uniform is a tie (−0.003, p=0.79).
- Energy: DP highest (0.185; adaptive 0.081, p<0.001 for the gap). Adaptive
  vs uniform: tie (p=0.35).
- Pitch: adaptive/adaptive_hard highest (0.291/0.276 vs DP 0.171), but the
  bootstrap over shared utterance subsets is not significant
  (adaptive−DP p=0.30, adaptive−uniform p=0.35).
- **Duration — the one adaptive-specific signal, and it survives the
  geometric control**: adaptive R² = 0.70 (norm-only control: −0.01) vs DP
  R² = 0.14 (norm-only 0.44, i.e. DP's duration "signal" is *mostly the
  artifact*) and uniform ≈ 0 (degenerate constant durations). The artifact
  control confirms the adaptive duration decodability is carried by segment
  *content*, not by vector geometry.

### Soft-mask geometry (adaptive only)

| statistic | value |
|---|---:|
| mass concentration Σm²/Σm | 0.839 |
| mass inside own hard segment | 0.885 |
| mean peak value | 0.995 |

The trained masks are **nearly hard**: concentrated, aligned with the
hard-warp boundaries, near-binary peaks. Correspondingly, adaptive vs
adaptive_hard probe differences are small (phone 0.265 vs 0.279, pitch 0.291
vs 0.276, energy 0.081 vs 0.075 — soft pooling is not adding information).

## Findings

1. **No information-organization difference at the partition level.**
   Raw MI differences are fully explained by segment-occupancy entropy; the
   normalized profiles of adaptive, DP, and uniform are indistinguishable
   (≤1.4 points across all normalized channels). Adaptive masks do not slice
   the attribute space differently than reconstruction-optimal or even
   uniform partitions.
2. **On pooled-vector probes, adaptive does not beat the reconstruction
   oracle on any channel.** DP is significantly better on phone purity
   (+0.037 vs adaptive, p=0.04) and energy (R² +0.109, p<0.001); pitch
   differences favor adaptive but do not reach significance (p=0.30).
3. **The single adaptive-specific property: segment duration is decodable
   from segment content** (R² ≈ 0.70 vs 0.14 for DP), and the norm control
   shows this is genuine content–duration coupling, not geometry. This is
   interpretable from the mechanism: density-weighted phase allocation makes
   segment length co-vary with local acoustic homogeneity, so content and
   duration become entangled in z_k. It is a *mechanism consequence*, not an
   information advantage — no downstream channel benefits from it.
4. **The masks are essentially hard boundaries.** Soft-mask geometry shows
   near-binary masks (88.5% of mass inside the corresponding hard segment),
   and soft vs hard pooling is probe-equivalent. The "masks as a separate
   object" hypothesis (the MaskFormer transfer) has no empirical support at
   this K/tau: the trained mechanism *is* a segmenter, not a mask
   decomposition.

## Verdict

**Branch 1 of the pre-registered rule: adaptive masks carry no unique
information beyond what reconstruction-optimal (or even uniform) partitions
carry — with one mechanism-specific exception (content–duration coupling)
that is a consequence of density-based allocation, not an organizational
advantage.** Combined with B/B2 (no compression gain) and C/D (no phonetic
or boundary-placement novelty), the mask-based decomposition is now fully
characterized: it is an acoustic change-point segmenter whose apparent
structure is entirely explained by acoustic reconstruction, plus a content-
duration entanglement visible only when you probe for it.

The original MaskFormer-inspired hypothesis is now exhausted on all three
fronts — representation (B/B2), linguistic structure (C), boundary mechanism
(D), and information organization (E). The honest conclusion of the program:

> Reconstruction-driven adaptive temporal segmentation of WavLM features
> converges to classical acoustic change-point segmentation; it does not
> compress better, is not uniquely phonetic, and does not organize
> information differently than optimal or even uniform partitions.

## Caveats

- Plug-in MI is biased upward; identical K=32 cluster counts across hard
  conditions keep the bias comparable, and the H(segment) tracking makes the
  conclusion robust to the bias level. Normalized-MI comparisons are the
  conservative view.
- Pitch MI and pitch R² use voicing-weighted frames; unvoiced frames carry
  zero weight rather than corrupting the statistic. The probe CV is
  utterance-level (5 folds, shared across conditions), so leakage across
  utterances is excluded; per-utterance paired bootstrap uses shared
  evaluated subsets.
- Segment count per class in the phone probe is modest (1,536 segments,
  ~100 phone classes → sparse per-class centroids); the permutation control
  (≈0.01) and majority baseline (≈0.11) bracket the achievable accuracy, and
  all conditions face identical class sparsity.
- 48 utterances, one language (Nepali), single speaker (speaker channel
  impossible), MAS-derived phones (the model's own alignments, not human
  labels).

## Seed replication (seeds 3 and 11)

The two findings that could have been seed artifacts were re-measured with
the adaptive arm retrained at seeds 3 and 11 (identical recipe;
`results_experiment_e_s3.json`, `results_experiment_e_s11.json`):

| quantity | seed 7 | seed 3 | seed 11 |
|---|---:|---:|---:|
| final soft MSE | 0.0931 | 0.0936 | 0.0930 |
| duration R² (adaptive) | 0.704 | 0.727 | 0.734 |
| duration R² (adaptive_hard) | 0.721 | 0.746 | 0.747 |
| duration R² (DP, deterministic) | 0.139 | 0.139 | 0.139 |
| norm-only control (adaptive) | −0.007 | −0.006 | +0.000 |
| phone acc (adaptive) | 0.265 | 0.258 | 0.263 |
| phone acc (DP) | 0.302 | 0.302 | 0.302 |
| adaptive−DP phone (p) | −0.037 (0.04) | −0.044 (0.007) | −0.039 (0.014) |
| adaptive−DP energy R² (p) | −0.109 (<0.001) | −0.080 (<0.001) | −0.156 (<0.001) |

**Both E findings are seed-robust**: the content–duration coupling (R² =
0.722 ± 0.016 sd across seeds, control ≈ 0) and the DP probe advantage (phone
and energy, all p < 0.05 / p < 0.001). The single-seed caveat on the
two load-bearing E results is closed.
- η²(segment; duration) ≈ 1.0 for all hard partitions is expected (a segment
  id almost determines its duration in any complete partition); uniform's
  0.979 reflects cross-utterance length variance. The informative duration
  statistic is the pooled-vector R², not η².

## Artifacts

- `experiments/experiment_e.py` — MI channels, pooled-vector probes,
  permutation + norm-only controls, soft-mask geometry, paired bootstrap
- `experiments/results_experiment_e.json` — config, aggregates, bootstrap
  CIs, per-utterance partition rows, probe results, soft-mask rows
- Seed replicates: `experiments/results_experiment_e_s3.json`,
  `experiments/results_experiment_e_s11.json`
- Reuses: `experiments/features_wavlm_l6/`, `experiments/alignments_c/`,
  `dataset/raw/wavs/` (fastspeech2.0)
