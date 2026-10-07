# Experiment D — What Does Adaptive Segmentation Actually Discover?

Status: **complete** (K=32 arm, seed 7, 48 test utterances, 1,488 valid
boundaries per condition — 1,484 for random; learnable-mechanism metrics +
displacement-response curves).

## Question

Experiments B/B2 killed the compression claim and C established that learned
boundaries beat uniform/random on phone-edge alignment but statistically tie
the reconstruction-optimal DP. The pre-registered question for D (per the
project review) was:

> **For every learned boundary: is there a phone transition, silence, a pitch
> or energy discontinuity, a spectral change — or nothing obvious? And does
> the learned placement have any property that the DP reconstruction-optimal
> placement does not?**

Two-way interpretation, fixed before running:

- If learned ≈ DP on *every* acoustic statistic → the mechanism is
  **acoustic change-point discovery**; say so and stop refining boundaries.
- If learned ≠ DP on some statistic → that property is where genuine novelty
  could hide, and it defines the next experiment.

## Data

- Features: cached WavLM L6 (Experiment B), PCA-64; test split, first 48
  utterances (T mean 371 SSL frames).
- Alignments: MAS phone durations from the v6 FastSpeech2 checkpoint
  (`experiments/alignments_c/`, 1,300 JSONs); structural token IDs
  {0,1,2,3,4,5,21,51,52,53}.
- Waveforms: `dataset/raw/wavs/{uid}.wav` (fastspeech2.0 dataset), resampled
  to 16 kHz mono; RMS energy and normalized-autocorrelation voicing computed
  at mel-frame rate and interpolated to the 50 Hz SSL grid.
- Segmenter: K=32 adaptive arm retrained with the identical Experiment B/C
  recipe (DP warm start + tau anneal 0.35→0.08, seed 7). Final soft MSE
  **0.0931** vs 0.0941 in Experiment C — the retrained arm is faithful.
- Four boundary conditions per utterance at matched K=32: **learned**,
  **DP** (cached reconstruction-optimal), **uniform**, **random**
  (resampled per utterance).

## Method

Four measurements per boundary, all conditions treated identically:

1. **Spectral change** — L2 frame-difference magnitude of the PCA features,
   averaged over ±2 frames around the boundary; plus (a) the ratio to the
   same statistic measured in the surrounding segment interiors and (b) a
   z-score against the utterance's own spectral-change distribution
   (removes per-utterance baseline differences).
2. **Energy jump** — |Δ log RMS| across the boundary (SSL grid).
3. **Silence proximity** — distance in SSL frames to the nearest frame below
   the utterance's 20th-percentile energy.
4. **Phone-edge hit rate** — fraction of boundaries with a MAS phone edge
   within ±tol frames, swept over tol ∈ {0, 1, 2, 3, 5, 8, 13}. The sweep
   replaces C's fixed ±2 because a 40 ms window around ~87 phone edges per
   371-frame utterance covers most of the timeline (C already showed random
   scores 0.40 F1 at ±2; the hit-rate version saturates harder, e.g. random
   reaches 0.79 by ±2). Structural-boundary fraction is kept at the fixed
   ±2 tolerance.
5. **Displacement response** — each boundary moved by ±{2,4,8,16} frames
   within its segment, recomputing statistics at the moved position.
   Run **symmetrically** for learned and DP boundaries: identical response
   curves mean identical mechanisms; divergence is the novelty probe.

## Results

### Boundary statistics by condition (means over valid boundaries)

| condition | spectral Δ (raw) | spectral Δ (z) | energy jump | silence prox (frames) | structural frac (±2) |
|---|---:|---:|---:|---:|---:|
| learned | 4.394 | **+0.135** | 0.600 | 10.16 | 0.161 |
| DP | 4.225 | −0.008 | 0.609 | 9.86 | 0.158 |
| uniform | 4.234 | −0.000 | 0.616 | 10.25 | 0.159 |
| random | 4.284 | +0.042 | 0.621 | 9.87 | 0.155 |

### Phone-edge hit rate by tolerance

| condition | ±0 | ±1 | ±2 | ±3 | ±5 | ±8 | ±13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned | 0.305 | 0.670 | 0.829 | 0.903 | 0.964 | 0.989 | 0.997 |
| DP | 0.386 | 0.739 | 0.842 | 0.900 | 0.956 | 0.985 | 0.997 |
| uniform | 0.271 | 0.623 | 0.800 | 0.885 | 0.944 | 0.983 | 0.998 |
| random | 0.262 | 0.629 | 0.795 | 0.875 | 0.945 | 0.981 | 0.996 |

### Displacement response — spectral change at the moved position (mean |Δ| from the boundary's own value)

| |Δ| | learned | DP |
|---|---:|---:|
| 2 | 0.376 | 0.408 |
| 4 | 0.555 | 0.567 |
| 8 | 0.601 | 0.578 |
| 16 | 0.616 | 0.583 |

The energy-jump response is flat in both conditions (≈0.58–0.61 at every
displacement; the log-RMS difference across a moved boundary is dominated by
ordinary frame-to-frame energy fluctuation, so this statistic does not
resolve boundary quality at these scales). Phone-flip counts per displacement
are also near-identical (learned 154/233/227/221 vs DP 133/238/214/199 for
|Δ|=2/4/8/16).

## Findings

1. **Learned ≈ DP on every acoustic statistic — the pre-registered null.**
   Phone-edge hit rates: DP leads at ±0/±1/±2 (by 8.1/6.9/1.3 points);
   beyond that learned is marginally ahead at ±3/±5/±8 (≤0.9 points) and the
   two tie at ±13 (±0.1). Energy jump, silence
   proximity, and structural fraction are all within noise of the DP oracle.
   The single statistic that separates the conditions is spectral-change z
   (Finding 2). Combined with C's statistical tie on F1, there is no measured
   property on which the learned placement departs meaningfully from
   reconstruction-optimal placement. **Branch 1 of the interpretation
   applies: the mechanism is acoustic change-point discovery.**
2. **The one signal that separates learned from everything else is
   spectral-change z: +0.135 vs ≤ +0.042 for all controls.** Learned
   boundaries sit on local maxima of frame-to-frame feature change *after*
   removing each utterance's own baseline. Notably, DP does not show this
   (+0.008 below its baseline): DP minimizes within-segment SSE globally and
   is free to place an edge one or two frames off the steepest step. Learned
   boundaries sit slightly closer to large spectral steps, yet score
   *lower* phone-edge hit rates at ±0/±1 (0.305/0.670 vs 0.386/0.739) —
   under the matched-K comparison (matched count, not matched MSE). This is
   consistent with the model preferring the largest available steps, which
   need not coincide with the dense phone-edge field. We flag this as a
   **weak-mechanism observation** (single seed, no significance test) rather
   than a finding.
3. **Displacement responses are indistinguishable.** Both conditions show
   the same saturating curve: moving a boundary 2 frames costs ~0.38–0.41
   spectral change, 16 frames costs ~0.58–0.62. No divergence at any
   displacement; the learned solution occupies the same local geometry as
   the DP solution.
4. **Silence is not the driver.** Learned boundaries are no closer to
   low-energy frames than uniform (10.16 vs 10.25) and barely closer than
   random (9.87). Structural (silence/pause-adjacent) boundaries are ~16%
   in all conditions, matching C's content-vs-all edge gap.

## Verdict

**Mechanism identified: reconstruction-driven segmentation is acoustic
change-point discovery, with no learned-specific property beyond a weak
preference for large spectral steps.** This closes the boundary-forensics
program the review proposed: learned boundaries carry no information about
phones, energy, silence, or temporal position that the pure
reconstruction-optimal DP does not. There is nothing left to extract from
comparing boundary *placements* — the next unit of scientific progress must
come from a different object (multi-scale hierarchy, mask/content analysis,
or a representation-class comparison), not from more boundary statistics.

## Caveats

- Hit rates here are **not** F1 and are not comparable to C's table: with
  1,488 placed boundaries vs ~87 reference edges per utterance, each edge
  can be claimed multiple times, so even random placement scores 0.79 at ±2.
  The sweep is a saturation diagnostic, not an alignment score; C's greedy
  1-1 F1 remains the alignment estimate of record.
- The learned-vs-DP comparison is matched in K but not in achieved
  reconstruction error (Experiment B: hard MSE 0.315 learned vs 0.285 DP at
  K=32; D's retrained arm lands at ≈0.331, consistent with B's learned
  range). Small placement differences are therefore expected; none of the
  measured statistics exceed noise.
- Single seed, 48 utterances, one language (Nepali), MAS-derived reference
  edges (the model's own learned durations, not human labels).
- The voicing statistic was computed but not aggregated (all conditions
  read from the same utterance-level voicing track; between-condition voicing
  contrast requires segment-level attribution, deferred to the mask-analysis
  direction).

## Artifacts

- `experiments/experiment_d.py` — analyzer, tolerance sweep, symmetric
  displacement-response protocol
- `experiments/results_experiment_d.json` — full per-condition aggregates
  (including both tolerance sweeps and displacement-response aggregates by
  condition) and per-utterance rows (48); raw per-perturbation records are
  not persisted, only their aggregates
- Reuses: `experiments/features_wavlm_l6/`, `experiments/alignments_c/`,
  `dataset/raw/wavs/` (fastspeech2.0)
