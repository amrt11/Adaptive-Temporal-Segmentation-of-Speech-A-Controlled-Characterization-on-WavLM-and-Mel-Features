# Experiment C — Do Learned Boundaries Correspond to Phoneme Structure?

Status: **complete** (tolerance sweep ±2/±1/±0 SSL frames; 3 seeds for the K=32
arm; paired bootstrap over 48 test utterances).

## Question

After Experiments B/B2 falsified the compression/TTS framing, the pre-registered
question was:

> Do the boundaries chosen by the adaptive segmenter (trained purely for
> reconstruction, no phonetic supervision) correspond to phoneme structure, as
> revealed by MAS phone alignments from the trained v6 FastSpeech2?

Decision rule: beating **random** placement = alignment above chance (structure
exists); beating **uniform** = placement tracks phone structure specifically,
beyond duration-distribution effects. Structure evidence does *not* revive the
compression/TTS claim (B/B2 verdicts stand).

## Data

- Reference: MAS durations from `checkpoints_old/staged_4077_rtx2050_v6/latest.pt`
  (step 19802), extracted batch-wise via the project inference stack
  (`extract_alignments_c.py`); 1,300/1,300 utterances, 0 failures, per-utterance
  check sum(durations) == valid mel length. Alignments in `experiments/alignments_c/`.
- Features: cached WavLM L6 (Experiment B), PCA-64, test split first 48 utterances.
- Segmenters: adaptive arms retrained with the identical Experiment B recipe
  (DP warm start 300 steps + tau anneal 0.35→0.08 over 1,700 steps, seed 7),
  K=16 and K=32. Final soft MSE 0.1333 / 0.0941 — matches Experiment B (0.130 /
  0.090), so the retrained arms are faithful.
- Reference edges mapped mel→SSL at 0.5805 SSL frames per mel frame. Structural
  token IDs {0,1,2,3,4,5,21,51,52,53} excluded in the content-only variant.
- Metric: greedy 1-1 matching F1 within ±tol SSL frames, mean distance to
  nearest reference edge.

## Results — tolerance sweep (F1 vs all MAS phone edges)

| condition | ±2 (40 ms) | ±1 (20 ms) | ±0 | mean dist (frames) |
|---|---:|---:|---:|---:|
| learned K=16 | 0.237 | 0.179 | 0.073 | 1.84 |
| learned K=16 (vs content edges) | — | 0.186 | 0.074 | 2.05 |
| learned K=32 | 0.442 | 0.362 | 0.165 | 1.44 |
| uniform@32 | 0.427 | 0.332 | 0.143 | 1.68 |
| random@32 | 0.399 | 0.316 | 0.139 | 1.73 |
| DP@32 (recon oracle) | 0.447 | 0.393 | 0.205 | 1.35 |
| learned K=32 (vs content edges) | 0.465 | 0.375 | 0.168 | 1.60 |

Mean ~87 phone edges per ~371-frame utterance → at ±2 the reference windows
cover ≈95% of the timeline, so even random placement scores 0.40. The sweep is
the honest view; ±1 is the tightest tolerance where mel→SSL rounding (±0.6
frame) does not distort comparisons.

## Significance (paired bootstrap, 10k resamples, tol ±2, seed 7)

| comparison | ΔF1 | 95% CI | p | wins |
|---|---:|---|---:|---:|
| learned K=32 vs uniform | +0.0148 | [+0.005, +0.025] | 0.004 | 27/48 |
| learned K=32 vs random | +0.0435 | [+0.030, +0.056] | <0.001 | 39/48 |
| learned K=32 vs DP@32 | −0.0049 | [−0.015, +0.005] | 0.32 | 17/48 |
| learned K=16 vs random | −0.1611 | [−0.173, −0.149] | <0.001 | 0/48 |
| learned K=16 vs uniform | −0.1898 | [−0.202, −0.178] | <0.001 | 0/48 |

Seed robustness (K=32, tol ±2): 0.442 / 0.447 / 0.444 for seeds 7 / 3 / 11
(mean 0.444, sd 0.002); uniform and DP controls are deterministic.

## Findings

1. **K=32 learned boundaries carry weak but real phone-structure signal.**
   They beat random decisively (+0.044 F1, 39/48 utterances, p<0.001) and beat
   uniform significantly though narrowly (+0.015, p=0.004). The uniform-vs-random
   gap (0.427 vs 0.399) shows most achievable alignment comes from phone
   durations being more regular than random; the learned model adds only ~⅓ of
   the remaining gap on top of uniform.
2. **K=16 learned boundaries carry no signal.** They lose to *random* 0/48
   (−0.16 F1) — the model actively avoids the dense phone-edge field, which
   reconstruction at low K does not need. Any structure claim must be qualified
   as K-dependent, not a general property of the segmenter.
3. **Learned ≈ DP oracle.** At K=32 the learned arm statistically ties the
   reconstruction-optimal DP (−0.005, p=0.32): boundary placement is explained
   by acoustic homogeneity, not by anything extra that would indicate phonetic
   awareness beyond homogeneity. DP@32 scores higher at ±1/±0, i.e. its edges
   sit slightly closer to phone edges.
4. **Content-only edges score higher** (0.465 vs 0.442 at K=32): some learned
   boundaries coincide with silence/pause edges, which are acoustically obvious
   rather than phonetic.

## Verdict

**Qualified yes: weak, K-dependent structure — and nothing beyond acoustic
homogeneity.** The pre-registered interpretation holds: boundaries reflect
low-level acoustic organization with a modest tendency to coincide with phone
transitions at K≈32. This is *structure, not compression* — consistent with
B/B2: frame-VQ beats segmental codes at matched bits, and the learned placement
is statistically indistinguishable from the pure reconstruction-optimal DP.

## Caveats

- Single checkpoint, one language (Nepali), 48 test utterances; seed variation
  quantified only for the K=32 arm (negligible: sd 0.002).
- MAS alignments are the model's own learned durations, not human labels; the
  experiment validates the segmenter against a second learned structure, not
  against ground truth.
- ±1 frame (20 ms) is the floor for meaningful comparisons (mel→SSL rounding);
  mean distances of 1.4–1.8 SSL frames (~28–36 ms) sit at the edge of phone-
  transition acoustics.
- Random control is resampled per utterance; its F1 varies ±0.01 across seeds
  (0.388–0.399) — does not affect any conclusion.

## Artifacts

- `experiments/extract_alignments_c.py`, `experiments/alignments_c/` (1,300 JSONs)
- `experiments/experiment_c.py` → `results_experiment_c.json` (tol 2, per-utterance
  rows), `results_experiment_c_tol1.json`, `results_experiment_c_tol0.json`,
  `results_experiment_c_s3.json`, `results_experiment_c_s11.json`
- `experiments/bootstrap_c.py` — paired bootstrap + seed replicate summary
- Run logs: `experiments/experiment_c_run*.log`
