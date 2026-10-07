# Implementation Status Report

Tracks the review-recommended improvements. Updated after the verification pass
(2026-10-07) and the second-round review fixes (same day).

## Round 5: the variable-rate control (decisive)

The second-round review identified the cheapest missing control: **frame-VQ +
run-length coding** (`experiments/experiment_frame_rle.py`, run,
`results_frame_rle.json`). Identical frame models and reconstruction; only the rate
changes, via ideal two-part enumerative accounting of (code, run-length) pairs.

**Result: the control erases the Round-4 low-rate segment advantage.** RLE-frame
reaches 0.337 at 57.8 b/s (vs billed adaptive's best 0.348 at 55.1 b/s) and 0.290 at
75.1 b/s (vs adaptive 0.298 at 78.6 b/s); RLE better 3/3 seeds at 50–58 and ~110 b/s,
parity at ~78 b/s. Under billed timing + matched rates + a variable-rate control,
**learned adaptive segmentation provides no rate-distortion advantage over per-frame
coding in this setting.** Paper §4.14 added; abstract/§1.2/§2.4/§5/§6 and a Round-5
revision-log entry synced.

Also fixed per the second-round review:
- §2.4 contradiction removed ("segmentation does not win on RD" → no advantage
  survives the controls)
- "from ~116 b/s upward" scoped to tested matched pairs (116–132 b/s) everywhere
- monotonicity proof rewritten in discrete form (no continuous IVT)
- "warp realizes far fewer partitions" claim replaced with airtight
  enumerative-description wording
- AUC sentence removed from §4.10; seed-level ΔD interval column added to the
  matched-pairs table
- "equal capacity" replaced with "matched decoder architecture and latent
  dimensionality per transmitted token"
- Free-Channel **Principle** renamed Free-Channel **Audit** (with framing sentence)
- "none of them claims" softened to "not directly tested by these systems"
- abstract tightened; Figures 1 and 2 generated and embedded
  (`experiments/make_figures.py` → `figures/fig1_rd_crossover.png`,
  `figures/fig2_timing_ablation.png`)

## Verified status

### P0 (Absolutely Necessary)

1. **True matched-rate RD curves** — COMPLETE (run + analyzed)
   - `experiment_matched_rate.py` / `experiment_matched_rate_multiseed.py` exist and
     reproduce the B2/F recipe exactly (verified against `experiment_b_vq.py`).
   - Verification fix: default `--ks` extended from `16,32` to `16,32,64`. With only
     K=16/32 the segment arms span ~19–71 b/s vs. frame 50–300 b/s, leaving a single
     frame anchor (N=2, 50 b/s) in the "overlap" — the exact defect the experiment
     exists to fix. K=64 spans ~58–132 b/s and interleaves with frame N=3..6
     (79/100/116/129 b/s), giving a genuine 50–132 b/s matched-rate window.
   - **Run completed: 189/189 points, 3 seeds, summary statistics written.**
   - **RESULT (§4.10 of the paper): the sweep REVERSES the interim verdict.** At
     matched rates, billed adaptive beats frame 3/3 seeds at 49.8/55.1/78.6 b/s
     (0.357 vs 0.396; 0.348 vs 0.396; 0.298 vs 0.340), parity at ~100 b/s, frame
     wins from ~116 b/s (3/3 seeds) and is the only arm measured at 150–300 b/s.
     Even uniform K=64 wins the lowest pair. Free timing was worth ~2.7× the honest
     advantage. The paper's headline is now a measured crossover, not "no
     advantage" — see the Round 4 revision-log entry.

2. **3+ seeds for RD experiments** — COMPLETE (seeds 7, 42, 123; per-config mean ± std
   and per-seed paired win counts in `results_matched_rate_multiseed.json`)

3. **Fix wording around "every rate"** — COMPLETE (verification round)
   - First pass had softened the abstract but left "wins at every tested rate" /
     "loses everywhere" in §4.7, §4.8, §5.1, §5.2, §6 — and the paper's own data
     contradicts those claims at the low end: billed adaptive (0.350–0.357 at 44–55 b/s)
     beats the lowest measured frame point (0.399 at 50 b/s WavLM; 0.53 at 86 b/s mel),
     while frame at 79 b/s is better — the curves cross inside the unmeasured window.
   - All live claims now state "no advantage at any measured point" + explicit
     pointer to §4.10 for the matched-rate resolution.
   - "Bits are exact" in §4.3 contradicted the new §3.5 ideal-code-length caveat;
     now reads "ideal combinatorial code lengths (§3.5)".
   - All 7 remaining "honest" occurrences removed; "learned the hard way",
     "what survives" neutralized.

### P1 (Very High Value)

4. **Soft/hard × timing ablation** — COMPLETE (run)
   - Verification fixes: (a) broken `results_experiment_b.json` reference loader
     (read a nonexistent `points` key; now parses the `arms` structure); (b) `hard_mass`
     computed `argmax(masks, dim=-1)` over the TIME axis of [B,K,T] — invalid; now
     indexes the segment axis; (c) added hard partition-mean MSE per trained model so
     the full 2x3 matrix comes from the same runs; (d) numpy-float JSON stringification.
   - Results (`results_soft_hard_timing_ablation.json`, K=32, N=256, seed 7):
     soft 0.181 (free) / 0.350 (none) / 0.343 (billed); hard-at-own-boundaries
     0.336–0.341; exp-B hard references 0.316–0.321 (uniform/learned), 0.290 (DP).
     **Conclusion: D(soft, no timing) ≈ D(hard, no timing) — the soft advantage was
     the timing leak.** Written up as §4.13.

5. **Duration residualization by phone class** — COMPLETE (run)
   - Verification finding: the original script was an empty placeholder (returned {},
     wrote `None`s). Now fully implemented: `experiment_e.py --dump-records` exports the
     exact per-segment records behind the published R^2 (re-run reproduces 0.7043 vs
     published 0.7037), and `analyze_duration_phone_class.py` reuses experiment E's
     CV-ridge protocol with three additions: phone-only R^2, leakage-free additive
     (phone + z) R^2, and the norm control, for all four conditions.
   - Results (`analysis_duration_phone_class.json`): phone-only 0.082; additive 0.608;
     **incremental +0.53 beyond phone class** — the R^2=0.72 coupling is NOT
     phone-class-confounded. DP contrast: full 0.139 with norm control 0.436 (mostly
     geometric artifact). Written up as §4.12.

6. **TOST equivalence testing** — COMPLETE (run)
   - Verification finding: the original script fell back to hardcoded PLACEHOLDER
     arrays and would have written fake conclusions to disk; it also used
     independent-samples tests on paired data. Rewritten: paired bootstrap + paired
     parametric TOST on the real 48 per-utterance F1 values in
     `results_experiment_c.json`; fixed a p-value bug (2*t.sf(-|t|) -> 2*t.sf(|t|)).
   - Results (`analysis_equivalence_tost.json`): learned vs DP diff −0.0049 (SE 0.0051);
     NOT equivalent at δ=0.01 (p=0.16), equivalent at δ=0.02 (p=0.002); K=16 identical
     pattern; control learned-vs-uniform (+0.0148) correctly not equivalent. The paper
     now quotes margins (abstract, §4.11, §5.1-4, §5.4-4).

### P2 / P3 — NOT IMPLEMENTED (unchanged)
k-means/residual VQ, latent-dim sweep 16/32/64/128, chance-corrected boundary metric,
second language/speaker, perceptual/downstream evaluation, prior-tokenizer head-to-heads.

## New paper sections (this round)

- §4.10 Matched-rate RD with multiple seeds — **written with full results; headline
  revised to the measured crossover** (segment wins 50–79 b/s 3/3 seeds, parity ~100,
  frame wins ≥116)
- §4.11 TOST equivalence (equivalent within δ=0.02 F1; not δ=0.01)
- §4.12 Duration-vs-phone-class residualization (coupling survives: +0.53 R² beyond phone)
- §4.13 Soft–hard discrepancy = the timing leak (D(soft, no timing) ≈ D(hard, no timing))
- Revision log Round 4 documents the reversal; abstract/§1.2/§4.3/§4.7/§4.8/§4.9/
  §5.1/§5.2/§6 all synced to the crossover result

## How to reproduce

```bash
PY=C:/Users/Lenovo/Desktop/fastspeech2.0/venv/Scripts/python.exe   # broken system torch otherwise

# matched-rate multiseed (running / rerun)
$PY experiments/experiment_matched_rate_multiseed.py --seeds 7,42,123
# analysis (no extrapolation; per-seed paired stats; overlap-window only)
$PY experiments/analyze_matched_rate.py
# soft/hard x timing matrix
$PY experiments/experiment_soft_hard_timing_ablation.py --k 32 --codebook 256
# TOST on real per-utterance F1
$PY experiments/analyze_equivalence_tost.py
# duration residualization on the exact published-probe records
$PY experiments/experiment_e.py --seed 7 --output experiments/results_experiment_e_records_recheck.json --dump-records experiments/segment_records_e_seed7.npz
$PY experiments/analyze_duration_phone_class.py --reference-r2 0.7037
```

## Remaining before submission

1. Mel matched-rate + RLE sweep (the mel low-rate window is still unmeasured).
2. Segment arm with entropy-coded boundaries (would the segment family recover parity
   against RLE by exploiting the non-uniform partition prior?) — the one caveat
   §4.14 leaves open.
3. Optional P2 (latent-dim sweep is the highest-value remaining mechanistic test).
4. Research-diary tone: further revision narrative could move to Appendix C if the
   venue enforces a strict page budget.
