# Paper Updates Summary

This document summarizes all changes made to the paper based on the review feedback.

## Second round (verification pass, 2026-10-07)

A full verification of the first-round changes found and fixed:

1. **"Every tested rate" claims survived in §4.7/§4.8/§5.1/§5.2/§6** and were
   contradicted by the paper's own data at the low end (billed adaptive 0.350–0.357 at
   44–55 b/s beats the lowest measured frame point 0.399 @ 50 b/s WavLM / 0.53 @ 86 b/s
   mel; frame @ 79 b/s is better — the curves cross in the unmeasured window). All live
   claims now say "no advantage at any measured point" and defer the low-rate region
   to §4.10's matched-rate sweep.
2. **"Bits are exact" (§4.3)** contradicted the new §3.5 ideal-code-length caveat —
   aligned.
3. **7 "honest" occurrences, "learned the hard way", "what survives"** remained — removed.
4. **New results sections** added: §4.11 (paired TOST: equivalent within δ=0.02 F1,
   not δ=0.01), §4.12 (duration coupling survives phone-class residualization,
   incremental ≈ +0.53 R²; DP's signal is mostly the geometric artifact), §4.13
   (soft–hard discrepancy resolved: D(soft, no timing) ≈ D(hard, no timing) — the
   soft advantage was the timing leak). §4.10 (matched-rate multi-seed RD) added with
   results pending the running sweep.
5. **Analysis scripts were placeholders or broken**: TOST used hardcoded fake data,
   duration analysis was an empty skeleton, the ablation's experiment-B reference
   loader read a nonexistent key, `hard_mass` indexed the wrong tensor axis, and the
   matched-rate analysis had a NameError in its multi-seed branch and extrapolated
   RD curves outside their measured ranges. All rewritten; see
   `docs/implementation_status.md` for the run results.
6. **The matched-rate sweep (189 runs, 3 seeds) REVERSED the interim headline.** At
   genuinely matched bitrates, billed segment-VQ beats frame-VQ 3/3 seeds at 49.8–78.6 b/s
   (by 10–12%), ties at ~100 b/s, and frame wins from ~116 b/s. The prior "billed segment
   loses at every measured point" was itself an artifact of unmatched rate ranges.
   §4.10 was written with the full results; the abstract, §1.2, §4.3, §4.7, §4.8,
   §4.9, §5.1, §5.2, §6 and a new Round-4 revision-log entry were all synced to the
   measured crossover.

## Changes to `negative_result_paper.md`

### 1. Abstract (Line 5)
**Before**: "segment-VQ loses to frame-VQ at every tested rate"
**After**: "we find no rate-distortion advantage for segment-VQ over frame-VQ across the tested bitrate range"

**Rationale**: The current experiments don't have genuinely matched bitrates, so "every tested rate" is misleading. The new wording is more precise and defensible.

### 2. Section 1.2 - Stage 2 (Line 25)
**Before**: "frame-VQ wins at every overlapping rate on both representations"
**After**: "the corrected experiments provide no evidence that segment-VQ outperforms frame-VQ under billed timing"

**Rationale**: More precise claim that doesn't overstate the evidence given the rate mismatch issue.

### 3. Section 1.3 - "honest operating envelope" (Line 5)
**Before**: "Three findings define the architecture's honest operating envelope"
**After**: "Three findings define the architecture's operating envelope under controlled accounting"

**Rationale**: Removed defensive "honest" terminology for more neutral academic language.

### 4. Section 2.3 - Updated Related Work (Lines 45-50)
**Added**:
- CodecSlime reference [14]
- "Interpreting and Evaluating Dynamic-Rate Speech Codec Boundaries" reference [15]

**Rationale**: The literature has moved quickly; these references acknowledge recent work on dynamic-rate speech tokenization and boundary interpretation, making the paper's positioning more accurate.

### 5. Section 2.4 - New Section 2.5: Free-Channel Principle (Lines 51-70)
**Added**: A new subsection "The Free-Channel Principle for Compression Evaluation"

**Content**:
- Formal principle statement
- Mathematical formulation (X, Y, D(Y))
- Rate decomposition: R_total = R_token + R_boundary + R_timing + R_other
- Broad applicability to learned compression, neural codecs, variable-rate representations
- Contrast with VARSTok's explicit duration coding

**Rationale**: Elevates the methodological contribution from a specific finding to a general evaluation principle applicable beyond speech segmentation.

### 6. Section 3.2 - Added Proposition (Lines 90-101)
**Added**: Formal proposition with proof for monotonic partitions

**Content**:
- Proposition: If ρ_t > 0, then C(t) is strictly monotonic with unique inverse boundaries
- 3-line proof using intermediate value theorem

**Rationale**: Makes the mathematical contribution cleaner and more rigorous, as suggested in the review.

### 7. Section 3.5 - Distinguished Three Notions of Rate (Lines 122-134)
**Added**:
- Representation rate R_token
- Boundary rate R_boundary
- Side-channel rate R_timing
- Total rate decomposition
- Requirement for honest codec: R_decoder-only feature information = 0
- Note on accounting precision (ideal vs actual transmitted bits)

**Rationale**: Formalizes the rate accounting and acknowledges the distinction between ideal combinatorial code length and actual transmitted bits.

### 8. Section 4.3 - Updated Wording (Line 216)
**Before**: "With billing, the segment arm loses at every tested rate"
**After**: "With billing, the corrected experiments provide no evidence that segment-VQ outperforms frame-VQ at the tested bitrates"

**Rationale**: More precise claim about the current evidence rather than overgeneralizing.

### 9. Section 4.8 - Updated Wording (Line 431)
**Before**: "loses to frame at every rate"
**After**: "loses to frame at all tested rates"

**Rationale**: Subtle change to be more precise about "all tested rates" vs "every rate" (universe of rates).

### 10. Section 4.9 - Updated Wording (Line 450)
**Before**: "the honest segment arm loses to frame-VQ at every overlapping rate"
**After**: "under boundary-derived timing, the segment arm does not outperform the frame baseline"

**Rationale**: Removed defensive "honest" language and made the claim more methodologically precise.

### 11. Removed Defensive Language Throughout
**Changed**:
- "honest caveat" → "important caveat"
- "honestly stated" → removed or replaced with neutral phrasing
- "Scope, honestly stated" → "Scope"
- "honest within-family statement" → "within-family statement under controlled accounting"

**Rationale**: The review noted the paper felt too defensive, as if arguing with a reviewer who isn't present. More neutral academic language is preferred.

### 12. Section 4.5 - Updated Placement Claim (Line 465)
**Before**: ""Learned ≈ DP" is retracted as a placement claim; what survives is metric-level equivalence of genuinely distinct partitions"
**After**: "The displacement analysis does not support placement equivalence; the two partitions differ by approximately 10.5 frames, though they remain metric-level equivalent on saturated alignment metrics"

**Rationale**: More precise statement that doesn't use "retracted" language, instead directly stating what the analysis shows.

### 13. Section 4.6 - Updated Wording (Line 463)
**Before**: "The honest headline: apparent low-rate segment advantages in this architecture are accounting artifacts"
**After**: "The headline: apparent low-rate segment advantages in this architecture are accounting artifacts"

**Rationale**: Removed "honest" qualifier.

## New Experiment Scripts

### 1. `experiments/experiment_matched_rate.py`
Denser codebook sweeps for genuinely overlapping rate-distortion curves.

### 2. `experiments/experiment_matched_rate_multiseed.py`
Multi-seed version with statistical confidence intervals.

### 3. `experiments/analyze_matched_rate.py`
Analysis tool for interpolating and comparing RD curves at matched bitrates.

### 4. `experiments/experiment_soft_hard_timing_ablation.py`
Soft/hard × timing ablation matrix to investigate the 0.09 vs 0.32 MSE discrepancy.

### 5. `experiments/analyze_duration_phone_class.py`
Framework for duration residualization by phone class analysis.

### 6. `experiments/analyze_equivalence_tost.py`
TOST equivalence testing to replace non-significant difference tests.

## Key Principles Added

### The Free-Channel Principle
"When evaluating compressed representations, every decoder input that is a function of the original input must either be transmitted, derived solely from transmitted information, or explicitly excluded from the claimed compression pathway."

This elevates the paper's contribution from a specific finding about speech segmentation to a general methodological principle applicable to:
- Learned compression systems
- Neural codecs
- Variable-rate representations
- Tokenizers and quantized representations
- Retrieval/compression systems

## Next Steps for the Authors

1. **Run the matched-rate multi-seed experiment** to get data for the updated RD claims
2. **Run the soft/hard timing ablation** to explain the 0.09 vs 0.32 discrepancy
3. **Extract segment vectors** and run the duration residualization analysis
4. **Extract per-utterance F1 scores** and run the TOST equivalence testing
5. **Update the paper figures** with the new matched-rate RD curves
6. **Consider updating the paper structure** as suggested in the review (elevation of evaluation protocol to its own section)

## Publication Readiness Assessment

Based on the review's original assessment (~70-75%), these improvements address the most critical issues:

- **Rate-distortion claim**: Will be strengthened once matched-rate experiments are run
- **Novelty**: Better positioned with the Free-Channel Principle and updated related work
- **Writing**: Less defensive, more precise claims
- **Statistical rigor**: Will be improved with multi-seed experiments and TOST equivalence testing

The paper is moving toward a stronger negative-results/methodological contribution that focuses on evaluation methodology rather than specific performance claims.
