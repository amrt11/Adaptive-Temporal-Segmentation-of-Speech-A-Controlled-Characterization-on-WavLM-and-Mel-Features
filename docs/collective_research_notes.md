# Collective Research Notes

## Topic

**Mask- and segmentation-inspired representations for speech and TTS**

This document consolidates the discussions in:

- `image_segmentation.md`
- `fastspeech_as_segmentation.md`
- `tts_as_segmentation.md`
- `paper_study.md`

The central idea is not to transplant Mask2Former or SAM directly into TTS. The more useful research question is:

> Can learned, ordered temporal masks and set/query-based latent units provide a more expressive and rate-efficient representation of speech than conventional token-to-frame alignment, while retaining enough structure for generation?

The initial claim should remain narrow: **adaptive acoustic segmentation may improve rate-distortion efficiency**. Whether the learned segments correspond to phonemes, syllables, words, or prosodic units is a later empirical question.

---

## 1. Segmentation concepts to transfer

### 1.1 Semantic, instance, and panoptic segmentation

Segmentation models operate at different conceptual levels:

| Model/family | Main role | Typical output |
|---|---|---|
| CNN | Architecture family/backbone | Can support classification, detection, or segmentation with a suitable head |
| Mask R-CNN | Instance segmentation | Object class, bounding box, and one mask per detected object |
| MaskFormer | Mask classification formulation | A set of masks and labels; supports semantic, instance, and panoptic tasks |
| Mask2Former | Improved universal mask classification | Query-based masks with masked attention; supports semantic, instance, and panoptic tasks |
| SAM 1 | Promptable object-mask prediction | Image + point/box/mask prompt → object mask |
| SAM 2 | Promptable image/video segmentation | Prompted masks propagated through video |
| SAM 3 | Concept/text-prompted segmentation | Text or concept prompt → detected and segmented objects |

Important distinctions:

- **CNN is not a segmentation task.** It is a broad architecture family.
- **Mask R-CNN is primarily detection plus instance masks.**
- **MaskFormer and Mask2Former predict a set of masks and classes**, rather than only a dense per-pixel class map.
- **SAM is promptable segmentation**, not originally conventional closed-set semantic segmentation.

### 1.2 The architectural progression

The useful conceptual progression is:

```text
dense pixel prediction
    → CNN encoder/decoder (FCN, U-Net, DeepLab)
    → object detection plus masks (Mask R-CNN)
    → set prediction of masks (MaskFormer)
    → masked attention and universal segmentation (Mask2Former)
    → promptable segmentation (SAM/SAM 2)
    → concept-level/open-vocabulary segmentation (SAM 3)
```

The transferable abstractions are:

1. **Set prediction:** represent a scene as a collection of regions and descriptors.
2. **Queries:** learned queries can represent objects/events/units.
3. **Masks:** a query can select the temporal region relevant to its latent representation.
4. **Masked attention:** a query can attend mainly to its predicted region.
5. **Promptability:** prompts can select or control a desired factor.

---

## 2. Mapping segmentation to speech

An image is spatially indexed by (H \times W); speech can be represented as a time-frequency array (T \times F), or more simply as a temporal sequence of length (T).

Image segmentation:

$$
M \in \{0,1\}^{H\times W}
$$

Possible speech analogues:

$$
M \in \{0,1\}^{T\times F}
\qquad\text{or}\qquad
M \in \{0,1\}^{T}.
$$

Temporal masks could describe:

- phoneme-like regions;
- syllables;
- words or multiword units;
- prosodic phrases and gestures;
- vowels, consonants, transitions, and silence;
- pauses, breaths, and hesitations;
- speaker transitions or other acoustic events.

Speech differs from images in two important ways:

1. It is fundamentally sequential, so mask order and monotonicity matter.
2. Coarticulation means boundaries are not always sharp. Soft or overlapping masks may be more appropriate than perfectly disjoint object masks.

---

## 3. Two related formulations

### 3.1 Mask-based latent speech events

Let an utterance representation be (X \in \mathbb{R}^{T\times F}). A set of (K) learned queries produces temporal masks and latent vectors:

$$
\{(M_k,Z_k)\}_{k=1}^{K},
\qquad M_k(t)\in[0,1].
$$

The frame representation can be composed as:

$$
H_t = \sum_{k=1}^{K} M_k(t) Z_k,
$$

followed by a decoder:

$$
\hat X = D(H_{1:T}).
$$

For a generative model, the corresponding high-level path is:

```text
text / prompts
    → linguistic or latent queries
    → temporal masks and latent speech events
    → mask-weighted composition
    → acoustic decoder
    → mel-spectrogram or waveform
```

The masks should initially remain **unsupervised latent regions**. Do not impose “one mask equals one phoneme” before testing what the representation naturally discovers.

### 3.2 Segmentation as a relaxation of alignment

FastSpeech-style TTS uses:

$$
\text{text}
\rightarrow \text{encoder}
\rightarrow \text{duration}
\rightarrow \text{length regulator}
\rightarrow \text{decoder}
\rightarrow \text{mel}.
$$

The length regulator repeats a token representation (h_i) for its predicted duration (d_i):

$$
h_i \rightarrow \underbrace{h_i,h_i,\ldots,h_i}_{d_i}.
$$

This is a restricted form of temporal assignment. A general mask formulation replaces one hard duration per text token with a learned assignment (A_{t,i}) or a set of temporal masks. MAS and forced alignment can be viewed as special cases: hard, monotonic, one-hot masks tied to text tokens.

This gives a principled comparison:

$$
\text{forced alignment/MAS}
\quad vs \quad
\text{learned temporal masks}
\quad vs \quad
\text{implicit alignment}.
$$

The learned representation might capture structures that do not map to one token, such as syllables, prosodic phrases, coarticulatory transitions, or non-text events.

---

## 4. Promptable and compositional speech

SAM suggests the abstraction:

$$
\text{representation} + \text{prompt} \rightarrow \text{mask}.
$$

Speech prompts could specify:

- speaker identity;
- emotion or style;
- speaking rate;
- prosody;
- phonetic or syllabic content;
- an acoustic reference or temporal span.

A possible speech representation model is:

$$
E(x)\rightarrow H,
\qquad
M_p=f(H,p),
$$

where (p) asks for a factor such as “vowel,” “high pitch,” “speaker A,” or “phoneme /a/.”

For TTS, a compositional version could separate:

```text
text → content representation
speaker/style prompt → speaker or style representation
prosody prompt → prosodic representation
all representations → composition → speech decoder
```

This is better described as **mask-based latent decomposition for compositional speech generation** than as “SAM for TTS.” SAM and MaskFormer are inspirations, not the contribution itself.

### Overlapping masks

For speech, it may be useful to permit:

$$
M_k(t)\in[0,1],
\qquad
\sum_k M_k(t)\neq 1,
$$

so multiple latent factors can contribute to one frame. However, this should not be confused with source separation: for a single speaker, the latent regions are temporal units and prosodic factors, not necessarily additive sound sources.

Use partition masks for sequential timing and additive channels for global factors when appropriate.

---

## 5. What existing evidence suggests

The literature review found related precedents but not the exact proposed formulation. This should be stated as “not found in the reviewed material,” not as proof that no such paper exists.

### Evidence supporting the transfer

- Query-based event prediction exists in audio, including DETR-like sound-event detection.
- Slot-based models have been used for audio decomposition and speech separation.
- Promptable audio separation shows that text, visual, and temporal prompts can select audio content.
- Unsupervised or self-distilled systems such as Sylber demonstrate that syllabic structure can be discovered without direct boundary labels.
- Variable-rate tokenizers show that adaptive temporal units can outperform fixed-rate representations at lower token counts.

### Limitations and warnings

- Visual alpha-mask semantics do not transfer directly to audio.
- In continuous speech, span-only prompts can be weak for sounds that overlap or persist.
- Set-based bottlenecks may reduce intelligibility when frame alignment is discarded.
- Factorized speech representations often need auxiliary supervision for content, prosody, timbre, or speaker identity.
- Unsupervised disentanglement is not identifiable without inductive biases and/or supervision.
- Learned masks may discover acoustic stationarity rather than linguistic units.

The strongest justified conclusion is:

> Query and mask mechanisms plausibly transfer to speech representation learning, but a clean separation of content, timing, and acoustics is open and must be demonstrated experimentally.

---

## 6. Recommended research hypothesis

### Primary hypothesis

> A learned monotonic temporal segmentation can represent speech with lower distortion at a matched number of units or matched total bitrate than a uniform temporal partition.

### Secondary questions

1. Do the boundaries correspond to phonemes, syllables, prosodic units, or only acoustic changes?
2. Can the representation reconstruct speech with fewer units than a frame-level latent?
3. Do masks and latent vectors support meaningful swapping or manipulation?
4. Can content, timing, speaker identity, and prosody be separated sufficiently for transfer or generation?
5. Does the representation improve low-resource TTS, including Nepali speech, after the representation study succeeds?

Avoid claiming initially that the model “discovers linguistic units.” A safer claim is:

> The model learns adaptive acoustic segments; their linguistic or prosodic interpretation is evaluated afterward.

---

## 7. Minimal well-posed model

Start with frozen features (F\in\mathbb{R}^{T\times H}), not raw waveforms. Suitable first choices are WavLM layer 6 or HuBERT at approximately 50 Hz, optionally PCA-reduced. These features are content-biased, so add a mel-spectrogram experiment later to test acoustic behavior.

### 7.1 Ordered monotonic partition

Independent sigmoid masks are insufficient: they do not guarantee order or a valid partition. A better parameterization predicts a positive boundary density:

$$
\rho_t = \rho_{\min}+\operatorname{softplus}(w^\top v_t),
$$

where (v) is a small local encoder of (F). Define a cumulative warped coordinate:

$$
C(t)=\frac{(K-1)\operatorname{cumsum}(\rho)_t}{\operatorname{cumsum}(\rho)_T}.
$$

The boundaries are (b_k=C^{-1}(k)), with (b_0=0) and (b_K=T). Define soft transitions in the warped coordinate and set:

$$
m_k(t)=\sigma_{k-1}(t)-\sigma_k(t),
$$

with fixed endpoints (\sigma_0=1) and (\sigma_K=0). Then:

$$
\sum_{k=1}^{K}m_k(t)=1
$$

exactly, subject to the discretization convention.

The uniform baseline is simply constant (\rho_t). The adaptive and uniform systems therefore share the same decoder and latent dimensions.

### 7.2 Segment representation

Pool from a linear projection (PF_t), not a context-rich encoder that could smuggle neighboring information into each segment:

$$
z_k=g\left(\frac{\sum_t m_k(t)PF_t}{\sum_t m_k(t)}\right),
\qquad z_k\in\mathbb{R}^{d_z}.
$$

Use a small (d_z), for example 16–32, so the model cannot ignore the segmentation.

### 7.3 Decoder input

The frame representation should include:

$$
h_t = \left[
\sum_k m_k(t)z_k\ ;\ u(t)\ ;\ \log d_k(t)
\right],
$$

where (u(t)) is within-segment phase and (d_k(t)) is the active segment length. Without phase and duration, the composition is nearly piecewise constant and cannot reconstruct intra-segment dynamics. However, phase can become a hidden timing channel, so it requires ablation and manipulation tests.

### 7.4 Initial objective

Start without VQ:

$$
\mathcal{L}=\|F-\hat F\|^2+\lambda_R R.
$$

Anneal the mask temperature (\tau) from smooth to sharp. An entropy/sharpness penalty can be added if needed. If using a rate claim, keep (K), (d_z), and quantization/noise assumptions fixed, or use a differentiable entropy model. VQ should be postponed until the segmentation mechanism has been isolated.

Later variants may add:

$$
\mathcal{L}_{VQ}(z),
\qquad
\lambda_H H(m),
$$

but not in the first sanity experiment.

### 7.5 Boundary-corrected Stage-1 specification

The naïve telescoping mask deserves an explicit correction. If (b_0=0) and (b_K=T), then

$$
\sigma((t-b_0)/\tau)-\sigma((t-b_K)/\tau)
$$

is approximately (0.5) at the sequence edges, so the decoder receives boundary leakage. The density-warp parameterization above fixes this by defining the endpoint transitions explicitly:

```text
ρ_t = ρ_min + softplus(w·v_t)        v = small encoder on F
C(t) = (K−1)·cumsum(ρ)_t / cumsum(ρ)_T      C: [0,T] → [0,K−1]
```

Boundaries are (b_k=C^{-1}(k)), monotonic by construction, with (b_0=0) and (b_K=T) automatically. Define (\sigma_0:=1), (\sigma_K:=0), and (\sigma_k=\sigma((C(t)-k)/\tau)) for (k=1,\ldots,K-1):

```text
m_k(t) = σ_{k-1}(t) − σ_k(t)      ⇒      Σ_k m_k(t) = 1 exactly
```

Because the sigmoids operate in warped coordinates, (\tau) has the same interpretation for short and long segments. The uniform baseline is (\rho_t=\text{constant}), so adaptive and uniform systems share all other components. (\rho_{\min}) prevents segments from collapsing to zero width.

The complete Stage-1 objective, without VQ, is:

```text
z_k = g( Σ_t m_k(t)·P F_t / Σ_t m_k(t) ),   z_k ∈ R^{d_z}, d_z small (e.g. 16–32)
h_t = [ Σ_k m_k(t) z_k ; u(t) ; log d_k(t) ]       u = frac(C(t)), d_k = segment length
F̂ = D(h_{1:T})      (small bidirectional decoder)
L = ‖F − F̂‖² + λ_R · R,   with τ annealed from large to small
```

Two implementation constraints are important:

- Pool (z_k) from a linear projection (PF), not from the context-rich encoder (v). Otherwise (z_k) can smuggle neighboring information into the segment, making the “segment content” claim empty.
- Keep the boundary predictor small and local for the same reason.

Dropping VQ is acceptable for the first mechanism test, but a bitrate comparison is undefined without a bottleneck. Either fix (d_z) and (K) and call the comparison rate-matched, or add additive uniform noise plus a learned entropy model. Do not leave the rate assumption implicit.

### 7.6 Bit fairness and the informative headline metric

Adaptive boundaries carry side information while a uniform grid does not. Placing (K-1) boundaries on a (T)-frame grid costs approximately:

$$
\log_2 {T\choose K-1}
$$

bits. A fair comparison can give the uniform baseline (K_u>K) extra segments, where approximately:

$$
K_u-K \approx \frac{\text{boundary bits}}
{d_z\times\text{bits per dimension}}.
$$

Otherwise an adaptive model can appear better because its boundary information is hidden from the rate calculation.

Use the following baselines:

1. Uniform segmentation at (K).
2. Uniform segmentation at bit-matched (K_u).
3. DP-optimal segmentation.
4. Learned adaptive segmentation.

Dynamic programming gives the minimum-distortion piecewise-constant segmentation of (F) for each (K), with (O(KT^2)) complexity, which is practical for short utterances. It is a lower bound for this reconstruction family.

Since uniform segmentation is a feasible solution to the optimal (K)-segmentation problem, “adaptive beats uniform” is weak evidence by itself. The informative quantity is the fraction of the DP gap closed:

```text
G = (D_uniform − D_learned) / (D_uniform − D_DP)
```

The decisive test is whether the learned segmenter still wins at matched total bitrate.

### 7.7 Features and evaluation

Start with frozen WavLM layer 6 or HuBERT at approximately 50 Hz, optionally PCA-reduced, as in related query-bottleneck work. These features are content-biased, so the first result says little about acoustics. Add a mel-spectrogram variant after the initial representation works. Experiment A uses synthetic signals and needs no speech features.

Evaluate each stage as follows:

- **Experiment A:** mean absolute boundary error and boundary F1 at ±1 frame across short, long, unequal, and noisy segments.
- **Experiment B:** (D(K)), distortion at bit-matched (K_u), and (G) over a (K)-sweep.
- **Boundary perturbation:** compare (\Delta L) after perturbing a boundary inside a homogeneous region versus near a feature change. Run with and without (u) and (d), since the decoder may otherwise rely on them instead of the boundary.

The immediate implementation target is therefore Experiment A: implement the warp, exact partition masks, and DP baseline on synthetic piecewise-constant signals; verify that masks sum to one and that the known boundaries can be recovered.

---

## 8. Collapse modes and identifiability risks

The model must survive these failure modes:

1. **Identity solution:** (K\approx T) recreates a frame-level latent. Keep (K\ll T) and control the rate.
2. **Mask ignored:** large (z_k) carries all information. Limit (d_z) and match bitrate.
3. **Blurry masks:** large (\tau) lets the decoder ignore segmentation. Anneal temperature and measure boundary sharpness.
4. **Uniform solution:** adaptive boundaries become equally spaced. The matched uniform baseline is mandatory.
5. **Phase leakage:** (u(t)) and (d_k) encode enough timing that (z_k) and masks become unimportant. Run with and without these signals and perturb them at inference.
6. **Boundary leakage:** naïve sigmoid masks are not a perfect partition at sequence edges. Use explicit endpoints or the density-warp construction.
7. **Decoder cheating:** a powerful bidirectional decoder can reconstruct from local context while ignoring the proposed representation. Use a small controlled decoder.
8. **False disentanglement:** content, speaker, and prosody can remain entangled. Add explicit inductive biases before making separation claims.

Timing should live primarily in (b_k,d_k), segment content in (z_k), and global acoustic/style information in an utterance-level vector (g) or a residual stream. This is a testable design hypothesis, not a guaranteed property.

---

## 9. Fair baselines and metrics

Adaptive boundaries carry side information. If the baseline is uniform, boundary locations are free; if the adaptive model transmits (K-1) locations, the comparison is unfair unless total bitrate is matched.

Use these baselines:

1. Uniform segmentation with (K) segments.
2. Uniform segmentation with (K_u>K) segments, chosen to match total bits.
3. Dynamic-programming optimal (K)-segment piecewise-constant reconstruction.
4. Learned adaptive segmentation.
5. Query-based representation without temporal masks.
6. Frame-level latent representation.

For a frame-grid with (T) positions, boundary side information is roughly:

$$
\log_2 {T\choose K-1}
$$

The useful headline quantity is not merely “adaptive beats uniform”; a learned optimizer should beat or approach uniform by construction. Report the fraction of the dynamic-programming gap closed:

$$
G=\frac{D_{\text{uniform}}-D_{\text{learned}}}
{D_{\text{uniform}}-D_{\text{DP}}}.
$$

Report (D(K)), bit-matched distortion, and the full (K)-sweep.

Suggested metrics:

- synthetic boundary mean absolute error;
- boundary F1 at ±1 frame;
- reconstruction distortion;
- intelligibility or dWER;
- speaker similarity;
- UTMOS or another perceptual metric;
- number of units and total bitrate;
- swap-effect size with random-slot controls;
- degradation from boundary perturbation.

---

## 10. Experimental ladder

### Experiment A — Mathematical sanity

Use synthetic piecewise-constant signals:

$$
x(t)=c_k+\epsilon_t
$$

with unequal segment lengths, short and long segments, noisy boundaries, and known ground-truth boundaries.

Check:

- masks sum to one;
- boundaries remain ordered;
- recovered boundaries approximate the truth;
- adaptive segmentation beats or matches uniform at equal (K);
- the model behaves correctly as (K) changes.

If this fails, stop before using speech data.

### Experiment B — Frozen-feature compression

Train on WavLM/HuBERT features first. Compare frame-level, uniform, DP-optimal, query-only, and learned adaptive representations. Evaluate distortion at equal (K) and matched total bitrate.

The initial claim is:

> Adaptive monotonic segmentation provides a better rate-distortion representation than a uniform partition.

### Experiment C — Boundary interpretation

Compare learned boundaries with:

- phoneme alignments;
- syllable boundaries or Sylber-style units;
- pauses and breaths;
- prosodic boundaries;
- acoustic feature changes.

Interpretation is secondary. The model may discover useful acoustic units that are not linguistic units.

### Experiment D — Representation transfer

Use (z_k) for:

- phoneme prediction;
- syllable prediction;
- speech recognition;
- speaker classification;
- prosody or stress prediction.

This tests what information resides in the segments without immediately requiring a full TTS system.

### Experiment E — Composition and swapping

Swap masks, segment latents, and global vectors between utterances. Test whether:

- timing swaps change rhythm;
- content swaps change lexical content;
- global swaps change speaker/style;
- mask or boundary perturbations cause localized, interpretable changes.

Use synthetic data with known factors before trusting unsupervised disentanglement results.

### Experiment F — TTS generation

Only after the representation is validated:

$$
\text{text}
\rightarrow
\text{queries and ordered masks}
\rightarrow
\text{latent composition}
\rightarrow
\text{speech}.
$$

Compare against FastSpeech/MAS-style duration modeling and implicit-alignment systems under the same data and model budget. At inference, text still has to predict the number of segments, their boundaries, and a stop condition. Learned masks relocate the duration problem; they do not remove it.

---

## 11. Additional diagnostic: boundary perturbation

After training, perturb learned boundaries:

$$
b'_k=b_k+\Delta_k.
$$

Measure:

$$
\Delta L=L(B')-L(B).
$$

If boundaries are causally useful, perturbations near important transitions should hurt more than perturbations inside homogeneous regions. Run the test with and without (u(t)) and (d_k(t)), because those features may otherwise hide whether the boundary itself matters.

---

## 12. Practical positioning and final conclusions

### Strongest current research framing

> **Adaptive monotonic temporal segmentation for rate-efficient speech representations**

The generative extension can be described as:

> **Mask-based latent decomposition for compositional speech generation**

### Claims to avoid initially

- “Mask2Former is being applied to TTS.”
- “The model discovers phonemes.”
- “The architecture automatically disentangles content, timing, and acoustics.”
- “Adaptive segmentation beats uniform segmentation,” without equal-(K) and bitrate-matched controls.

### Claims that are currently defensible

- Query and mask mechanisms provide a useful conceptual bridge from image segmentation to speech.
- Duration/alignment mechanisms can be interpreted as restricted temporal assignment models.
- Ordered soft masks are a plausible representation for speech's sequential structure.
- Existing work supports the possibility of unsupervised temporal units, but does not establish the proposed formulation.
- The first scientifically meaningful test is a controlled rate-distortion study, not a full TTS demo.

### Recommended next implementation step

Implement Experiment A on synthetic piecewise-constant signals with:

1. the density-warp boundary parameterization;
2. exact partition masks;
3. uniform and dynamic-programming baselines;
4. a (K)-sweep;
5. boundary error and distortion metrics.

Only if the mechanism passes these tests should it move to frozen speech representations, boundary interpretation, and eventually TTS.

---

## 13. Literature audit and revised novelty assessment — October 2026

This section records a later research pass over primary papers and official proceedings. The purpose is to distinguish what is already established from what remains a defensible research contribution.

### 13.1 The broad idea is no longer novel by itself

The claim “speech has non-uniform information density, so variable-length units may be more efficient than fixed-rate units” now has substantial prior art:

- **SlowAEs** learn variable-rate event representations for speech and connect them to run-length Transformers.
- **SCPC** uses a differentiable boundary detector and segment encoder for variable-length speech units; it reports variable-rate representations down to roughly 14.5 Hz on average while retaining useful phone-classification performance.
- **Sylber** learns syllabic structure through self-distillation and supports syllabic tokenization and reconstruction.
- **TFC** introduces variable frame rates into neural speech codecs using temporal entropy.
- **CodecSlime** compresses temporal redundancy in existing codecs with dynamic frame rate.
- **VARSTok** adaptively segments speech using local feature similarity and reports fewer tokens than a fixed-rate baseline, including downstream zero-shot TTS experiments.
- **FlexiCodec** uses dynamically merged frames and supports controllable very-low-rate speech codecs and TTS.

Therefore, these are no longer strong standalone claims:

1. Adaptive segmentation can reduce token count.
2. Speech information is unevenly distributed in time.
3. Variable-rate speech tokens can be used for generation.
4. Learned segments may correlate with phones or syllables.

The project must now be positioned against these works, not only against a uniform grid.

### 13.2 What remains potentially novel

A defensible contribution could combine the following elements, provided experiments demonstrate that they matter:

1. **A differentiable, ordered, monotonic segmentation parameterization** based on a continuous density warp, rather than a post-hoc clustering or codec-specific merge heuristic.
2. **Exact partition masks** with a formal boundary condition, avoiding edge leakage and making the representation mathematically auditable.
3. **Explicit rate accounting** for both segment content and boundary side information, including a bit-matched uniform baseline.
4. **A DP-optimal segmentation oracle** and a gap-closed metric, rather than treating “adaptive beats uniform” as sufficient evidence.
5. **Causal boundary-use tests**, such as boundary perturbation, phase ablation, and mask replacement, rather than only reporting reconstruction or correlation with linguistic labels.
6. **A cross-feature and cross-language study**, especially comparing content-biased SSL features with mel/acoustic features and testing whether the behavior transfers to Nepali.

None of these is automatically novel in isolation. The novelty would be the controlled combination and the evidence that learned masks provide a useful, interpretable trade-off that existing variable-rate tokenizers do not provide.

### 13.3 Priority reading list

| Priority | Paper | What it establishes | What to compare against |
|---:|---|---|---|
| 1 | [Segmental Contrastive Predictive Coding for Unsupervised Word Segmentation](https://arxiv.org/abs/2106.02170) and [the variable-rate extension](https://arxiv.org/abs/2110.02345) | Differentiable boundaries, segment representations, phone/word discovery, and variable-rate features | Boundary detector, segment pooling, phone metrics |
| 2 | [Variable-rate Discrete Representation Learning](https://research.google/pubs/variable-rate-discrete-representation-learning/) | Slow autoencoders and event-based variable-rate representations | Rate control and event duration modeling |
| 3 | [Sylber](https://arxiv.org/abs/2410.07168) | Self-distilled syllabic structure and low-rate reconstruction | Linguistic boundary quality and syllabic tokenization |
| 4 | [Self-Supervised Syllable Discovery Based on Speaker-Disentangled HuBERT](https://arxiv.org/abs/2409.10103) | Speaker-disentangled syllable discovery from speech-only data | Speaker invariance and boundary robustness |
| 5 | [Unlocking Temporal Flexibility](https://arxiv.org/abs/2505.16845) | Entropy-guided variable frame rate in neural codecs | Dynamic rate allocation and codec integration |
| 6 | [CodecSlime](https://arxiv.org/abs/2506.21074) | Architecture-agnostic dynamic-frame-rate codec compression | Same-model fixed-rate versus dynamic-rate comparison |
| 7 | [VARSTok](https://arxiv.org/abs/2509.04685) and its [AAAI 2026 publication](https://ojs.aaai.org/index.php/AAAI/article/view/40807) | Adaptive clustering, implicit duration coding, and downstream TTS | The closest current variable-rate tokenizer baseline |
| 8 | [FlexiCodec](https://openreview.net/pdf?id=kYkfCs4ZAH) | Dynamic merging at very low frame rates and TTS | Low-rate semantic preservation and controllable frame rate |
| 9 | [WavTokenizer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html) | Strong fixed-rate codec compression at 40/75 tokens per second | A strong fixed-rate acoustic baseline |
| 10 | [Glow-TTS](https://arxiv.org/abs/2005.11129) and [Parallel Tacotron 2](https://arxiv.org/abs/2103.14574) | Monotonic alignment search and differentiable duration learning for TTS | Explicit alignment baselines |

### 13.4 Claims that need correction

**Claim: “Alignment is a restricted mask predictor.”**

This is a useful abstraction, but it is not a new result. MAS, forced alignment, CTC-like monotonic paths, and differentiable duration models already provide closely related assignment mechanisms. Present it as the conceptual bridge that motivates the comparison.

**Claim: “Learned masks discover linguistic structure.”**

Too strong. SCPC, Sylber, and speaker-disentangled HuBERT show that useful phone- or syllable-related structure can emerge under particular objectives and biases. Reconstruction alone may instead discover acoustic stationarity, silence, speaker traits, or codec-friendly boundaries.

**Claim: “Mask2Former/SAM transfers to speech.”**

The transferable ideas are query sets, masks, and prompt-conditioned selection. A direct architectural transplant is not the most compelling contribution, and ordinary Hungarian set matching is poorly suited to ordered speech unless monotonic structure is added.

**Claim: “Adaptive beats uniform.”**

This is weak unless (K), latent dimension, decoder capacity, and total bitrate are controlled. If boundary locations are transmitted, their bits must be counted. Compare against both uniform-at-(K) and bit-matched uniform-at-(K_u).

**Claim: “The representation separates content, timing, and acoustics.”**

Not identifiable without inductive bias. Timing can be assigned to boundaries and durations, content to pooled segment vectors, and global acoustics to a separate vector, but the model can still ignore this intended factorization. Treat separation as a falsifiable evaluation, not a design guarantee.

### 13.5 New questions generated by the literature

1. **What is the unit being optimized?** Does the model seek acoustic stationarity, linguistic change, predictive surprise, perceptual importance, or downstream task utility?
2. **Where should rate be spent?** Is a boundary worth more than another latent dimension or another codebook bit?
3. **Are boundaries causal?** Does moving a boundary damage reconstruction because it identifies a real transition, or because the decoder has learned to depend on the supplied timing coordinates?
4. **What happens under feature substitution?** Do the same boundaries appear in WavLM, HuBERT, mel, and waveform models?
5. **Do boundaries transfer across languages?** A Nepali experiment could distinguish universal acoustic structure from English-trained linguistic bias.
6. **Is one scale enough?** Speech may require nested acoustic, phonetic, syllabic, and prosodic segmentations rather than one flat partition.
7. **Can the method be causal?** The global normalization (C(t)) uses the whole utterance, so the current density-warp design is not naturally streaming. A causal version would need a local budget, online boundary decisions, or delayed normalization.
8. **What is the right distortion?** MSE on SSL features may reward content preservation while ignoring intelligibility, speaker identity, prosody, or waveform quality. Use multiple distortion spaces.
9. **Does adaptive segmentation help TTS after strong codec baselines?** Recent variable-rate codecs already report downstream TTS gains, so the comparison must isolate whether the proposed masks add value beyond better token allocation.

### 13.6 Revised project plan after the audit

**Phase 0 — Reproduction and baseline audit.** Implement uniform segmentation, DP-optimal segmentation, SCPC-style differentiable boundaries, and at least one modern variable-rate codec baseline or its segmentation output. Record total bits, token count, boundary rate, and distortion.

**Phase 1 — Synthetic identifiability.** Test exact partition masks, density warping, minimum segment widths, boundary recovery, noisy transitions, and the effect of phase/duration inputs.

**Phase 2 — Frozen-feature rate-distortion.** Compare WavLM, HuBERT, and mel features. Evaluate equal (K), equal nominal bits, and equal total bits including boundary side information.

**Phase 3 — Boundary meaning and causality.** Compare phone/syllable/prosodic labels, run boundary perturbations, replace masks between utterances, and test whether boundaries remain stable across speakers and languages.

**Phase 4 — Hierarchical extension.** Only if the flat model works, test nested segmentations or separate acoustic and linguistic rates.

**Phase 5 — TTS.** Compare a FastSpeech/MAS baseline, a learned-duration baseline, and the proposed representation under matched data, decoder, speaker conditioning, and bitrate.

### 13.7 Current truth

The original broad idea—“use segmentation ideas to make speech/TTS more efficient”—is valid but already crowded. The strongest remaining version is:

> **A controlled study of differentiable monotonic temporal partitions as rate-constrained speech representations, with explicit boundary costs and causal tests of whether the learned boundaries are functionally useful.**

The first research deliverable should not be a TTS architecture. It should be a reproducible benchmark that answers whether the proposed continuous partition is better, fairer, or more interpretable than existing fixed-rate, clustering-based, and differentiable segmental baselines.
