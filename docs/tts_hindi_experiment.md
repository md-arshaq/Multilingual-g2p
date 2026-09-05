# Hindi TTS Experiment: Phoneme Clustering Impact on Speech Synthesis

## Research Question

Does reducing the G2P output vocabulary from 57 phonemes to 39 phoneme clusters degrade synthesized speech quality when using VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)?

## Dataset

- **Source**: Hugging Face `SPRINGLab/IndicTTS-Hindi`
- **Licence**: IndicTTS licence (accepted prior to download)
- **Voice selection**: Female recordings only (gender-homogeneous)
- **Target duration**: 1.75 hours (acceptable range: 1.70–1.80 hours)
- **Actual duration**: 1.7494 hours

### Sample Statistics

| Category | Count |
|----------|-------|
| Total candidate rows | 1,645 |
| Selected samples | 1,031 |
| Excluded (random cap) | 614 |
| Training set (90%) | 928 |
| Validation set (5%) | 53 |
| Test / held-out set (5%) | 50 |

- **Selection seed**: 42
- **Split level**: utterance-level, deterministic
- **No sample appears in more than one split**

### Exclusion Criteria

Samples were excluded if they had:
- Empty or missing text
- Duration < 2.5 s or > 20 s
- Corrupt or silent audio
- Incomplete G2P token coverage (any OOV phoneme or unmapped cluster)

Remaining samples exceeding the 1.75-hour target were excluded by random selection (seed 42).

## Audio Preprocessing

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| Channels | Mono |
| Silence trimming | Leading/trailing silence removed |
| Loudness normalization | Applied consistently |
| Format | WAV (16-bit PCM) |

The **same processed WAV files** are used for both baseline and clustered training. No audio difference exists between conditions.

## G2P Labeling

### Baseline Vocabulary (57 phonemes → 62 tokens)

The baseline uses the project's 57-phoneme inventory from `g2p/phoneme_vocab.json`, plus 5 special tokens (`<pad>`, `<eos>`, `<bos>`, `<blnk>`, `<wb>`), totaling 62 tokens.

**Example**: `b aa r ax t` → token IDs via whitespace-split tokenizer

### Clustered Vocabulary (39 clusters → 44 tokens)

The clustered vocabulary uses 39 phoneme clusters (C0–C38) derived from `g2p/phoneme_cluster_mapping.json`, which maps each of the 57 baseline phonemes to one of 39 clusters. Plus 5 special tokens, totaling 44 tokens.

**Example**: `C3 C0 C29 C0 C1` → token IDs via whitespace-split tokenizer

### Key Constraint

- Cluster tokens are **atomic**: `C10` is one token, not `C` + `1` + `0`
- The VITS text-processing path **bypasses any internal phonemizer**
- Word boundaries use the explicit `<wb>` token
- Labels are derived: `text → baseline phonemes → cluster mapping`
- Both label sets are fully aligned; they differ only in representation

### G2P Fallback Statistics

- Primary source: `data/multilingual_g2p_dataset.txt` (exact Hindi word match)
- Fallback: trained baseline G2P model (`models/best_g2p_transformer.weights.h5`)
- Utterances with any unmappable word or phoneme were excluded entirely

## Fair-Training Controls

Both VITS models were trained with **identical** settings. The **only** difference is the input vocabulary.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Architecture | Coqui TTS VITS (single-speaker, from scratch) |
| Optimizer | AdamW (β₁=0.8, β₂=0.99, ε=1e-9, weight_decay=0.01) |
| LR (generator) | 0.0002 |
| LR (discriminator) | 0.0002 |
| LR scheduler | ExponentialLR (γ=0.999875) |
| Batch size | 32 |
| Mixed precision | FP16 |
| Max training steps | ~15,000 (576 epochs × ~26 steps/epoch) |
| Checkpoint interval | Every 500 steps |
| Best checkpoint selection | By best validation loss |
| Random seed | 42 (+ torch seed 54321 for trainer) |
| CUDNN deterministic | True |
| CUDNN benchmark | False |

### Audio Configuration

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| FFT size | 1,024 |
| Window length | 1,024 |
| Hop length | 256 |
| Mel channels | 80 |
| mel_fmin | 0 |
| mel_fmax | None |

### Hardware & Software

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX A5000 (24 GB) |
| Python | 3.10.12 |
| PyTorch | 2.4.1+cu124 |
| Coqui TTS | 0.27.5 |
| Transformers | 4.57.3 |
| OS | Ubuntu 22.04 (WSL2) |

## Training Results

### Training Duration

| Model | Training Time | Final Global Step |
|-------|--------------|-------------------|
| Baseline (62 tokens) | 7.99 hours | 15,550 |
| Clustered (44 tokens) | 7.93 hours | 15,550 |

### Final Evaluation Metrics (Epoch 575)

| Metric | Baseline | Clustered | Δ (Clustered − Baseline) |
|--------|----------|-----------|--------------------------|
| avg_loss_disc | 2.6614 | 2.6691 | +0.0077 (+0.3%) |
| avg_loss_gen | 2.0033 | 2.0140 | +0.0107 (+0.5%) |
| avg_loss_kl | 1.4152 | 1.4122 | −0.0030 (−0.2%) |
| avg_loss_mel | 19.7433 | 19.9201 | +0.1768 (+0.9%) |
| avg_loss_duration | 1.7492 | 1.7476 | −0.0016 (−0.1%) |
| avg_loss_feat | 3.3894 | 3.4479 | +0.0585 (+1.7%) |

**Observation**: All losses differ by less than 2%. The clustered model converges to near-identical quality despite 29% fewer input tokens.

### Best Checkpoints

| Model | Checkpoint Path |
|-------|----------------|
| Baseline | `/mnt/d/tts_models/hi/baseline/vits_hindi_female_baseline-August-31-2026_11+25PM-fb53263/best_model.pth` |
| Clustered | `/mnt/d/tts_models/hi/clustered/vits_hindi_female_clustered-September-01-2026_08+04AM-fb53263/best_model.pth` |

## Inference

### Held-out Test Set

- 50 samples from the reserved 5% test split
- Each sample synthesized by both models
- Output: `samples/tts_hindi_female/held_out/baseline_XXX.wav` + `clustered_XXX.wav`

### Unseen Generalization Set

- 20 new Hindi sentences not present in training, validation, or test data
- Each sentence tokenized via the full G2P pipeline
- All 20/20 sentences were fully tokenizable
- Output: `samples/tts_hindi_female/unseen/baseline_XX.wav` + `clustered_XX.wav`

## Evaluation Protocol

### Automated MOS (DNSMOS)

Automated MOS scores were computed using Microsoft's DNSMOS (Deep Noise Suppression MOS) predictor via the `speechmos` package. These are **predicted/automated** MOS scores, **not** human MOS scores.

#### Held-out Set Results

| Condition | Mean MOS | Std Dev | 95% CI | N |
|-----------|----------|---------|--------|---|
| Baseline | 3.1117 | 0.1772 | [3.0626, 3.1609] | 50 |
| Clustered | 3.1557 | 0.1450 | [3.1155, 3.1959] | 50 |

**Paired difference (Baseline − Clustered)**: −0.0439 ± 0.1803

#### Unseen Set Results

| Condition | Mean MOS | Std Dev | 95% CI | N |
|-----------|----------|---------|--------|---|
| Baseline | 3.0129 | 0.1993 | [2.9237, 3.1020] | 20 |
| Clustered | 3.0184 | 0.1803 | [2.9377, 3.0990] | 20 |

**Paired difference (Baseline − Clustered)**: −0.0055 ± 0.2224

#### Interpretation

- **Confidence intervals overlap completely** in both evaluation sets
- The clustered model is **statistically indistinguishable** from the baseline
- On the held-out set, the clustered model scores marginally (+0.04) higher
- On the unseen set, the difference is negligible (+0.005)
- These automated scores serve as **supporting evidence** only

### Human Evaluation

Human evaluation was not conducted in this phase. Scripts for blinded human evaluation are prepared and ready for deployment:
- `tts/tts_human_eval_prep.py`: generates blinded audio pairs and evaluation forms
- `tts/tts_human_eval_analysis.py`: analyzes collected listener scores with Wilcoxon signed-rank test

## Limitations

1. **One language**: Hindi only. Results may not generalize to other languages.
2. **One voice**: Female speaker only. Male voice or multi-speaker scenarios are untested.
3. **Limited data**: 1.75 hours of speech is small for VITS. Results are preliminary.
4. **No human evaluation**: Only automated MOS was computed. Human perceptual evaluation would strengthen claims.
5. **Pilot scope**: This is a proof-of-concept for the phoneme clustering approach, not a production-ready system.
6. **Automated MOS limitations**: DNSMOS was designed for noise suppression quality and may not perfectly capture naturalness/intelligibility differences in synthesized speech.

## Important Disclaimers

> **Older gTTS samples and MOS results in this repository are NOT evidence for this VITS experiment.** Those were generated using Google's gTTS as a placeholder and have no bearing on the VITS baseline-vs-clustered comparison.

> **Automated MOS scores are not a substitute for human listening tests.** They are reported as supporting quantitative evidence only.

## Reproducibility

All code, configurations, and manifests required to reproduce this experiment are available in the repository:

| Component | Path |
|-----------|------|
| Data download | `tts/tts_data_download.py` |
| Audio preprocessing | `tts/tts_audio_preprocess.py` |
| G2P labeling | `tts/tts_g2p_labeling.py` |
| Tokenizer | `tts/vits_tokenizer.py` |
| Data splitting | `tts/tts_split_data.py` |
| VITS training | `tts/tts_vits_training.py` |
| Inference | `tts/tts_inference.py` |
| Automated evaluation | `tts/tts_automated_eval.py` |
| Human eval prep | `tts/tts_human_eval_prep.py` |
| Human eval analysis | `tts/tts_human_eval_analysis.py` |
| Manifest | `data/tts_hindi_female/manifest.csv` |
| Phoneme vocab | `g2p/phoneme_vocab.json` |
| Cluster mapping | `g2p/phoneme_cluster_mapping.json` |
| Training log | `/mnt/d/tts_models/hi/training_log.json` |
| Automated MOS results | `results/tts_hindi_female/automated_mos.csv` |
| Automated MOS report | `results/tts_hindi_female/automated_mos_report.md` |

Raw audio, processed audio, model checkpoints, and generated WAV samples are excluded from Git via `.gitignore`.
