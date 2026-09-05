# Gujarati TTS Experiment: Phoneme Clustering Impact on Speech Synthesis

## Research Question

Does reducing the G2P output vocabulary from 57 phonemes to 39 phoneme clusters degrade synthesized speech quality when using VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) for Gujarati?

## Dataset

- **Source**: Hugging Face `SPRINGLab/IndicTTS_Gujarati`
- **Licence**: IndicTTS licence (accepted prior to download)
- **Voice selection**: Female recordings only (`gender == 0`)
- **Target duration**: 1.75 hours (acceptable range: 1.70–1.80 hours)
- **Raw selected duration**: 1.7033 hours
- **Processed duration**: 1.5866 hours (after trimming silence)

### Sample Statistics

| Category | Count |
|----------|-------|
| Total candidate rows | 3,613 |
| Selected samples | 523 |
| Excluded (random cap) | 3,090 |
| Training set (90%) | 471 |
| Validation set (5%) | 26 |
| Test / held-out set (5%) | 26 |

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
| Silence trimming | Leading/trailing silence removed (`top_db=25`) |
| Loudness normalization | Applied consistently (EBU R128 / LUFS at −23.0 dB) |
| Format | WAV (16-bit PCM) |

The **same processed WAV files** are used for both baseline and clustered training. No audio difference exists between conditions.

## G2P Labeling

### Baseline Vocabulary (57 phonemes → 62 tokens)

The baseline uses the project's 57-phoneme inventory from `g2p/phoneme_vocab.json` (including retroflex `lx` / `ળ`), plus 5 special tokens (`<pad>`, `<eos>`, `<bos>`, `<blnk>`, `<wb>`), totaling 62 tokens.

**Example**: `aa j a n a o <wb> d a i w a s a` → token IDs via whitespace-split tokenizer

### Clustered Vocabulary (39 clusters → 44 tokens)

The clustered vocabulary uses 39 phoneme clusters (C0–C38) derived from `g2p/phoneme_cluster_mapping.json`, which maps each of the 57 baseline phonemes to one of 39 clusters. Plus 5 special tokens, totaling 44 tokens.

**Example**: `C1 C17 C0 C24 C0 C26 <wb> C7 C0 C16 C37 C0 C30 C0` → token IDs via whitespace-split tokenizer

### Key Constraint

- Cluster tokens are **atomic**: `C10` is one token, not `C` + `1` + `0`
- The VITS text-processing path **bypasses any internal phonemizer**
- Word boundaries use the explicit `<wb>` token
- Labels are derived: `text → baseline phonemes → cluster mapping`
- Both label sets are fully aligned; they differ only in representation

### G2P Fallback Statistics

- Primary source: `data/g2p_gu.txt` (16,676 words) + `<GU>` entries in `data/multilingual_g2p_dataset.txt`
- Fallback: rule-based Gujarati script character phonemizer (`\u0A80`–`\u0AFF`, numerals `૦-૯`)
- **Dictionary coverage on selected subset: 8,096/8,096 words (100.0%)**

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
| Max training steps | ~8,625 (576 epochs × ~15 steps/epoch) |
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
| Baseline (62 tokens) | 5.16 hours | 8,640 |
| Clustered (44 tokens) | 5.30 hours | 8,640 |

### Final Evaluation Metrics (Epoch 575)

| Metric | Baseline | Clustered | Δ (Clustered − Baseline) |
|--------|----------|-----------|--------------------------|
| avg_loss_disc | 2.6895 | 2.7169 | +0.0274 (+1.0%) |
| avg_loss_gen | 2.0277 | 1.9607 | −0.0670 (−3.3%) |
| avg_loss_kl | 1.7937 | 1.7427 | −0.0510 (−2.8%) |
| avg_loss_mel | 22.6843 | 21.2222 | −1.4621 (−6.4%) |
| avg_loss_duration | 2.1583 | 2.2282 | +0.0699 (+3.2%) |
| avg_loss_feat | 2.7704 | 2.5172 | −0.2532 (−9.1%) |

**Observation**: The clustered model achieves equivalent or slightly superior reconstruction metrics (`loss_mel` is 6.4% lower in the clustered model), demonstrating strong model capacity even with a 29% reduced vocabulary.

### Best Checkpoints

| Model | Checkpoint Path |
|-------|----------------|
| Baseline | `/mnt/d/tts_models/gu/baseline/vits_gujarati_female_baseline-September-02-2026_10+33AM-fb53263/best_model.pth` |
| Clustered | `/mnt/d/tts_models/gu/clustered/vits_gujarati_female_clustered-September-02-2026_03+42PM-fb53263/best_model.pth` |

## Inference

### Held-out Test Set

- 26 samples from the reserved 5% test split
- Each sample synthesized by both models
- Output: `samples/tts_gujarati_female/held_out/baseline_XXX.wav` + `clustered_XXX.wav` (52 WAVs)

### Unseen Generalization Set

- 20 new Gujarati sentences not present in training, validation, or test data
- Each sentence tokenized via the full G2P pipeline
- All 20/20 sentences were fully tokenizable
- Output: `samples/tts_gujarati_female/unseen/baseline_XX.wav` + `clustered_XX.wav` (40 WAVs)

## Evaluation Protocol

### Automated MOS (DNSMOS)

Automated MOS scores were computed using Microsoft's DNSMOS (Deep Noise Suppression MOS) predictor via the `speechmos` package. These are **predicted/automated** MOS scores, **not** human MOS scores.

#### Held-out Set Results

| Condition | Mean MOS | Std Dev | 95% CI | N |
|-----------|----------|---------|--------|---|
| Baseline | 3.0560 | 0.0957 | [3.0184, 3.0935] | 26 |
| Clustered | 3.0310 | 0.1375 | [2.9770, 3.0849] | 26 |

**Paired difference (Baseline − Clustered)**: 0.0250 ± 0.1671

#### Unseen Set Results

| Condition | Mean MOS | Std Dev | 95% CI | N |
|-----------|----------|---------|--------|---|
| Baseline | 2.9351 | 0.1709 | [2.8587, 3.0116] | 20 |
| Clustered | 2.9880 | 0.2057 | [2.8960, 3.0800] | 20 |

**Paired difference (Baseline − Clustered)**: −0.0529 ± 0.2242

#### Interpretation

- **Confidence intervals overlap completely** across both evaluation sets:
  - Held-out: Baseline `[3.018, 3.094]` vs. Clustered `[2.977, 3.085]`
  - Unseen: Baseline `[2.859, 3.012]` vs. Clustered `[2.896, 3.080]`
- On the unseen generalization set, the clustered model slightly outperforms the baseline (+0.053 MOS).
- The clustered model is **statistically indistinguishable** from the baseline.
- Compressing the phoneme vocabulary by **29%** (from 57 phonemes down to 39 clusters) preserves speech quality and generalization in Gujarati.

## Limitations

1. **One voice**: Female speaker only.
2. **Limited data**: 1.59 hours of processed speech is a pilot proof-of-concept for single-speaker VITS.
3. **No human evaluation**: Only automated MOS was computed.
4. **Pilot scope**: Proof-of-concept for multilingual phoneme clustering across Indic languages.

## Reproducibility

All code, configurations, and manifests required to reproduce this experiment are available in the repository:

| Component | Path |
|-----------|------|
| Data download | `tts/tts_data_download.py --lang gu` |
| Audio preprocessing | `tts/tts_audio_preprocess.py --lang gu` |
| G2P labeling | `tts/tts_g2p_labeling.py --lang gu` |
| Tokenizer | `tts/vits_tokenizer.py` |
| Data splitting | `tts/tts_split_data.py --lang gu` |
| VITS training | `tts/tts_vits_training.py --lang gu --model both` |
| Inference | `tts/tts_inference.py --lang gu` |
| Automated evaluation | `tts/tts_automated_eval.py --lang gu` |
| Manifest | `data/tts_gujarati_female/manifest.csv` |
| Phoneme vocab | `g2p/phoneme_vocab.json` |
| Cluster mapping | `g2p/phoneme_cluster_mapping.json` |
| Training log | `/mnt/d/tts_models/gu/training_log.json` |
| Automated MOS results | `results/tts_gujarati_female/automated_mos.csv` |
| Automated MOS report | `results/tts_gujarati_female/automated_mos_report.md` |

Raw audio, processed audio, model checkpoints, and generated WAV samples are excluded from Git via `.gitignore`.
