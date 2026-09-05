# Marathi TTS Experiment: Phoneme Clustering Impact on Speech Synthesis

## Research Question

Does reducing the G2P output vocabulary from 57 phonemes to 39 phoneme clusters degrade synthesized speech quality when using VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) for Marathi?

## Dataset

- **Source**: Hugging Face `SPRINGLab/IndicTTS_Marathi`
- **Licence**: IndicTTS licence (accepted prior to download)
- **Voice selection**: Female recordings only (`gender == 0`)
- **Target duration**: 1.75 hours (acceptable range: 1.70–1.80 hours)
- **Raw selected duration**: 1.7021 hours
- **Processed duration**: 1.5400 hours (after trimming silence)

### Sample Statistics

| Category | Count |
|----------|-------|
| Total candidate rows | 5,330 |
| Selected samples | 803 |
| Excluded (random cap) | 4,527 |
| Training set (90%) | 723 |
| Validation set (5%) | 40 |
| Test / held-out set (5%) | 40 |

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

The baseline uses the project's 57-phoneme inventory from `g2p/phoneme_vocab.json` (including Marathi retroflex `lx` / `ळ`), plus 5 special tokens (`<pad>`, `<eos>`, `<bos>`, `<blnk>`, `<wb>`), totaling 62 tokens.

**Example**: `aa j a c aa <wb> d a i w a s a` → token IDs via whitespace-split tokenizer

### Clustered Vocabulary (39 clusters → 44 tokens)

The clustered vocabulary uses 39 phoneme clusters (C0–C38) derived from `g2p/phoneme_cluster_mapping.json`, which maps each of the 57 baseline phonemes to one of 39 clusters. Plus 5 special tokens, totaling 44 tokens.

**Example**: `C1 C17 C0 C5 C1 <wb> C7 C0 C16 C37 C0 C30 C0` → token IDs via whitespace-split tokenizer

### Key Constraint

- Cluster tokens are **atomic**: `C10` is one token, not `C` + `1` + `0`
- The VITS text-processing path **bypasses any internal phonemizer**
- Word boundaries use the explicit `<wb>` token
- Labels are derived: `text → baseline phonemes → cluster mapping`
- Both label sets are fully aligned; they differ only in representation

### G2P Fallback Statistics

- Primary source: `data/g2p_mr.txt` (22,454 words) + `<MR>` entries in `data/multilingual_g2p_dataset.txt`
- Fallback: rule-based Devanagari character phonemizer with explicit support for Marathi retroflex `ळ` (`lx`), `ॅ` (`ae`), and `ॉ` (`o`)
- **Dictionary coverage on selected subset: 8,456/8,456 words (100.0%)**

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
| Max training steps | ~15,000 (576 epochs × ~23 steps/epoch) |
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
| Baseline (62 tokens) | 7.04 hours | 13,248 |
| Clustered (44 tokens) | 7.12 hours | 13,248 |

### Final Evaluation Metrics (Epoch 575)

| Metric | Baseline | Clustered | Δ (Clustered − Baseline) |
|--------|----------|-----------|--------------------------|
| avg_loss_disc | 2.7428 | 2.7826 | +0.0398 (+1.4%) |
| avg_loss_gen | 1.9018 | 1.9156 | +0.0138 (+0.7%) |
| avg_loss_kl | 1.7684 | 1.7172 | −0.0512 (−2.9%) |
| avg_loss_mel | 18.7136 | 18.7366 | +0.0230 (+0.1%) |
| avg_loss_duration | 1.8983 | 1.8954 | −0.0029 (−0.1%) |
| avg_loss_feat | 2.5743 | 2.6204 | +0.0461 (+1.8%) |

**Observation**: Loss metrics are virtually identical across both models. `loss_mel` differs by only **+0.1%**, while `loss_duration` and `loss_kl` show matched convergence.

### Best Checkpoints

| Model | Checkpoint Path |
|-------|----------------|
| Baseline | `/mnt/d/tts_models/mr/baseline/vits_marathi_female_baseline-September-01-2026_06+29PM-fb53263/best_model.pth` |
| Clustered | `/mnt/d/tts_models/mr/clustered/vits_marathi_female_clustered-September-02-2026_01+32AM-fb53263/best_model.pth` |

## Inference

### Held-out Test Set

- 40 samples from the reserved 5% test split
- Each sample synthesized by both models
- Output: `samples/tts_marathi_female/held_out/baseline_XXX.wav` + `clustered_XXX.wav`

### Unseen Generalization Set

- 20 new Marathi sentences not present in training, validation, or test data
- Each sentence tokenized via the full G2P pipeline
- All 20/20 sentences were fully tokenizable
- Output: `samples/tts_marathi_female/unseen/baseline_XX.wav` + `clustered_XX.wav`

## Evaluation Protocol

### Automated MOS (DNSMOS)

Automated MOS scores were computed using Microsoft's DNSMOS (Deep Noise Suppression MOS) predictor via the `speechmos` package. These are **predicted/automated** MOS scores, **not** human MOS scores.

#### Held-out Set Results

| Condition | Mean MOS | Std Dev | 95% CI | N |
|-----------|----------|---------|--------|---|
| Baseline | 3.1074 | 0.1188 | [3.0706, 3.1442] | 40 |
| Clustered | 3.0893 | 0.1776 | [3.0342, 3.1443] | 40 |

**Paired difference (Baseline − Clustered)**: 0.0182 ± 0.1822

#### Unseen Set Results

| Condition | Mean MOS | Std Dev | 95% CI | N |
|-----------|----------|---------|--------|---|
| Baseline | 3.0355 | 0.1816 | [2.9542, 3.1167] | 20 |
| Clustered | 3.0059 | 0.1743 | [2.9279, 3.0838] | 20 |

**Paired difference (Baseline − Clustered)**: 0.0296 ± 0.2044

#### Interpretation

- **Confidence intervals overlap completely** in both evaluation sets:
  - Held-out: Baseline `[3.071, 3.144]` vs. Clustered `[3.034, 3.144]`
  - Unseen: Baseline `[2.954, 3.117]` vs. Clustered `[2.928, 3.084]`
- The clustered model is **statistically indistinguishable** from the baseline.
- Reducing the phoneme vocabulary by **29%** (from 57 phonemes down to 39 clusters) does not compromise acoustic reconstruction or speech quality in Marathi.

## Limitations

1. **One voice**: Female speaker only.
2. **Limited data**: 1.54 hours of processed speech is a pilot proof-of-concept for single-speaker VITS.
3. **No human evaluation**: Only automated MOS was computed.
4. **Pilot scope**: Proof-of-concept for multilingual phoneme clustering across Indic languages.

## Reproducibility

All code, configurations, and manifests required to reproduce this experiment are available in the repository:

| Component | Path |
|-----------|------|
| Data download | `tts/tts_data_download.py --lang mr` |
| Audio preprocessing | `tts/tts_audio_preprocess.py --lang mr` |
| G2P labeling | `tts/tts_g2p_labeling.py --lang mr` |
| Tokenizer | `tts/vits_tokenizer.py` |
| Data splitting | `tts/tts_split_data.py --lang mr` |
| VITS training | `tts/tts_vits_training.py --lang mr --model both` |
| Inference | `tts/tts_inference.py --lang mr` |
| Automated evaluation | `tts/tts_automated_eval.py --lang mr` |
| Manifest | `data/tts_marathi_female/manifest.csv` |
| Phoneme vocab | `g2p/phoneme_vocab.json` |
| Cluster mapping | `g2p/phoneme_cluster_mapping.json` |
| Training log | `/mnt/d/tts_models/mr/training_log.json` |
| Automated MOS results | `results/tts_marathi_female/automated_mos.csv` |
| Automated MOS report | `results/tts_marathi_female/automated_mos_report.md` |

Raw audio, processed audio, model checkpoints, and generated WAV samples are excluded from Git via `.gitignore`.
