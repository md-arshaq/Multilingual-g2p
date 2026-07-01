# TTS Model Selection — Rationale Document

> **Project:** Multilingual G2P with Phoneme Clustering (Samsung R&D Internship)
> **Objective:** Select an open-source TTS model that accepts phoneme-level input for Hindi, Gujarati, and Marathi

---

## Requirements

The TTS model must satisfy these constraints:

| Constraint | Required |
|------------|----------|
| **Phoneme-level input** | Must accept phoneme sequences (not grapheme-only). Our G2P model outputs phonemes/cluster IDs. |
| **Indian language support** | Must support Hindi, Gujarati, Marathi — or be trainable from scratch for these languages |
| **Open-source** | Open weights and training code (for paper reproducibility) |
| **Custom phoneme vocabulary** | Must allow swapping the phoneme set to our 12-cluster vocabulary |
| **GPU budget** | Must train on a single T4 (Colab) or consumer GPU |
| **Training data** | We have IndicTTS WAV + text files for all 3 languages |

---

## Candidates Evaluated

### Option 1: Coqui TTS — VITS

**Architecture:** VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
— combines VAE, normalizing flow, and adversarial training in a single end-to-end model.

| Criterion | Assessment |
|-----------|------------|
| Phoneme input | ✅ **Native.** VITS is designed for phoneme-level input. The `text_to_sequence` function maps phonemes to IDs. |
| Custom vocab | ✅ **Easy.** Edit `symbols.py` to define your cluster labels (C0–C11) as the phoneme set. |
| Indian languages | ✅ **Trainable.** No pre-trained Indian language models, but VITS trains from scratch in ~24h on T4 with ~2h of audio. |
| Open-source | ✅ Mozilla Public License 2.0 |
| GPU requirement | ✅ Fits in 15 GB VRAM (T4 = 16 GB) |
| Training data needed | ~1–2 hours of clean audio per language (we have this) |
| Python API | ✅ `pip install TTS` — inference in 3 lines of code |
| Community | ✅ Active community, well-documented, many forks |
| Audio quality | ✅ Near state-of-the-art for single-speaker synthesis |

**Pros:**
- End-to-end: no separate vocoder needed
- Directly accepts phoneme sequences → trivial to plug in our G2P output
- Fast training (~12–24h for a single speaker on T4)
- Active maintenance and documentation

**Cons:**
- Single-speaker by default (multi-speaker requires speaker embeddings)
- Quality depends on audio data cleanliness

### Option 2: FastSpeech 2 + HiFi-GAN

**Architecture:** FastSpeech 2 (non-autoregressive acoustic model with duration, pitch, energy predictors) + HiFi-GAN (neural vocoder for waveform generation).

| Criterion | Assessment |
|-----------|------------|
| Phoneme input | ✅ **Native.** FastSpeech 2 uses phoneme IDs with duration alignment. |
| Custom vocab | ✅ **Possible** but requires editing the data preprocessing pipeline. |
| Indian languages | ✅ **Trainable.** Needs Montreal Forced Aligner (MFA) for duration extraction. |
| Open-source | ✅ MIT License (multiple implementations: ming024, espnet) |
| GPU requirement | ✅ Fits on T4, but HiFi-GAN vocoder adds training overhead |
| Training data needed | ~3–5 hours per language (more than VITS) |
| Python API | ⚠️ No unified pip package — requires cloning repos and custom scripts |
| Audio quality | ✅ Very good, especially with HiFi-GAN v2 vocoder |

**Pros:**
- Non-autoregressive → very fast inference
- Explicit duration/pitch control — good for debugging
- Well-studied architecture with many research papers

**Cons:**
- **Two-stage pipeline:** Must train acoustic model AND vocoder separately
- Needs forced alignment (MFA) as preprocessing step → adds complexity
- More training data needed for good quality
- No single unified Python package

### Option 3: XTTS v2 (Coqui)

**Architecture:** GPT-based autoregressive model with cross-lingual voice cloning capability.

| Criterion | Assessment |
|-----------|------------|
| Phoneme input | ❌ **Grapheme-level.** XTTS uses an internal tokenizer/phonemizer — cannot easily inject custom phoneme sequences. |
| Custom vocab | ❌ **Difficult.** The model's text processing is tightly coupled to its pre-trained tokenizer. |
| Indian languages | ⚠️ Has some Hindi support, but Gujarati and Marathi are not officially supported. |
| Open-source | ⚠️ Coqui license — non-commercial use only |
| GPU requirement | ❌ Needs >8 GB VRAM for inference, >16 GB for fine-tuning |
| Training data needed | Zero-shot (6 seconds of reference audio) — but quality varies significantly |
| Audio quality | ✅ Excellent for supported languages |

**Pros:**
- Zero-shot voice cloning — impressive demos
- Multilingual out of the box (17 languages)

**Cons:**
- **Cannot use our G2P output** — defeats the purpose of this project
- Non-commercial license
- Heavy compute requirements
- Gujarati/Marathi not supported
- Black-box phonemizer → no control over phoneme-to-audio mapping

---

## Decision Matrix

| Criterion (weight) | VITS (Coqui) | FastSpeech 2 + HiFi-GAN | XTTS v2 |
|---------------------|:---:|:---:|:---:|
| Phoneme input (critical) | ✅ | ✅ | ❌ |
| Custom vocab (critical) | ✅ | ✅ | ❌ |
| Indian languages (high) | ✅ | ✅ | ⚠️ |
| Ease of integration (high) | ✅ | ⚠️ | ❌ |
| Training efficiency (medium) | ✅ | ⚠️ | ✅ |
| Open-source (medium) | ✅ | ✅ | ⚠️ |
| Audio quality (medium) | ✅ | ✅ | ✅ |
| **Score** | **7/7** | **5/7** | **2/7** |

---

## Recommendation

### 🏆 Primary: **Coqui VITS**

VITS is the best fit for this project because:

1. **It natively accepts phoneme input** — we can directly feed our G2P model's output (either raw phonemes or cluster-decoded phonemes) into VITS without any adapter layer
2. **Custom vocabulary is trivial** — edit `symbols.py` to use our 12 cluster labels
3. **End-to-end** — no separate vocoder training needed
4. **Proven on low-resource languages** — VITS has been successfully trained on languages with <2h of audio data
5. **Single pip install** — `pip install TTS`

### Integration Architecture

```
┌──────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────┐
│  Input   │────▶│  G2P Model   │────▶│  Cluster → IPA │────▶│  VITS    │────▶ Audio
│  Text    │     │ (Transformer)│     │   Mapping      │     │  Model   │
│ <HI> भारत│     │ → C0 C2 C7...│     │ → b aa r ə t   │     │          │
└──────────┘     └──────────────┘     └────────────────┘     └──────────┘
```

**Key step:** The "Cluster → IPA Mapping" converts cluster IDs back to representative phonemes that VITS understands. For each cluster, we pick the most frequent member phoneme as the representative.

### Fallback: **FastSpeech 2 + HiFi-GAN**

If VITS training is unstable or quality is poor, FastSpeech 2 is the backup. It requires more setup (forced alignment) but gives explicit control over duration and pitch.

---

## Implementation Plan

### Phase 1: Setup & Baseline TTS
1. Install Coqui TTS: `pip install TTS`
2. Define custom phoneme set in VITS config
3. Prepare IndicTTS audio data in LJ-Speech format
4. Train single-speaker VITS for Hindi first (proof of concept)
5. Evaluate output quality

### Phase 2: Multilingual Training
1. Train separate VITS models for Gujarati and Marathi
2. OR: train one multi-speaker model with language conditioning (if data permits)

### Phase 3: G2P Integration
1. Replace VITS phonemizer with our G2P model
2. Test with both baseline phoneme output and clustered output
3. Generate comparison audio samples

### Phase 4: MOS Evaluation
1. Generate ≥10 test sentences per language under both conditions
2. Score with UTMOS (automated) + human listeners (4 available)
3. Compare MOS scores

---

## References

- **VITS:** Kim et al., "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (ICML 2021)
- **FastSpeech 2:** Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (ICLR 2021)
- **XTTS:** Coqui AI, "XTTS: Cross-lingual Text-to-Speech" (2023)
- **Coqui TTS:** https://github.com/coqui-ai/TTS
- **IndicTTS:** https://www.iitm.ac.in/donlab/indictts/database
