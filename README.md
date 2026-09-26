# Multilingual Grapheme-to-Phoneme (G2P) with Phoneme Clustering

> **Samsung R&D Internship Project**

A multilingual G2P system for Indian languages (Hindi, Gujarati, Marathi) that converts written text into phoneme sequences, with **Acoustic Phonetic Folding** to reduce output vocabulary while preserving pronunciation quality. Validated end-to-end through VITS neural TTS synthesis across all three languages.

---

## Key Results

### G2P Model Accuracy

| Metric | Baseline | Clustered (Phonetic Folding) | Change |
|--------|----------|------------------------------|--------|
| **PER** | 0.03% | 0.08% | +0.05pp |
| **WER** | 0.22% | 0.35% | +0.13pp |
| **Output Vocab** | 57 phonemes | 39 clusters | **−31.6%** |
| **Parameters** | 1,422,141 | 1,417,515 | −4,626 |

### TTS Speech Quality (VITS, DNSMOS)

| Language | Baseline MOS | Clustered MOS | Significant? |
|----------|-------------|---------------|--------------|
| Hindi | 3.11 ± 0.18 | 3.16 ± 0.15 | No (p=0.09) |
| Marathi | 3.11 ± 0.12 | 3.09 ± 0.18 | No (p=0.52) |
| Gujarati | 3.06 ± 0.10 | 3.03 ± 0.14 | No (p=0.39) |

> Both models achieve >99.9% accuracy. The clustered model reduces vocabulary by 31.6% with **no statistically significant loss** in speech quality across all three languages (Cohen's d < 0.25).

---

## Project Structure

```
├── data/                                # Datasets & pronunciation dictionaries
│   ├── g2p_hi.txt                       # Hindi pronunciation lexicon
│   ├── g2p_gu.txt                       # Gujarati pronunciation lexicon
│   ├── g2p_mr.txt                       # Marathi pronunciation lexicon
│   ├── multilingual_g2p_dataset.txt     # Combined multilingual dataset (54,753 entries)
│   ├── multilingual_g2p_clustered.txt   # Clustered version (phonetic folding applied)
│   ├── tts_hindi_female/                # Hindi TTS training data (audio + manifests)
│   ├── tts_marathi_female/              # Marathi TTS training data
│   └── tts_gujarati_female/             # Gujarati TTS training data
│
├── docs/                                # Documentation & experiment reports
│   ├── G2P_Project_Context.md           # Project context & background
│   ├── PROJECT-PIPELINE.pdf             # Pipeline overview
│   ├── PROJECT_OVERVIEW.jpeg            # Architecture diagram
│   ├── tts_model_selection.md           # TTS model comparison & selection rationale
│   ├── tts_hindi_experiment.md          # Hindi VITS experiment report
│   ├── tts_marathi_experiment.md        # Marathi VITS experiment report
│   ├── tts_gujarati_experiment.md       # Gujarati VITS experiment report
│   ├── cross_language_analysis.md       # Cross-language statistical analysis
│   └── human_evaluation_guide.md        # Human evaluation methodology
│
├── g2p/                                 # Core G2P scripts & artifacts
│   ├── 2phonemepipeline.py              # Phoneme embedding pipeline (PPMI + UMAP)
│   ├── 2clustering.py                   # K-Means phoneme clustering
│   ├── 2viz.py                          # Clustering visualization
│   ├── task1_cluster_substitution.py    # Replace phonemes with cluster labels
│   ├── task1b_phonetic_folding.py       # Acoustic Phonetic Folding (smart clustering)
│   ├── phoneme_vocab.json               # 57 unique phonemes inventory
│   ├── phoneme_cluster_mapping.json     # Phoneme → cluster mapping (39 clusters)
│   ├── phoneme_cluster_mapping.csv      # Same mapping in CSV format
│   ├── clusters.json                    # Cluster definitions
│   ├── src_tokenizer.json               # Source (grapheme) tokenizer
│   ├── tgt_tokenizer.json               # Target (phoneme) tokenizer — baseline
│   └── tgt_tokenizer_clustered.json     # Target (cluster) tokenizer — clustered
│
├── models/                              # Trained G2P model weights
│   ├── best_g2p_transformer.weights.h5  # Baseline G2P model (57 phonemes)
│   └── g2p_clustered_model.weights.h5   # Clustered G2P model (39 clusters)
│
├── tts/                                 # VITS TTS pipeline scripts
│   ├── tts_data_download.py             # Download IndicTTS datasets from HuggingFace
│   ├── tts_audio_preprocess.py          # Audio normalization (22050 Hz, mono, trimmed)
│   ├── tts_split_data.py                # Train/val/test split & manifest generation
│   ├── tts_g2p_labeling.py              # G2P phoneme/cluster labeling for TTS
│   ├── tts_tokenizer.py                 # VITS tokenizer builder (baseline + clustered)
│   ├── vits_tokenizer.py                # Runtime tokenizer patching for Coqui TTS
│   ├── tts_vits_training.py             # VITS training configuration & launcher
│   ├── tts_inference.py                 # Paired inference (held-out + unseen sets)
│   ├── tts_automated_eval.py            # DNSMOS automated scoring
│   ├── tts_cross_language_analysis.py   # Cross-language statistical analysis
│   └── pre_train_check.py               # Pre-training validation checks
│
├── tts_app/                             # Web application (Flask)
│   ├── app.py                           # Flask server (human eval + live demo)
│   ├── synthesizer.py                   # GPU in-memory VITS model manager
│   ├── analysis.py                      # Offline statistical analysis
│   ├── templates/                       # HTML templates (base, demo, evaluate, results)
│   └── static/                          # CSS & JS (glassmorphism dark mode UI)
│
├── notebooks/                           # Jupyter/Colab notebooks
│   ├── Samsung_Pipeline.ipynb           # Data preparation pipeline
│   ├── Phase2_Baseline_G2P.ipynb        # Baseline Transformer training
│   ├── task2_clustered_g2p_training.ipynb # Clustered model training (Colab)
│   └── tts_vits_training.ipynb          # VITS training notebook (Colab)
│
├── results/                             # Evaluation outputs & reports
│   ├── evaluation_report.md             # Baseline vs Clustered G2P comparison
│   ├── comparison_table.csv             # G2P comparison metrics (CSV)
│   ├── vocab_reduction_report.md        # Vocabulary reduction analysis
│   ├── training_curves.png              # G2P training curves
│   ├── tts_hindi_female/                # Hindi VITS DNSMOS scores
│   ├── tts_marathi_female/              # Marathi VITS DNSMOS scores
│   ├── tts_gujarati_female/             # Gujarati VITS DNSMOS scores
│   ├── cross_language/                  # Cross-language analysis (figures + stats)
│   └── human_eval/                      # Human evaluation results
│
└── samples/                             # Generated VITS TTS audio samples
    ├── tts_hindi_female/                # Hindi (held-out + unseen, baseline + clustered)
    ├── tts_marathi_female/              # Marathi (held-out + unseen, baseline + clustered)
    └── tts_gujarati_female/             # Gujarati (held-out + unseen, baseline + clustered)
```

---

## Pipeline

### Phase 1: Data & G2P (Months 1–2)

| Step | Description | Script/Notebook |
|------|-------------|-----------------|
| 1 | **Data Preparation** — Download IndicTTS, normalize, tokenize, phonemize | `notebooks/Samsung_Pipeline.ipynb` |
| 2 | **Baseline G2P Model** — Transformer seq2seq (`<lang> + word → phonemes`) | `notebooks/Phase2_Baseline_G2P.ipynb` |
| 3 | **Phoneme Inventory** — Extract 57 unique phonemes across 3 languages | `g2p/2phonemepipeline.py` |
| 4 | **Phoneme Embeddings** — Co-occurrence PPMI + UMAP representations | `g2p/2phonemepipeline.py` |
| 5 | **Phoneme Clustering** — K-Means grouping + Acoustic Phonetic Folding | `g2p/2clustering.py`, `g2p/task1b_phonetic_folding.py` |

### Phase 2: Clustering & Retraining (Month 3)

| Step | Description | Script |
|------|-------------|--------|
| 6 | **Replace phonemes with cluster labels** in dataset | `g2p/task1_cluster_substitution.py` |
| 7 | **Retrain G2P** on clustered output (39 clusters) | `notebooks/task2_clustered_g2p_training.ipynb` |
| 8 | **Evaluation** — PER/WER comparison baseline vs clustered | `g2p/task3_evaluation.ipynb` |

### Phase 3: TTS Validation (Months 4–5)

| Step | Description | Script |
|------|-------------|--------|
| 9 | **TTS Data Download** — IndicTTS Hindi/Marathi/Gujarati from HuggingFace | `tts/tts_data_download.py` |
| 10 | **Audio Preprocessing** — 22050 Hz, mono, silence trimming | `tts/tts_audio_preprocess.py` |
| 11 | **Data Splitting** — Train/val/test manifests | `tts/tts_split_data.py` |
| 12 | **G2P Labeling** — Phoneme & cluster sequences for each utterance | `tts/tts_g2p_labeling.py` |
| 13 | **VITS Training** — 6 models (3 languages × baseline/clustered) | `tts/tts_vits_training.py` |
| 14 | **Inference** — Paired synthesis on held-out & unseen test sets | `tts/tts_inference.py` |
| 15 | **Automated Evaluation** — DNSMOS scoring (352 audio files) | `tts/tts_automated_eval.py` |
| 16 | **Cross-Language Analysis** — Statistical tests & publication figures | `tts/tts_cross_language_analysis.py` |
| 17 | **Human Evaluation** — Blind A/B web app + live TTS playground | `tts_app/` |

---

## Acoustic Phonetic Folding

The original K-Means clustering (K=12) was too aggressive — it merged all vowels into one cluster, producing unintelligible audio. We replaced it with **Acoustic Phonetic Folding**, a linguistically-informed approach that only merges:

- **Allophones & rare phonemes** → base form (`kq` → `k`, `f` → `ph`)
- **Long/short vowel pairs** → short form (`ii` → `i`, `uu` → `u`)
- **Regional variants** → standard form (`lx` → `l`, `sx` → `sh`)

This achieves **31.6% vocabulary reduction** (57 → 39) while keeping all critical phonetic distinctions intact.

---

## Languages Supported

- 🇮🇳 Hindi (`HI`) — ~15,623 entries
- 🇮🇳 Gujarati (`GU`) — ~16,676 entries
- 🇮🇳 Marathi (`MR`) — ~22,454 entries

---

## Tech Stack

Python · TensorFlow · PyTorch · Coqui TTS (VITS) · Scikit-learn · UMAP · Flask · Microsoft DNSMOS · NumPy · Matplotlib · Seaborn · SciPy · librosa

---

## Quick Start

```bash
# 1. Apply phonetic folding to generate cluster mapping
python g2p/task1b_phonetic_folding.py

# 2. Substitute phonemes with cluster labels in dataset
python g2p/task1_cluster_substitution.py

# 3. Train clustered model (on Google Colab with GPU)
#    Upload data/multilingual_g2p_clustered.txt to Drive
#    Run notebooks/task2_clustered_g2p_training.ipynb

# 4. Train VITS TTS models (requires GPU — see tts/ scripts)
python tts/tts_data_download.py --lang hi
python tts/tts_audio_preprocess.py --lang hi
python tts/tts_split_data.py --lang hi
python tts/tts_g2p_labeling.py --lang hi
python tts/tts_vits_training.py --lang hi --condition baseline

# 5. Run inference & evaluation
python tts/tts_inference.py --lang hi
python tts/tts_automated_eval.py --lang hi
python tts/tts_cross_language_analysis.py

# 6. Launch the web app (human eval + live demo)
python tts_app/app.py
# Open http://localhost:5000
```

---

## Dataset

[IndicTTS Dataset](https://www.iitm.ac.in/donlab/indictts/database) — Audio recordings + text transcripts for Indian languages, processed into pronunciation lexicons.

[SPRINGLab/IndicTTS (HuggingFace)](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) — Used for VITS TTS model training.
