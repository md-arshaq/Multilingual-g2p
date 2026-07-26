# Multilingual Grapheme-to-Phoneme (G2P) with Phoneme Clustering

> **Samsung R&D Internship Project**

A multilingual G2P system for Indian languages (Hindi, Gujarati, Marathi) that converts written text into phoneme sequences, with **Acoustic Phonetic Folding** to reduce output vocabulary while preserving pronunciation quality — a core component of TTS and ASR systems.

---

##  Key Results

| Metric | Baseline | Clustered (Phonetic Folding) | Change |
|--------|----------|------------------------------|--------|
| **PER** | 0.03% | 0.08% | +0.05pp |
| **WER** | 0.22% | 0.35% | +0.13pp |
| **Output Vocab** | 57 phonemes | 39 clusters | **−31.6%** |
| **Parameters** | 1,422,141 | 1,417,515 | −4,626 |
| **MOS (Audio Quality)** | 4.77–4.80 | 4.70–4.80 | No significant difference (p=0.53) |

> Both models achieve >99.9% accuracy. The clustered model reduces vocabulary by 31.6% with **no statistically significant loss** in audio quality.

---

##  Project Structure

```
├── data/                                # Datasets & pronunciation dictionaries
│   ├── g2p_hi.txt                       # Hindi pronunciation lexicon
│   ├── g2p_gu.txt                       # Gujarati pronunciation lexicon
│   ├── g2p_mr.txt                       # Marathi pronunciation lexicon
│   ├── multilingual_g2p_dataset.txt     # Combined multilingual dataset (54,753 entries)
│   └── multilingual_g2p_clustered.txt   # Clustered version (phonetic folding applied)
│
├── docs/                                # Documentation & planning
│   ├── G2P_Project_Context.md           # Project context & background
│   ├── PROJECT-PIPELINE.pdf             # Pipeline overview
│   ├── PROJECT_OVERVIEW.jpeg            # Architecture diagram
│   └── tts_model_selection.md           # TTS model comparison & selection rationale
│
├── g2p/                                 # Core G2P scripts & artifacts
│   ├── 2phonemepipeline.py              # Phoneme embedding pipeline (PPMI + UMAP)
│   ├── 2clustering.py                   # K-Means phoneme clustering
│   ├── 2viz.py                          # Clustering visualization
│   ├── task1_cluster_substitution.py    # Task 1: Replace phonemes with cluster labels
│   ├── task1b_phonetic_folding.py       # Acoustic Phonetic Folding (smart clustering)
│   ├── task5_tts_data_prep.py           # Task 5: TTS data preparation
│   ├── task6_g2p_tts_pipeline.py        # Task 6: G2P → TTS end-to-end pipeline
│   ├── task7_mos_evaluation.py          # Task 7: MOS evaluation scoring
│   ├── task8_mos_visualization.py       # Task 8: MOS visualization & statistical tests
│   ├── task9_correlation_analysis.py    # Task 9: PER ↔ MOS correlation analysis
│   ├── phoneme_vocab.json               # 57 unique phonemes inventory
│   ├── phoneme_cluster_mapping.json     # Phoneme → cluster mapping (39 clusters)
│   ├── src_tokenizer.json               # Source (grapheme) tokenizer
│   ├── tgt_tokenizer.json               # Target (phoneme) tokenizer — baseline
│   └── tgt_tokenizer_clustered.json     # Target (cluster) tokenizer — clustered
│
├── models/                              # Trained model weights
│   ├── best_g2p_transformer.weights.h5  # Baseline G2P model (57 phonemes)
│   └── g2p_clustered_model.weights.h5   # Clustered G2P model (39 clusters)
│
├── notebooks/                           # Jupyter/Colab notebooks
│   ├── Samsung_Pipeline.ipynb           # Data preparation pipeline
│   ├── Phase2_Baseline_G2P.ipynb        # Baseline Transformer training
│   └── task2_clustered_g2p_training.ipynb# Clustered model training (Colab)
│
├── results/                             # Evaluation outputs & reports
│   ├── evaluation_report.md             # Baseline vs Clustered comparison
│   ├── comparison_table.csv             # Comparison metrics (CSV)
│   ├── vocab_reduction_report.md        # Vocabulary reduction analysis
│   ├── mos_scores.csv                   # Per-sample MOS scores
│   ├── mos_report.md                    # MOS evaluation summary
│   ├── mos_comparison.png               # MOS bar chart (baseline vs clustered)
│   ├── correlation_analysis.md          # PER ↔ MOS correlation report
│   ├── per_vs_mos_scatter.png           # Correlation scatter plot
│   └── training_curves.png              # Clustered model training curves
│
└── samples/                             # Generated TTS audio samples
    ├── hi/                              # Hindi (baseline + clustered)
    ├── gu/                              # Gujarati (baseline + clustered)
    └── mr/                              # Marathi (baseline + clustered)
```

---

##  Pipeline (9 Tasks)

### Month 1–2: Data & Baseline
| Phase | Description | Script/Notebook |
|-------|-------------|-----------------|
| 1 | **Data Preparation** — Download IndicTTS, normalize, tokenize, phonemize | `notebooks/Samsung_Pipeline.ipynb` |
| 2 | **Baseline G2P Model** — Transformer seq2seq (`<lang> + word → phonemes`) | `notebooks/Phase2_Baseline_G2P.ipynb` |
| 3 | **Phoneme Inventory** — Extract 57 unique phonemes across 3 languages | `g2p/2phonemepipeline.py` |
| 4 | **Phoneme Embeddings** — Co-occurrence PPMI + UMAP representations | `g2p/2phonemepipeline.py` |
| 5 | **Phoneme Clustering** — K-Means grouping + Acoustic Phonetic Folding | `g2p/2clustering.py`, `g2p/task1b_phonetic_folding.py` |

### Month 3: Clustering, TTS & Evaluation
| Task | Description | Script |
|------|-------------|--------|
| 1 | **Replace phonemes with cluster labels** in dataset | `g2p/task1_cluster_substitution.py` |
| 2 | **Retrain G2P** on clustered output (39 clusters) | `notebooks/task3_evaluation.ipynb` |
| 3 | **Evaluation** — PER/WER comparison baseline vs clustered | Run on Colab (see `g2p/task3_evaluation.ipynb`) |
| 4 | **TTS Model Selection** — Research & recommend | `docs/tts_model_selection.md` |
| 5 | **TTS Data Preparation** | `g2p/task5_tts_data_prep.py` |
| 6 | **G2P → TTS Pipeline** — End-to-end audio synthesis | `g2p/task6_g2p_tts_pipeline.py` |
| 7 | **MOS Evaluation** — Automated audio quality scoring | `g2p/task7_mos_evaluation.py` |
| 8 | **MOS Visualization** — Charts & statistical tests | `g2p/task8_mos_visualization.py` |
| 9 | **PER ↔ MOS Correlation** — Analyze quality vs accuracy | `g2p/task9_correlation_analysis.py` |

---

##  Acoustic Phonetic Folding

The original K-Means clustering (K=12) was too aggressive — it merged all vowels into one cluster, producing unintelligible audio. We replaced it with **Acoustic Phonetic Folding**, a linguistically-informed approach that only merges:

- **Allophones & rare phonemes** → base form (`kq` → `k`, `f` → `ph`)
- **Long/short vowel pairs** → short form (`ii` → `i`, `uu` → `u`)
- **Regional variants** → standard form (`lx` → `l`, `sx` → `sh`)

This achieves **31.6% vocabulary reduction** (57 → 39) while keeping all critical phonetic distinctions intact.

---

##  Languages Supported

- 🇮🇳 Hindi (`HI`) — ~30,000 entries
- 🇮🇳 Gujarati (`GU`) — ~12,000 entries
- 🇮🇳 Marathi (`MR`) — ~12,000 entries

---

##  Tech Stack

Python · TensorFlow · Scikit-learn · NumPy · Matplotlib · librosa · gTTS · editdistance

---

##  Quick Start

```bash
# 1. Apply phonetic folding to generate cluster mapping
python g2p/task1b_phonetic_folding.py

# 2. Substitute phonemes with cluster labels in dataset
python g2p/task1_cluster_substitution.py

# 3. Train clustered model (on Google Colab with GPU)
#    Upload data/multilingual_g2p_clustered.txt to Drive
#    Run notebooks/task2_clustered_g2p_training.ipynb

# 4. Generate TTS audio samples (baseline vs clustered)
python g2p/task6_g2p_tts_pipeline.py --use_gtts

# 5. Run MOS evaluation pipeline
python g2p/task7_mos_evaluation.py
python g2p/task8_mos_visualization.py
python g2p/task9_correlation_analysis.py
```

---

##  Dataset

[IndicTTS Dataset](https://www.iitm.ac.in/donlab/indictts/database) — Audio recordings + text transcripts for Indian languages, processed into pronunciation lexicons.
