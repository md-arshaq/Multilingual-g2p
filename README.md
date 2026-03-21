# 🔤 Multilingual Grapheme-to-Phoneme (G2P) with Phoneme Clustering

> **Samsung R&D Internship Project**

A multilingual G2P system for Indian languages (Hindi, Gujarati, Marathi) that converts written text into phoneme sequences, with phoneme clustering to reduce output vocabulary — a core component of TTS and ASR systems.

---

## 📂 Project Structure

```
├── docs/                              # Documentation & planning
│   ├── G2P_Project_Context.md
│   ├── PROJECT-PIPELINE.pdf
│   └── PROJECT_OVERVIEW.jpeg
│
├── notebooks/                         # Jupyter notebooks
│   ├── Samsung_Pipeline.ipynb         # Data preparation pipeline
│   └── Phase2_Baseline_G2P.ipynb      # Baseline Transformer model
│
├── data/                              # Datasets & pronunciation dictionaries
│   ├── g2p_hi.txt                     # Hindi
│   ├── g2p_gu.txt                     # Gujarati
│   ├── g2p_mr.txt                     # Marathi
│   └── multilingual_g2p_dataset.txt   # Combined multilingual dataset
│
├── models/                            # Trained model weights
│   └── best_g2p_transformer.weights.h5
│
└── results/                           # Analysis outputs
    ├── phoneme_frequency.csv
    └── phoneme_inventory.txt
```

---

## 🔬 Pipeline (8 Phases)

| Phase | Description |
|-------|-------------|
| 1 | **Data Preparation** — Download IndicTTS, extract, normalize, tokenize, phonemize |
| 2 | **Baseline G2P Model** — Transformer seq2seq (`<lang> + word → phonemes`) |
| 3 | **Phoneme Inventory Extraction** — Unique phonemes across all languages |
| 4 | **Phoneme Embeddings** — Vector representations capturing similarity |
| 5 | **Phoneme Clustering** — K-Means / Hierarchical grouping |
| 6 | **Replace with Cluster IDs** — Reduce vocabulary size |
| 7 | **Retrain G2P Model** — Train on clustered output tokens |
| 8 | **Evaluation** — Compare baseline vs. clustered model |

---

## 🌐 Languages Supported

- 🇮🇳 Hindi (`hi`)
- 🇮🇳 Gujarati (`gu`)
- 🇮🇳 Marathi (`mr`)

## 📊 Metrics

| Model | PER | WER | Output Vocab Size |
|-------|-----|-----|-------------------|
| Baseline | ~12% | ~18% | ~80 phonemes |
| Clustered | ~10% | ~16% | ~40 clusters |

---

## 🛠️ Tech Stack

Python · TensorFlow · PyTorch · Scikit-learn · NumPy · Pandas · Matplotlib · Phonemizer · Epitran

---

## 📖 Dataset

[IndicTTS Dataset](https://www.iitm.ac.in/donlab/indictts/database) — Audio recordings + text transcripts for Indian languages, processed into pronunciation lexicons.
