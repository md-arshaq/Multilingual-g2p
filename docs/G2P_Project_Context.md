# Project Context: Multilingual Grapheme-to-Phoneme (G2P) with Phoneme Clustering

## What This Project Does
This project builds a **multilingual Grapheme-to-Phoneme (G2P) system** for Indian languages (Hindi, Gujarati, Marathi), then improves it by **clustering similar phonemes** to reduce the output vocabulary. G2P models convert written text into phoneme sequences — a core component of Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) systems.

---

## Problem Being Solved
1. Most G2P systems train a **separate model per language** — this project explores one unified multilingual model.
2. Many phonemes across languages represent the **same sound** but are treated as distinct units, inflating the vocabulary unnecessarily.

**Example:** Hindi `क`, Gujarati `ક`, and Marathi `क` all produce `/k/` but are typically treated as separate phonemes.

---

## Languages
- Hindi (`hi`)
- Gujarati (`gu`)
- Marathi (`mr`)

---

## Dataset
**IndicTTS Dataset** — https://www.iitm.ac.in/donlab/indictts/database

Contains audio recordings + text transcripts for multiple Indian languages. Processed into a pronunciation lexicon:

```
word → phoneme sequence
```

### Final Dataset Format

| word | phonemes | language |
|------|----------|----------|
| भारत | b aa r ax t | hi |
| ભારત | b aa r ax t | gu |
| भारत | b aa r ax t | mr |

---

## Pipeline (8 Phases)

### Phase 1 — Data Preparation
Download IndicTTS → extract transcripts → normalize/clean → tokenize → phonemize → build pronunciation dictionary → merge multilingual dictionaries with language tags → train/val/test split.

### Phase 2 — Baseline Multilingual G2P Model
Train a sequence-to-sequence model (Transformer or BiLSTM with attention).

- **Input:** `<language_tag> + word` (e.g., `<hi> भारत`)
- **Output:** phoneme sequence (e.g., `b aa r ax t`)
- **Metrics:** Phoneme Error Rate (PER), Word Error Rate (WER)

### Phase 3 — Phoneme Inventory Extraction
Extract all unique phonemes across all languages.

Example inventory: `{p, b, t, d, k, g, kh, gh, aa, ax, e, i, o}`

### Phase 4 — Phoneme Embeddings
Represent each phoneme as a numeric vector to capture similarity.

Example: `k → [0.32, 0.41, -0.18]`, `g → [0.35, 0.38, -0.17]`

### Phase 5 — Phoneme Clustering
Cluster similar phonemes using K-Means or Hierarchical Clustering.

| Cluster | Phonemes |
|---------|----------|
| C1 | k, g, kh |
| C2 | p, b |
| C3 | t, d |

### Phase 6 — Replace Phonemes with Cluster IDs
Convert phoneme outputs to cluster labels to reduce vocabulary size.

- Original: `b aa r ax t`
- Clustered: `C2 C8 C5 C7 C4`

### Phase 7 — Retrain G2P Model
Retrain the model using cluster IDs as output tokens. Expected benefits: smaller output space, faster training, better cross-language generalization.

### Phase 8 — Evaluation
Compare baseline vs. clustered model.

| Model | PER | WER | Output Vocab Size |
|-------|-----|-----|-------------------|
| Baseline | ~12% | ~18% | ~80 phonemes |
| Clustered | ~10% | ~16% | ~40 clusters |

Additional metrics: model size, training time, cluster visualizations.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| ML Frameworks | PyTorch, TensorFlow |
| Data & Math | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, UMAP, t-SNE |
| Speech/Phoneme Tools | Phonemizer, Epitran, Montreal Forced Aligner |

---

## Key Terms

| Term | Meaning |
|------|---------|
| Grapheme | Written character |
| Phoneme | Smallest unit of sound |
| G2P | Grapheme-to-Phoneme conversion |
| Phoneme Inventory | Complete set of phonemes in a dataset |
| Embedding | Vector (numeric) representation of a symbol |
| Clustering | Grouping similar items together |
| PER | Phoneme Error Rate |
| WER | Word Error Rate |

---

## Expected Deliverables
1. A multilingual G2P model supporting Hindi, Gujarati, and Marathi
2. A phoneme clustering method that reduces output vocabulary
3. Quantitative comparison: baseline vs. clustered model
4. Visualizations of phoneme clusters (UMAP/t-SNE)
5. Reproducible pipeline and documentation
