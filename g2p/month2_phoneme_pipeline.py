"""
Month 2 Pipeline: Phoneme Extraction → Embeddings → Dimensionality Reduction
Dataset format: <LANG_TAG> grapheme\tphonemized_sequence
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────
# STEP 1: EXTRACT PHONEMES
# ─────────────────────────────────────────────

DATASET_PATH = "multilingual_g2p_dataset.txt"   # ← update path if needed
TGT_TOKENIZER_PATH = "tgt_tokenizer.json"

def load_dataset(path):
    """Load multilingual G2P dataset. Returns list of (lang, grapheme, phonemes)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Format: <LANG> grapheme\tphoneme1 phoneme2 ...
                parts = line.split("\t")
                left  = parts[0].strip()          # "<HI> अंक"
                right = parts[1].strip()           # "a q k a"

                lang_end = left.index(">") + 1
                lang     = left[:lang_end].strip() # "<HI>"
                grapheme = left[lang_end:].strip() # "अंक"
                phonemes = right.split()           # ["a", "q", "k", "a"]

                records.append((lang, grapheme, phonemes))
            except Exception:
                continue
    return records

print("Loading dataset...")
records = load_dataset(DATASET_PATH)
print(f"Total records: {len(records)}")

# Count per language
lang_counts = Counter(r[0] for r in records)
print("Language distribution:", dict(lang_counts))

# Extract all unique phonemes across all languages
all_phonemes = []
for _, _, phonemes in records:
    all_phonemes.extend(phonemes)

phoneme_counts   = Counter(all_phonemes)
unique_phonemes  = sorted(phoneme_counts.keys())  # sorted for reproducibility

print(f"\nTotal unique phonemes: {len(unique_phonemes)}")
print("Phoneme vocabulary:", unique_phonemes)

# ── Save phoneme vocabulary ──────────────────
phoneme_vocab = {
    "phoneme_to_id": {p: i for i, p in enumerate(unique_phonemes)},
    "id_to_phoneme": {i: p for i, p in enumerate(unique_phonemes)},
    "vocab_size": len(unique_phonemes),
    "phoneme_counts": dict(phoneme_counts)
}

with open("phoneme_vocab.json", "w", encoding="utf-8") as f:
    json.dump(phoneme_vocab, f, ensure_ascii=False, indent=2)

print("\n✅ Step 1 done → phoneme_vocab.json saved")


# ─────────────────────────────────────────────
# STEP 2: CREATE PHONEME EMBEDDINGS
# ─────────────────────────────────────────────

phoneme_to_id = phoneme_vocab["phoneme_to_id"]
VOCAB_SIZE    = phoneme_vocab["vocab_size"]

# ── Option A: One-Hot (Baseline) ─────────────
def one_hot_embeddings(phoneme_to_id):
    vocab_size = len(phoneme_to_id)
    embeddings = {}
    for phoneme, idx in phoneme_to_id.items():
        vec = np.zeros(vocab_size)
        vec[idx] = 1.0
        embeddings[phoneme] = vec
    return embeddings

one_hot_emb = one_hot_embeddings(phoneme_to_id)
print(f"\nOne-Hot embedding shape: {one_hot_emb['a'].shape}")


# ── Option B: Learned Embeddings (Better) ─────
# We build co-occurrence context vectors — no neural net needed,
# gives meaningful geometry (phonemes that appear together cluster together).

def build_cooccurrence_embeddings(records, phoneme_to_id, window=2):
    """
    For each phoneme, build a context vector based on neighbouring phonemes.
    This is a lightweight 'learned' embedding without a neural network.
    """
    vocab_size = len(phoneme_to_id)
    cooc_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for _, _, phonemes in records:
        for i, p in enumerate(phonemes):
            if p not in phoneme_to_id:
                continue
            pid = phoneme_to_id[p]
            for j in range(max(0, i - window), min(len(phonemes), i + window + 1)):
                if j == i:
                    continue
                neighbor = phonemes[j]
                if neighbor in phoneme_to_id:
                    cooc_matrix[pid][phoneme_to_id[neighbor]] += 1

    # PPMI (Positive Pointwise Mutual Information) — standard NLP trick
    total = cooc_matrix.sum()
    row_sums = cooc_matrix.sum(axis=1, keepdims=True) + 1e-9
    col_sums = cooc_matrix.sum(axis=0, keepdims=True) + 1e-9
    expected = (row_sums @ col_sums) / total
    ppmi = np.maximum(np.log(cooc_matrix / (expected + 1e-9) + 1e-9), 0)

    return ppmi  # shape: (vocab_size, vocab_size)

print("\nBuilding co-occurrence (learned) embeddings...")
learned_matrix = build_cooccurrence_embeddings(records, phoneme_to_id, window=2)
print(f"Learned embedding matrix shape: {learned_matrix.shape}")

# Store phoneme → vector dict
learned_embeddings = {
    phoneme: learned_matrix[idx]
    for phoneme, idx in phoneme_to_id.items()
}

# Save embedding matrix
np.save("phoneme_embeddings_onehot.npy", np.array([one_hot_emb[p] for p in unique_phonemes]))
np.save("phoneme_embeddings_learned.npy", learned_matrix)
print("✅ Step 2 done → phoneme_embeddings_onehot.npy & phoneme_embeddings_learned.npy saved")


# ─────────────────────────────────────────────
# STEP 3: DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────

# We'll use the learned embeddings (more meaningful than one-hot)
embedding_matrix = learned_matrix   # shape: (vocab_size, vocab_size)

# ── Option A: UMAP (Preferred) ───────────────
try:
    import umap
    print("\nRunning UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
    embeddings_2d_umap = reducer.fit_transform(embedding_matrix)
    method_used = "UMAP"
    embeddings_2d = embeddings_2d_umap
    np.save("phoneme_2d_umap.npy", embeddings_2d_umap)
    print("✅ UMAP done → phoneme_2d_umap.npy saved")

except ImportError:
    print("UMAP not installed → falling back to PCA")

    # ── Option B: PCA (Backup) ───────────────────
    from sklearn.decomposition import PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embedding_matrix)
    method_used = "PCA"
    np.save("phoneme_2d_pca.npy", embeddings_2d)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print("✅ PCA done → phoneme_2d_pca.npy saved")


# ─────────────────────────────────────────────
# VISUALIZE: 2D Phoneme Cluster Plot
# ─────────────────────────────────────────────

# Load tgt_tokenizer to cross-check our vocab
with open(TGT_TOKENIZER_PATH, "r") as f:
    tgt_tok = json.load(f)

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=80, alpha=0.7, color="steelblue")

for i, phoneme in enumerate(unique_phonemes):
    ax.annotate(
        phoneme,
        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
        fontsize=9, ha="center", va="bottom",
        color="darkred"
    )

ax.set_title(f"Phoneme Clusters ({method_used}) — Hindi + Gujarati + Marathi", fontsize=14)
ax.set_xlabel(f"{method_used} Dimension 1")
ax.set_ylabel(f"{method_used} Dimension 2")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("phoneme_clusters.png", dpi=150)
plt.show()
print("✅ Plot saved → phoneme_clusters.png")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("MONTH 2 PIPELINE SUMMARY")
print("="*50)
print(f"  Total records         : {len(records)}")
print(f"  Languages             : {list(lang_counts.keys())}")
print(f"  Unique phonemes       : {VOCAB_SIZE}")
print(f"  Embedding method      : Co-occurrence PPMI + {method_used}")
print(f"  Output files:")
print(f"    phoneme_vocab.json")
print(f"    phoneme_embeddings_onehot.npy")
print(f"    phoneme_embeddings_learned.npy")
print(f"    phoneme_2d_{method_used.lower()}.npy")
print(f"    phoneme_clusters.png")