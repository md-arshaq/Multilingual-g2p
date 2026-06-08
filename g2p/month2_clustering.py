"""
Month 2 Tasks 4-6: Phoneme Clustering, Analysis & Mapping
─────────────────────────────────────────────────────────
Run AFTER month2_phoneme_pipeline.py
Requires: phoneme_embeddings_learned.npy, phoneme_vocab.json

Outputs:
  phoneme_clusters.png          ← UMAP scatter + cluster table
  silhouette_scores.png         ← K selection chart
  phoneme_cluster_mapping.json  ← phoneme → cluster_id + label
  phoneme_cluster_mapping.csv   ← flat CSV version
"""

import json, csv, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — change paths if needed
# ─────────────────────────────────────────────
EMBEDDINGS_PATH = "phoneme_embeddings_learned.npy"
VOCAB_PATH      = "phoneme_vocab.json"
CHOSEN_K        = 12    # best by silhouette score on your data

# ─────────────────────────────────────────────
# STEP 4: K-MEANS CLUSTERING
# ─────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

unique_phonemes = [vocab["id_to_phoneme"][str(i)] for i in range(vocab["vocab_size"])]
phoneme_counts  = vocab["phoneme_counts"]

# ── Find best K using silhouette score ────────
print("Evaluating silhouette scores for K = 5 to 50...")
ks, scores = [], []
for k in range(5, 51, 5):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(embeddings)
    sc  = silhouette_score(embeddings, lbl)
    ks.append(k); scores.append(sc)
    print(f"  k={k:3d}  silhouette={sc:.4f}")

best_k = ks[scores.index(max(scores))]
print(f"\n→ Best K by silhouette: {best_k}  |  Chosen K: {CHOSEN_K}")

# ── Run final clustering ──────────────────────
km     = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=20)
labels = km.fit_predict(embeddings)

clusters = {}
for p, l in zip(unique_phonemes, labels):
    clusters.setdefault(int(l), []).append(p)

# ─────────────────────────────────────────────
# STEP 5: ANALYZE CLUSTERS
# ─────────────────────────────────────────────

# Linguistic category heuristics — maps known phoneme sets to names
LINGUISTIC_LABELS = {
    frozenset(['a','ee','hq','i','ii','o','rq','u']):             "Short Vowels",
    frozenset(['aa','uu','ei','ou','ae','ax']):                   "Long/Diphthong Vowels",
    frozenset(['k','kh','g','gh','kq','khq','gq']):               "Velar Stops",
    frozenset(['t','th','tx','txh','d','dh','dx','dxh','dxq','dxhq']): "Dental/Retroflex Stops",
    frozenset(['p','ph','b','bh']):                               "Bilabial Stops",
    frozenset(['ch','c','j','jh']):                               "Affricates",
    frozenset(['s','sh','sx','z','f']):                           "Fricatives",
    frozenset(['m','n','nx','ng','mq','nj']):                     "Nasals",
    frozenset(['r','rq','l','lx','w','y']):                       "Liquids/Semivowels",
    frozenset(['h','hq']):                                        "Glottals",
    frozenset(['q']):                                             "Special/Nasal Marker",
}

def get_label(phoneme_set):
    s = frozenset(phoneme_set)
    best_label, best_overlap = "Misc", 0
    for key, label in LINGUISTIC_LABELS.items():
        overlap = len(s & key)
        if overlap > best_overlap:
            best_overlap, best_label = overlap, label
    return best_label

cluster_labels = {cid: get_label(phones) for cid, phones in clusters.items()}

# ── Print analysis report ─────────────────────
print("\n" + "="*60)
print(f"CLUSTER ANALYSIS REPORT  (K={CHOSEN_K})")
print("="*60)
issues = []
for cid in sorted(clusters.keys()):
    phones     = sorted(clusters[cid])
    freqs      = {p: phoneme_counts.get(p, 0) for p in phones}
    total_freq = sum(freqs.values())
    most_common = max(freqs, key=freqs.get)

    flag = "⚠️  Singleton" if len(phones) == 1 else "✅"
    print(f"\nCluster {cid:2d} · {cluster_labels[cid]}  {flag}")
    print(f"  Phonemes  : {phones}")
    print(f"  Total freq: {total_freq:,}   |   Most frequent: {most_common} ({freqs[most_common]:,})")

    if len(phones) == 1:
        issues.append(f"C{cid} ({phones[0]}) is a singleton — rare phoneme, consider merging")

if issues:
    print("\n⚠️  Potential issues:")
    for iss in issues:
        print(f"   • {iss}")

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

COLORS = plt.cm.get_cmap("tab20", CHOSEN_K)

# ── Fig 1: Silhouette Score plot ──────────────
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(ks, scores, "o-", color="steelblue", linewidth=2, markersize=8)
ax1.axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1.5, label=f"Chosen K={CHOSEN_K}")
ax1.fill_between(ks, scores, alpha=0.1, color="steelblue")
ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
ax1.set_ylabel("Silhouette Score", fontsize=12)
ax1.set_title("Silhouette Score vs K — Phoneme Clustering", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("silhouette_scores.png", dpi=150)
print("\n✅ silhouette_scores.png saved")

# ── Fig 2: UMAP scatter + cluster table ───────
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
    emb2d = reducer.fit_transform(embeddings)
    dim_method = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA
    emb2d = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    dim_method = "PCA"

fig2, axes = plt.subplots(1, 2, figsize=(22, 9))

# Left: scatter
ax_scatter = axes[0]
for cid, phones in clusters.items():
    idxs = [unique_phonemes.index(p) for p in phones]
    xs, ys = emb2d[idxs, 0], emb2d[idxs, 1]
    ax_scatter.scatter(xs, ys, s=220, color=COLORS(cid), alpha=0.88,
                       edgecolors="white", linewidths=1.0, zorder=3)
    for p, x, y in zip(phones, xs, ys):
        ax_scatter.annotate(p, (x, y), fontsize=8.5, ha="center", va="center",
                            fontweight="bold",
                            color="white" if cid % 3 != 1 else "black")

patches = [mpatches.Patch(color=COLORS(cid),
           label=f"C{cid}: {cluster_labels[cid]}") for cid in sorted(clusters)]
ax_scatter.legend(handles=patches, fontsize=7.5, loc="upper left", framealpha=0.88)
ax_scatter.set_title(f"Phoneme Clusters ({dim_method} + K-Means, K={CHOSEN_K})",
                     fontsize=13, fontweight="bold")
ax_scatter.set_xlabel(f"{dim_method} Dim 1")
ax_scatter.set_ylabel(f"{dim_method} Dim 2")
ax_scatter.grid(True, alpha=0.2)

# Right: membership table
ax_table = axes[1]
ax_table.axis("off")
row_labels = [f"C{cid} · {cluster_labels[cid]}" for cid in sorted(clusters.keys())]
cell_text  = [", ".join(sorted(clusters[cid]))    for cid in sorted(clusters.keys())]
row_colors = [COLORS(cid)                          for cid in sorted(clusters.keys())]

table = ax_table.table(
    cellText=[[t] for t in cell_text],
    rowLabels=row_labels,
    rowColours=row_colors,
    colLabels=["Phonemes in cluster"],
    loc="center", cellLoc="left"
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.65)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e1e2e")
        cell.set_text_props(color="white", fontweight="bold")
ax_table.set_title("Cluster Membership", fontsize=13, fontweight="bold", pad=18)

plt.suptitle("Multilingual G2P Phoneme Clustering — Hindi + Gujarati + Marathi",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("phoneme_clusters.png", dpi=150, bbox_inches="tight")
print("✅ phoneme_clusters.png saved")

# ─────────────────────────────────────────────
# STEP 6: CREATE MAPPING FILES
# ─────────────────────────────────────────────

# ── JSON ──────────────────────────────────────
phoneme_mapping = {}
for p, lbl in zip(unique_phonemes, labels):
    cid = int(lbl)
    phoneme_mapping[p] = {
        "cluster_id":      cid,
        "cluster_label":   cluster_labels[cid],
        "cluster_members": sorted(clusters[cid]),
        "frequency":       phoneme_counts.get(p, 0)
    }

with open("phoneme_cluster_mapping.json", "w", encoding="utf-8") as f:
    json.dump(phoneme_mapping, f, indent=2, ensure_ascii=False)
print("✅ phoneme_cluster_mapping.json saved")

# ── CSV ───────────────────────────────────────
fieldnames = ["phoneme", "cluster_id", "cluster_label", "cluster_members", "frequency"]
with open("phoneme_cluster_mapping.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for p, info in sorted(phoneme_mapping.items()):
        writer.writerow({
            "phoneme":         p,
            "cluster_id":      info["cluster_id"],
            "cluster_label":   info["cluster_label"],
            "cluster_members": " | ".join(info["cluster_members"]),
            "frequency":       info["frequency"]
        })
print("✅ phoneme_cluster_mapping.csv saved")

# ── Final summary ─────────────────────────────
print("\n" + "="*60)
print("ALL OUTPUTS SAVED")
print("="*60)
print("  phoneme_clusters.png")
print("  silhouette_scores.png")
print("  phoneme_cluster_mapping.json")
print("  phoneme_cluster_mapping.csv")
print(f"\n  Total phonemes clustered : {len(unique_phonemes)}")
print(f"  K used                   : {CHOSEN_K}")
print(f"  Dim reduction method     : {dim_method}")
