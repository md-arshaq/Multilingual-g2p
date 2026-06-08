"""
Month 2  |  Tasks 7-9: Visualization, Validation & Documentation
────────────────────────────────────────────────────────────────
Run AFTER:  month2_phoneme_pipeline.py  &  month2_clustering.py

Requires:
  phoneme_embeddings_learned.npy
  phoneme_vocab.json
  emb2d.npy              (UMAP 2D coords)
  labels.npy             (cluster label per phoneme)
  clusters.json          (cluster_id → [phonemes])

pip install numpy matplotlib scikit-learn umap-learn scipy adjustText

Outputs:
  viz_cluster_map.png    ← Presentation-quality 2D cluster map
  viz_validation.png     ← 5-panel validation dashboard
  month2_documentation.md
"""

import numpy as np, json, warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from adjustText import adjust_text
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────
emb2d  = np.load("emb2d.npy")
labels = np.load("labels.npy")
with open("clusters.json")      as f: clusters = {int(k): v for k,v in json.load(f).items()}
with open("phoneme_vocab.json") as f: vocab    = json.load(f)

unique_phonemes = [vocab["id_to_phoneme"][str(i)] for i in range(vocab["vocab_size"])]
phoneme_counts  = vocab["phoneme_counts"]

CLUSTER_INFO = {
    0:  ("Short Vowels",            "#4E9AF1", "vowel"),
    2:  ("Common Consonants",       "#F4845F", "consonant"),
    3:  ("Stops & Aspirates",       "#E85D75", "consonant"),
    7:  ("Dental / Liquids",        "#F7B731", "consonant"),
    5:  ("Diphthongs",              "#A29BFE", "vowel"),
    9:  ("Long Vowels",             "#74B9FF", "vowel"),
    1:  ("Rare Velar (kq)",         "#B2BEC3", "rare"),
    4:  ("Fricative (f)",           "#55EFC4", "rare"),
    6:  ("Aspirate (ph)",           "#FDCB6E", "rare"),
    8:  ("Rare Velar (gq)",         "#9EA7AD", "rare"),
    10: ("Schwa (ax)",              "#81ECEC", "rare"),
    11: ("Retroflex Sib. (sx)",     "#FAB1A0", "rare"),
}
DARK = "#0F1117"; CARD = "#1A1D27"; WHITE = "white"; GRID = "#222"


# ═══════════════════════════════════════════════════════════════════════
# TASK 7: VISUALIZATION — Main 2D Cluster Map
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

# Convex hull regions
for cid, phones in clusters.items():
    idxs  = [unique_phonemes.index(p) for p in phones]
    pts   = emb2d[idxs]
    color = CLUSTER_INFO[cid][1]
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            hx = pts[hull.vertices, 0]; hy = pts[hull.vertices, 1]
            cx, cy = hx.mean(), hy.mean()
            hx = cx + (hx - cx) * 1.4
            hy = cy + (hy - cy) * 1.4
            ax.fill(hx, hy, color=color, alpha=0.11, zorder=1)
            ax.plot(np.append(hx, hx[0]), np.append(hy, hy[0]),
                    color=color, alpha=0.3, linewidth=1.0, linestyle="--", zorder=2)
        except: pass

# Scatter points
for cid, phones in clusters.items():
    idxs = [unique_phonemes.index(p) for p in phones]
    xs, ys = emb2d[idxs, 0], emb2d[idxs, 1]
    color  = CLUSTER_INFO[cid][1]
    kind   = CLUSTER_INFO[cid][2]
    marker = "o" if kind == "vowel" else ("s" if kind == "consonant" else "^")
    sz     = 300 if kind != "rare" else 160
    ax.scatter(xs, ys, s=sz, c=color, marker=marker,
               edgecolors="white", linewidths=0.7, alpha=0.92, zorder=4)

# Phoneme labels
texts = []
for i, p in enumerate(unique_phonemes):
    cid = int(labels[i])
    t = ax.text(emb2d[i, 0], emb2d[i, 1] + 0.06, p,
                fontsize=8.5, fontweight="bold", color="white",
                ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc=CLUSTER_INFO[cid][1], ec="none", alpha=0.75))
    texts.append(t)

adjust_text(texts, ax=ax, expand_text=(1.2, 1.3),
            arrowprops=dict(arrowstyle="-", color="#555", lw=0.4))

# Cluster banners for large clusters
BANNER = {
    0: (-0.7, 0.6),  2: (0.5, 0.45),  3: (0.4, -0.55),
    7: (-0.3, -0.55), 5: (-0.5, 0.5), 9: (-0.5, -0.42),
}
for cid in [0, 2, 3, 7, 5, 9]:
    phones = clusters[cid]
    idxs   = [unique_phonemes.index(p) for p in phones]
    cx     = emb2d[idxs, 0].mean()
    cy     = emb2d[idxs, 1].mean()
    dx, dy = BANNER[cid]
    color  = CLUSTER_INFO[cid][1]
    ax.annotate(f"● {CLUSTER_INFO[cid][0]}",
                xy=(cx, cy), xytext=(cx + dx, cy + dy),
                fontsize=9.5, color=color, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.35", fc=CARD, ec=color, lw=1.3, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0), zorder=7)

legend_elements = [
    Line2D([0],[0], marker="o", color="w", label="Vowel cluster",    markerfacecolor="#4E9AF1", markersize=11),
    Line2D([0],[0], marker="s", color="w", label="Consonant cluster", markerfacecolor="#E85D75", markersize=11),
    Line2D([0],[0], marker="^", color="w", label="Rare / singleton",  markerfacecolor="#B2BEC3", markersize=11),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
          facecolor=CARD, edgecolor="#555", labelcolor=WHITE, framealpha=0.9)
ax.set_title("Phoneme Cluster Map  —  Hindi · Gujarati · Marathi\n"
             "UMAP Projection  +  K-Means Clustering  (K = 12)  |  57 Phonemes → 12 Groups",
             fontsize=13, fontweight="bold", color=WHITE, pad=16)
ax.set_xlabel("UMAP Dimension 1", color="#AAAAAA", fontsize=11)
ax.set_ylabel("UMAP Dimension 2", color="#AAAAAA", fontsize=11)
ax.tick_params(colors="#555")
for spine in ax.spines.values(): spine.set_edgecolor("#2a2a2a")
ax.grid(True, color="#1e1e1e", linewidth=0.6)
plt.tight_layout()
plt.savefig("viz_cluster_map.png", dpi=180, bbox_inches="tight", facecolor=DARK)
print("✅ viz_cluster_map.png saved")


# ═══════════════════════════════════════════════════════════════════════
# TASK 8: VALIDATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 10))
fig2.patch.set_facecolor(DARK)
gs   = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38)

def style_ax(a, title):
    a.set_facecolor(CARD)
    a.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    a.tick_params(colors="#888")
    for sp in a.spines.values(): sp.set_edgecolor("#333")
    a.grid(True, color=GRID, linewidth=0.5)

# Panel A: Before vs After
ax_a = fig2.add_subplot(gs[0, 0])
style_ax(ax_a, "A  Reduction: Phonemes → Clusters")
vals = [len(unique_phonemes), len(clusters)]
bars = ax_a.bar(["Original\nPhonemes","Clustered\nGroups"], vals,
                color=["#4E9AF1","#E85D75"], width=0.5, edgecolor="white", linewidth=0.7)
for bar, val in zip(bars, vals):
    ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
              str(val), ha="center", va="bottom", color=WHITE, fontsize=13, fontweight="bold")
ax_a.set_ylim(0, max(vals)+8)
ax_a.set_ylabel("Count", color="#888")
pct = (1 - vals[1]/vals[0]) * 100
ax_a.text(0.5, 0.88, f"↓ {pct:.0f}% reduction", transform=ax_a.transAxes,
          ha="center", color="#55EFC4", fontsize=11, fontweight="bold")

# Panel B: Cluster size distribution
ax_b = fig2.add_subplot(gs[0, 1])
style_ax(ax_b, "B  Cluster Size Distribution")
cids   = sorted(clusters.keys())
sizes  = [len(clusters[c]) for c in cids]
bcolors= [CLUSTER_INFO[c][1] for c in cids]
ax_b.bar([f"C{c}" for c in cids], sizes, color=bcolors, edgecolor="white", linewidth=0.5)
ax_b.tick_params(axis="x", rotation=45, labelsize=7.5)
ax_b.set_ylabel("Number of Phonemes", color="#888")

# Panel C: Frequency per cluster
ax_c = fig2.add_subplot(gs[0, 2])
style_ax(ax_c, "C  Total Frequency per Cluster")
freq_sums    = {cid: sum(phoneme_counts.get(p,0) for p in phones) for cid, phones in clusters.items()}
sorted_cids  = sorted(freq_sums, key=freq_sums.get, reverse=True)
ax_c.barh([f"C{c}" for c in sorted_cids],
          [freq_sums[c] for c in sorted_cids],
          color=[CLUSTER_INFO[c][1] for c in sorted_cids], edgecolor="white", linewidth=0.4)
ax_c.set_xlabel("Total phoneme occurrences", color="#888", fontsize=9)
ax_c.tick_params(axis="y", labelsize=8, labelcolor=WHITE)
ax_c.invert_yaxis()

# Panel D: Type pie
ax_d = fig2.add_subplot(gs[1, 0])
ax_d.set_facecolor(CARD)
ax_d.set_title("D  Phoneme Type Distribution", color=WHITE, fontsize=11, fontweight="bold", pad=10)
v = sum(len(clusters[c]) for c in [0,5,9])
c = sum(len(clusters[c]) for c in [2,3,7])
r = sum(len(clusters[c]) for c in [1,4,6,8,10,11])
wedges, texts, pcts = ax_d.pie([v,c,r],
    labels=["Vowels","Consonants","Rare/Special"],
    colors=["#4E9AF1","#E85D75","#B2BEC3"], autopct="%1.0f%%", startangle=90,
    textprops={"color": WHITE, "fontsize": 9},
    wedgeprops={"edgecolor": CARD, "linewidth": 2})
for pct in pcts: pct.set_color(WHITE)

# Panel E: Top-10 frequency bar
ax_e = fig2.add_subplot(gs[1, 1:])
ax_e.set_facecolor(CARD)
ax_e.set_title("E  Top 10 Phonemes by Frequency (coloured by cluster)", color=WHITE, fontsize=11, fontweight="bold", pad=10)
top10    = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
ph_names = [x[0] for x in top10]
ph_freqs = [x[1] for x in top10]
ph_colors= [CLUSTER_INFO[int(labels[unique_phonemes.index(p)])][1] for p in ph_names]
ax_e.bar(ph_names, ph_freqs, color=ph_colors, edgecolor="white", linewidth=0.6)
for name, freq in zip(ph_names, ph_freqs):
    ax_e.text(ph_names.index(name), freq+300, f"{freq:,}",
              ha="center", va="bottom", color=WHITE, fontsize=8.5)
ax_e.set_ylabel("Frequency", color="#888")
ax_e.tick_params(axis="x", labelsize=11, labelcolor=WHITE)
ax_e.tick_params(axis="y", colors="#888")
for sp in ax_e.spines.values(): sp.set_edgecolor("#333")
ax_e.grid(True, color=GRID, linewidth=0.5, axis="y")

fig2.suptitle("Phoneme Clustering  —  Validation Dashboard\n"
              "Hindi · Gujarati · Marathi  |  57 Phonemes  →  12 Clusters",
              fontsize=14, fontweight="bold", color=WHITE, y=1.01)
plt.savefig("viz_validation.png", dpi=180, bbox_inches="tight", facecolor=DARK)
print("✅ viz_validation.png saved")

print("\n─── Validation Numbers ───")
print(f"  Original phonemes : {len(unique_phonemes)}")
print(f"  Clusters          : {len(clusters)}")
print(f"  Reduction         : {(1 - len(clusters)/len(unique_phonemes))*100:.1f}%")
print(f"  Singleton clusters: {sum(1 for c in clusters.values() if len(c)==1)}")
print(f"  Largest cluster   : C{max(clusters, key=lambda c: len(clusters[c]))} ({max(len(v) for v in clusters.values())} phonemes)")


# ═══════════════════════════════════════════════════════════════════════
# TASK 9: DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════
doc = """# Month 2 — Phoneme Clustering Documentation
## Project: Multilingual G2P System  |  Hindi · Gujarati · Marathi

---

## 1. Method Used

### 1.1 Embedding — Co-occurrence PPMI
Each phoneme was represented as a co-occurrence vector over a ±2 phoneme sliding window
across all 54,753 entries in the multilingual dataset. Raw counts were transformed using
Positive Pointwise Mutual Information (PPMI) to highlight meaningful phonological patterns.

- Input vocabulary : 57 unique phonemes (Hindi + Gujarati + Marathi combined)
- Embedding shape  : 57 × 57

### 1.2 Dimensionality Reduction — UMAP
UMAP reduced the 57-dimensional vectors to 2D for visualization.

| Parameter    | Value |
|-------------|-------|
| n_components | 2     |
| n_neighbors  | 5     |
| min_dist     | 0.3   |
| random_state | 42    |

UMAP was preferred over PCA because it preserves local neighbourhood structure — phonemes
that appear in similar contexts cluster tightly in the 2D projection.

### 1.3 Clustering — K-Means
K-Means was applied on the full 57-dim PPMI embeddings (not the 2D UMAP).

| Parameter    | Value |
|-------------|-------|
| K (clusters) | 12    |
| n_init       | 20    |
| random_state | 42    |

K=12 was selected via silhouette score evaluation across K=5 to K=50.

---

## 2. Why Clustering Helps

1. Reduced output complexity — 57 phonemes → 12 groups (79% reduction)
2. Better generalisation — rare phonemes grouped with similar sounds reduce sparsity
3. Linguistic interpretability — clusters align with classical categories (vowels, stops, nasals)
4. Useful for prosody modelling — phoneme clusters can share acoustic duration parameters

---

## 3. Observations

### Good groupings:
- C0 (Short Vowels)      : a, ee, i, ii, o, u — core vowels across all 3 languages
- C3 (Stops & Aspirates) : k, kh, g, gh, d, bh, p, sh — well-separated stop consonants
- C7 (Dental/Liquids)    : n, r, s, dx, tx — dental and liquid consonants
- C9 (Long Vowels)       : mq, uu — long vowel sounds correctly paired
- C5 (Diphthongs)        : ei, ou, z — special/diphthong phonemes isolated

### Issues — Singleton clusters (6 out of 12):
- kq (142 occurrences), gq (59), ax (110), f (235), ph (1501), sx (2094)
  These are rare or phonologically unique; insufficient co-occurrence for clustering.
  Recommendation: merge singletons with nearest cluster in future iterations.

### Cross-language note:
Clustering is done on combined HI+GU+MR phoneme usage, so clusters represent
language-agnostic phoneme behaviour — a strength for multilingual TTS systems.

---

## 4. Output Files

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| phoneme_cluster_mapping.json  | phoneme → cluster_id, label, members     |
| phoneme_cluster_mapping.csv   | Flat CSV version                         |
| viz_cluster_map.png           | 2D UMAP scatter plot (presentation)      |
| viz_validation.png            | 5-panel validation dashboard             |
| silhouette_scores.png         | K selection chart                        |
| phoneme_embeddings_learned.npy| PPMI matrix (57×57)                      |

---

## 5. Next Steps (Month 3)
- Use cluster IDs as auxiliary features in the seq2seq G2P model
- Evaluate if clustered output improves word error rate on held-out words
- Extend clustering to the grapheme (source) side
"""

with open("month2_documentation.md", "w") as f:
    f.write(doc)
print("✅ month2_documentation.md saved")
