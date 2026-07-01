"""
Month 2  |  Tasks 7-9: Visualization, Validation & Documentation
 

Outputs:
  viz_cluster_map.png    ← Presentation-quality 2D cluster map
  viz_validation.png     ← 5-panel validation dashboard
  month2_documentation.md
  emb2d.npy              ← alias copy for downstream compatibility
  labels.npy             ← derived from mapping JSON
  clusters.json          ← derived from mapping JSON
"""

import json, os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print(" adjustText not installed — labels may overlap. pip install adjustText")

DARK = "#0F1117"; CARD = "#1A1D27"; WHITE = "white"; GRID = "#222"


# STEP 0: LOAD ALL INPUTS

print("Loading inputs...")

# ── 2D embeddings ─────────────────────────────────────────────────────────────
if os.path.exists("phoneme_2d_umap.npy"):
    emb2d = np.load("phoneme_2d_umap.npy")
    dim_method = "UMAP"
elif os.path.exists("phoneme_2d_pca.npy"):
    emb2d = np.load("phoneme_2d_pca.npy")
    dim_method = "PCA"
else:
    raise FileNotFoundError(
        "Neither phoneme_2d_umap.npy nor phoneme_2d_pca.npy found. "
        "Run month2_phoneme_pipeline.py first."
    )
print(f"  2D embeddings loaded from phoneme_2d_{dim_method.lower()}.npy  shape={emb2d.shape}")

# ── Vocab ─────────────────────────────────────────────────────────────────────
with open("phoneme_vocab.json") as f:
    vocab = json.load(f)

unique_phonemes = [vocab["id_to_phoneme"][str(i)] for i in range(vocab["vocab_size"])]
phoneme_counts  = vocab["phoneme_counts"]

# ── Cluster mapping (single source of truth) ──────────────────────────────────
with open("phoneme_cluster_mapping.json") as f:
    mapping = json.load(f)

# Rebuild labels array aligned to unique_phonemes order
labels = np.array([mapping[p]["cluster_id"] for p in unique_phonemes], dtype=int)

# Rebuild clusters dict: cluster_id → [phonemes]
clusters: dict[int, list[str]] = {}
for p, info in mapping.items():
    cid = info["cluster_id"]
    clusters.setdefault(cid, [])
    if p not in clusters[cid]:
        clusters[cid].append(p)

# Cluster label strings, derived from mapping
cluster_label_str: dict[int, str] = {
    info["cluster_id"]: info["cluster_label"]
    for info in mapping.values()
}

print(f"  Phonemes: {len(unique_phonemes)}   Clusters: {len(clusters)}")

# ── Save compatibility aliases (so other scripts can load emb2d.npy etc.) ─────
np.save("emb2d.npy",   emb2d)
np.save("labels.npy",  labels)
with open("clusters.json", "w") as f:
    json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
print("  ✅ emb2d.npy / labels.npy / clusters.json written (compatibility aliases)")


# BUILD CLUSTER_INFO  — assigned DYNAMICALLY by label string, not hardcoded IDs

# Map linguistic label keywords → (hex colour, phoneme_type)
LABEL_STYLE_MAP = [
    # (substring_to_match,          colour,    type)
    ("Short Vowel",                 "#4E9AF1", "vowel"),
    ("Long Vowel",                  "#74B9FF", "vowel"),
    ("Diphthong",                   "#A29BFE", "vowel"),
    ("Vowel",                       "#5DADE2", "vowel"),      
    ("Nasal",                       "#F7B731", "consonant"),
    ("Liquid",                      "#FFA07A", "consonant"),
    ("Semivowel",                   "#FFA07A", "consonant"),
    ("Dental",                      "#F7B731", "consonant"),
    ("Bilabial",                    "#E85D75", "consonant"),
    ("Affricate",                   "#C0392B", "consonant"),
    ("Fricative",                   "#55EFC4", "consonant"),
    ("Velar",                       "#E85D75", "consonant"),
    ("Stop",                        "#E85D75", "consonant"),
    ("Common Consonant",            "#F4845F", "consonant"),
    ("Glottal",                     "#81ECEC", "rare"),
    ("Special",                     "#FAB1A0", "rare"),
    ("Schwa",                       "#81ECEC", "rare"),
    ("Retroflex",                   "#FAB1A0", "rare"),
    ("Rare",                        "#B2BEC3", "rare"),
    ("Misc",                        "#9EA7AD", "rare"),       
]

FALLBACK_COLORS = [
    "#FDCB6E","#6C5CE7","#00B894","#D63031","#0984E3",
    "#E17055","#2D3436","#636E72","#00CEC9","#BADC58",
    "#F9CA24","#6AB04C",
]

def style_for_label(label_str: str, fallback_idx: int):
    for keyword, color, ptype in LABEL_STYLE_MAP:
        if keyword.lower() in label_str.lower():
            return color, ptype
    return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)], "rare"

# Build CLUSTER_INFO keyed by actual cluster ID from this run
CLUSTER_INFO: dict[int, tuple[str, str, str]] = {}
for idx, cid in enumerate(sorted(clusters.keys())):
    label_str       = cluster_label_str.get(cid, "Misc")
    color, ptype    = style_for_label(label_str, idx)
    CLUSTER_INFO[cid] = (label_str, color, ptype)

print("\nCluster colour/type assignments:")
for cid in sorted(CLUSTER_INFO):
    label, color, ptype = CLUSTER_INFO[cid]
    print(f"  C{cid:2d}  {ptype:10s}  {color}  {label}")


#VISUALIZATION — Main 2D Cluster Map
print("\nRendering viz_cluster_map.png ...")

fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

# ── Cluster regions — tight std-dev ellipses
from matplotlib.patches import Ellipse

for cid, phones in clusters.items():
    idxs  = [unique_phonemes.index(p) for p in phones if p in unique_phonemes]
    pts   = emb2d[idxs]
    color = CLUSTER_INFO[cid][1]
    if len(pts) < 2:
        continue
    cx, cy  = pts[:, 0].mean(), pts[:, 1].mean()
    rx = max(pts[:, 0].std() * 1.5, 0.08)
    ry = max(pts[:, 1].std() * 1.5, 0.08)
    ellipse = Ellipse((cx, cy), width=rx * 2, height=ry * 2,
                      facecolor=color, alpha=0.13, edgecolor=color,
                      linewidth=1.0, linestyle="--", zorder=1)
    ax.add_patch(ellipse)

# ── Scatter points ────────────────────────────────────────────────────────────
for cid, phones in clusters.items():
    idxs = [unique_phonemes.index(p) for p in phones if p in unique_phonemes]
    xs, ys = emb2d[idxs, 0], emb2d[idxs, 1]
    color  = CLUSTER_INFO[cid][1]
    kind   = CLUSTER_INFO[cid][2]
    marker = "o" if kind == "vowel" else ("s" if kind == "consonant" else "^")
    sz     = 300 if kind != "rare" else 160
    ax.scatter(xs, ys, s=sz, c=color, marker=marker,
               edgecolors="white", linewidths=0.7, alpha=0.92, zorder=4)

# ── Phoneme labels ────────────────────────────────────────────────────────────
texts = []
for i, p in enumerate(unique_phonemes):
    cid = int(labels[i])
    t = ax.text(emb2d[i, 0], emb2d[i, 1] + 0.06, p,
                fontsize=8.5, fontweight="bold", color="white",
                ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=CLUSTER_INFO[cid][1], ec="none", alpha=0.75))
    texts.append(t)

if HAS_ADJUST_TEXT:
    adjust_text(texts, ax=ax, expand_text=(1.2, 1.3),
                arrowprops=dict(arrowstyle="-", color="#555", lw=0.4))

# ── Cluster banners for larger clusters (≥ 4 phonemes) ───────────────────────
banner_offsets = [(-0.7, 0.6), (0.5, 0.45), (0.4, -0.55),
                  (-0.3, -0.55), (-0.5, 0.5), (-0.5, -0.42)]
large_clusters = [cid for cid, phones in sorted(clusters.items())
                  if len(phones) >= 4][:6]

for i, cid in enumerate(large_clusters):
    phones = clusters[cid]
    idxs   = [unique_phonemes.index(p) for p in phones if p in unique_phonemes]
    cx     = emb2d[idxs, 0].mean()
    cy     = emb2d[idxs, 1].mean()
    dx, dy = banner_offsets[i % len(banner_offsets)]
    color  = CLUSTER_INFO[cid][1]
    ax.annotate(f"● {CLUSTER_INFO[cid][0]}",
                xy=(cx, cy), xytext=(cx + dx, cy + dy),
                fontsize=9.5, color=color, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.35", fc=CARD, ec=color,
                          lw=1.3, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0), zorder=7)

legend_elements = [
    Line2D([0],[0], marker="o", color="w", label="Vowel cluster",
           markerfacecolor="#4E9AF1", markersize=11),
    Line2D([0],[0], marker="s", color="w", label="Consonant cluster",
           markerfacecolor="#E85D75", markersize=11),
    Line2D([0],[0], marker="^", color="w", label="Rare / singleton",
           markerfacecolor="#B2BEC3", markersize=11),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
          facecolor=CARD, edgecolor="#555", labelcolor=WHITE, framealpha=0.9)

ax.set_title(
    f"Phoneme Cluster Map  —  Hindi · Gujarati · Marathi\n"
    f"{dim_method} Projection  +  K-Means Clustering  (K = {len(clusters)})  "
    f"|  {len(unique_phonemes)} Phonemes → {len(clusters)} Groups",
    fontsize=13, fontweight="bold", color=WHITE, pad=16)
ax.set_xlabel(f"{dim_method} Dimension 1", color="#AAAAAA", fontsize=11)
ax.set_ylabel(f"{dim_method} Dimension 2", color="#AAAAAA", fontsize=11)
ax.tick_params(colors="#555")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a2a2a")
ax.grid(True, color="#1e1e1e", linewidth=0.6)
plt.tight_layout()
plt.savefig("viz_cluster_map.png", dpi=180, bbox_inches="tight", facecolor=DARK)
print("✅ viz_cluster_map.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8: VALIDATION DASHBOARD  (5-panel)
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering viz_validation.png ...")

fig2 = plt.figure(figsize=(16, 10))
fig2.patch.set_facecolor(DARK)
gs   = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38)

def style_ax(a, title):
    a.set_facecolor(CARD)
    a.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    a.tick_params(colors="#888")
    for sp in a.spines.values():
        sp.set_edgecolor("#333")
    a.grid(True, color=GRID, linewidth=0.5)

# ── Panel A: Before vs After ──────────────────────────────────────────────────
ax_a = fig2.add_subplot(gs[0, 0])
style_ax(ax_a, "A  Reduction: Phonemes → Clusters")
vals = [len(unique_phonemes), len(clusters)]
bars = ax_a.bar(["Original\nPhonemes", "Clustered\nGroups"], vals,
                color=["#4E9AF1", "#E85D75"], width=0.5,
                edgecolor="white", linewidth=0.7)
for bar, val in zip(bars, vals):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
              str(val), ha="center", va="bottom",
              color=WHITE, fontsize=13, fontweight="bold")
ax_a.set_ylim(0, max(vals) + 8)
ax_a.set_ylabel("Count", color="#888")
pct = (1 - vals[1] / vals[0]) * 100
ax_a.text(0.5, 0.88, f"↓ {pct:.0f}% reduction", transform=ax_a.transAxes,
          ha="center", color="#55EFC4", fontsize=11, fontweight="bold")

# ── Panel B: Cluster size distribution ───────────────────────────────────────
ax_b = fig2.add_subplot(gs[0, 1])
style_ax(ax_b, "B  Cluster Size Distribution")
cids   = sorted(clusters.keys())
sizes  = [len(clusters[c]) for c in cids]
bcolors = [CLUSTER_INFO[c][1] for c in cids]
ax_b.bar([f"C{c}" for c in cids], sizes, color=bcolors,
         edgecolor="white", linewidth=0.5)
ax_b.tick_params(axis="x", rotation=45, labelsize=7.5)
ax_b.set_ylabel("Number of Phonemes", color="#888")

# ── Panel C: Frequency per cluster ───────────────────────────────────────────
ax_c = fig2.add_subplot(gs[0, 2])
style_ax(ax_c, "C  Total Frequency per Cluster")
freq_sums   = {cid: sum(phoneme_counts.get(p, 0) for p in phones)
               for cid, phones in clusters.items()}
sorted_cids = sorted(freq_sums, key=freq_sums.get, reverse=True)
ax_c.barh([f"C{c}" for c in sorted_cids],
          [freq_sums[c] for c in sorted_cids],
          color=[CLUSTER_INFO[c][1] for c in sorted_cids],
          edgecolor="white", linewidth=0.4)
ax_c.set_xlabel("Total phoneme occurrences", color="#888", fontsize=9)
ax_c.tick_params(axis="y", labelsize=8, labelcolor=WHITE)
ax_c.invert_yaxis()

# ── Panel D: Type pie — counts derived from CLUSTER_INFO type tags ────────────
ax_d = fig2.add_subplot(gs[1, 0])
ax_d.set_facecolor(CARD)
ax_d.set_title("D  Phoneme Type Distribution",
               color=WHITE, fontsize=11, fontweight="bold", pad=10)

type_counts = {"vowel": 0, "consonant": 0, "rare": 0}
for cid, phones in clusters.items():
    ptype = CLUSTER_INFO[cid][2]
    type_counts[ptype] += len(phones)

v = type_counts["vowel"]
c = type_counts["consonant"]
r = type_counts["rare"]
pie_data   = [(v, "Vowels", "#4E9AF1"),
              (c, "Consonants", "#E85D75"),
              (r, "Rare/Special", "#B2BEC3")]
pie_data   = [(val, lbl, col) for val, lbl, col in pie_data if val > 0]
pie_vals   = [x[0] for x in pie_data]
pie_labels = [x[1] for x in pie_data]
pie_colors = [x[2] for x in pie_data]

wedges, texts_pie, pcts = ax_d.pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct="%1.0f%%", startangle=90,
    textprops={"color": WHITE, "fontsize": 9},
    wedgeprops={"edgecolor": CARD, "linewidth": 2})
for pct in pcts:
    pct.set_color(WHITE)

# ── Panel E: Top-10 frequency bar ────────────────────────────────────────────
ax_e = fig2.add_subplot(gs[1, 1:])
ax_e.set_facecolor(CARD)
ax_e.set_title("E  Top 10 Phonemes by Frequency (coloured by cluster)",
               color=WHITE, fontsize=11, fontweight="bold", pad=10)
top10    = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
ph_names = [x[0] for x in top10]
ph_freqs = [x[1] for x in top10]
ph_colors = [CLUSTER_INFO[int(labels[unique_phonemes.index(p)])][1]
             for p in ph_names]
ax_e.bar(ph_names, ph_freqs, color=ph_colors, edgecolor="white", linewidth=0.6)
for i, (name, freq) in enumerate(zip(ph_names, ph_freqs)):
    ax_e.text(i, freq + 300, f"{freq:,}",
              ha="center", va="bottom", color=WHITE, fontsize=8.5)
ax_e.set_ylabel("Frequency", color="#888")
ax_e.tick_params(axis="x", labelsize=11, labelcolor=WHITE)
ax_e.tick_params(axis="y", colors="#888")
for sp in ax_e.spines.values():
    sp.set_edgecolor("#333")
ax_e.grid(True, color=GRID, linewidth=0.5, axis="y")

fig2.suptitle(
    f"Phoneme Clustering  —  Validation Dashboard\n"
    f"Hindi · Gujarati · Marathi  |  "
    f"{len(unique_phonemes)} Phonemes  →  {len(clusters)} Clusters",
    fontsize=14, fontweight="bold", color=WHITE, y=1.01)
plt.savefig("viz_validation.png", dpi=180, bbox_inches="tight", facecolor=DARK)
print("✅ viz_validation.png saved")

# ── Validation numbers to stdout ─────────────────────────────────────────────
singletons = [cid for cid, phones in clusters.items() if len(phones) == 1]
largest_cid = max(clusters, key=lambda c: len(clusters[c]))

print("\n─── Validation Numbers ───")
print(f"  Original phonemes  : {len(unique_phonemes)}")
print(f"  Clusters           : {len(clusters)}")
print(f"  Reduction          : {(1 - len(clusters)/len(unique_phonemes))*100:.1f}%")
print(f"  Singleton clusters : {len(singletons)}  {singletons}")
print(f"  Largest cluster    : C{largest_cid} ({len(clusters[largest_cid])} phonemes)  "
      f"→ {cluster_label_str[largest_cid]}")
print(f"  Vowel phonemes     : {type_counts['vowel']}")
print(f"  Consonant phonemes : {type_counts['consonant']}")
print(f"  Rare/special       : {type_counts['rare']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 9: DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\nWriting month2_documentation.md ...")

cluster_table_rows = ""
for cid in sorted(clusters.keys()):
    members = ", ".join(sorted(clusters[cid]))
    label   = cluster_label_str.get(cid, "Misc")
    freq    = sum(phoneme_counts.get(p, 0) for p in clusters[cid])
    singleton_flag = "  ⚠ singleton" if len(clusters[cid]) == 1 else ""
    cluster_table_rows += (
        f"| C{cid:<2d} | {label:<30s} | {len(clusters[cid]):>2d} | "
        f"{freq:>8,} | {members}{singleton_flag} |\n"
    )

singleton_notes = "\n".join(
    f"- **C{cid}** ({', '.join(clusters[cid])}) — "
    f"freq={sum(phoneme_counts.get(p,0) for p in clusters[cid]):,}; "
    "consider merging with nearest cluster"
    for cid in singletons
) if singletons else "_None — all clusters have ≥ 2 members._"

doc = f"""# Month 2 — Phoneme Clustering Documentation
## Project: Multilingual G2P System  |  Hindi · Gujarati · Marathi

---

## 1. Method Used

### 1.1 Embedding — Co-occurrence PPMI
Each phoneme was represented as a co-occurrence vector over a ±2 phoneme sliding window
across all records in the multilingual dataset. Raw counts were transformed using
Positive Pointwise Mutual Information (PPMI) to highlight meaningful phonological patterns.

- Input vocabulary : {len(unique_phonemes)} unique phonemes (Hindi + Gujarati + Marathi combined)
- Embedding shape  : {len(unique_phonemes)} × {len(unique_phonemes)}

### 1.2 Dimensionality Reduction — {dim_method}
{dim_method} reduced the {len(unique_phonemes)}-dimensional vectors to 2D for visualization.

| Parameter    | Value |
|-------------|-------|
| n_components | 2     |
| n_neighbors  | 5     |
| min_dist     | 0.3   |
| random_state | 42    |

{dim_method} was preferred over PCA because it preserves local neighbourhood structure —
phonemes that appear in similar contexts cluster tightly in the 2D projection.

### 1.3 Clustering — K-Means
K-Means was applied on the full {len(unique_phonemes)}-dim PPMI embeddings (not the 2D {dim_method}).

| Parameter    | Value |
|-------------|-------|
| K (clusters) | {len(clusters)}    |
| n_init       | 20    |
| random_state | 42    |

K={len(clusters)} was selected via silhouette score evaluation across K=5 to K=50.

**Note on cluster ID stability:** K-Means cluster IDs are not deterministic across runs.
All downstream scripts should look up cluster semantics via `phoneme_cluster_mapping.json`
(phoneme → cluster_id + label) rather than hardcoding numeric IDs.

---

## 2. Why Clustering Helps

1. Reduced output complexity — {len(unique_phonemes)} phonemes → {len(clusters)} groups ({pct:.0f}% reduction)
2. Better generalisation — rare phonemes grouped with similar sounds reduce sparsity
3. Linguistic interpretability — clusters align with classical categories (vowels, stops, nasals)
4. Useful for prosody modelling — phoneme clusters can share acoustic duration parameters

---

## 3. Cluster Summary

| ID  | Label                          |  N | Frequency | Members |
|-----|--------------------------------|----|-----------|---------|
{cluster_table_rows}

### Singleton clusters (potential issues)
{singleton_notes}

### Cross-language note
Clustering is done on combined HI+GU+MR phoneme usage, so clusters represent
language-agnostic phoneme behaviour — a strength for multilingual TTS systems.

---

## 4. Output Files

| File                          | Description                                   |
|-------------------------------|-----------------------------------------------|
| phoneme_cluster_mapping.json  | phoneme → cluster_id, label, members          |
| phoneme_cluster_mapping.csv   | Flat CSV version                              |
| viz_cluster_map.png           | 2D {dim_method} scatter plot (presentation)         |
| viz_validation.png            | 5-panel validation dashboard                  |
| silhouette_scores.png         | K selection chart                             |
| phoneme_embeddings_learned.npy| PPMI matrix ({len(unique_phonemes)}×{len(unique_phonemes)})                      |
| emb2d.npy                     | Alias copy of phoneme_2d_{dim_method.lower()}.npy (compatibility) |
| labels.npy                    | Per-phoneme cluster ID array                  |
| clusters.json                 | cluster_id → [phonemes] dict                  |

---

## 5. Known Issues & Recommendations

### Label assignment (fixed in this version)
The original `get_label()` used raw overlap count, causing large mixed clusters to
inherit misleading labels. The fix uses Jaccard similarity (overlap/union) so label
assignment is proportional — a 15-phoneme cluster sharing 3 members with "Stops" no
longer outscores a 4-phoneme cluster sharing 3 members.

### ID-hardcoding risk
Any script that references cluster IDs numerically (e.g. `clusters[0]`) will silently
break if K-Means assigns different IDs on a new run. Always resolve through
`phoneme_cluster_mapping.json`.

---

## 6. Next Steps (Month 3)
- Use cluster IDs as auxiliary features in the seq2seq G2P model
- Evaluate if clustered output improves word error rate on held-out words
- Extend clustering to the grapheme (source) side
- Merge singleton clusters with their nearest centroid neighbour
"""

with open("month2_documentation.md", "w", encoding="utf-8") as f:
    f.write(doc)
print("✅ month2_documentation.md saved")

print("\n" + "="*60)
print("ALL OUTPUTS SAVED")
print("="*60)
print("  viz_cluster_map.png")
print("  viz_validation.png")
print("  month2_documentation.md")
print("  emb2d.npy  /  labels.npy  /  clusters.json  (compatibility)")