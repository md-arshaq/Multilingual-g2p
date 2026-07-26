

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

# PATHS
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MOS_SCORES_PATH     = os.path.join(PROJECT_DIR, "results", "mos_scores.csv")
COMPARISON_CSV_PATH = os.path.join(PROJECT_DIR, "results", "comparison_table.csv")
CLUSTER_MAPPING     = os.path.join(SCRIPT_DIR, "phoneme_cluster_mapping.json")
OUTPUT_REPORT       = os.path.join(PROJECT_DIR, "results", "correlation_analysis.md")
OUTPUT_SCATTER      = os.path.join(PROJECT_DIR, "results", "per_vs_mos_scatter.png")


def load_mos_scores(csv_path):
    
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            auto_mos = row.get("auto_mos", "")
            human_mos = row.get("human_mos", "")

            if human_mos and human_mos != "None" and human_mos != "":
                mos = float(human_mos)
            elif auto_mos and auto_mos != "None" and auto_mos != "":
                mos = float(auto_mos)
            else:
                continue

            data.append({
                "filename": row.get("filename", ""),
                "language": row.get("language", ""),
                "condition": row.get("condition", ""),
                "mos": mos,
                "phonemes": row.get("output_phonemes", ""),
            })
    return data


def load_comparison_table(csv_path):
    
    per_values = {}
    if not os.path.exists(csv_path):
        return per_values

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("Metric", "")
            if metric == "PER":
                per_values["baseline"] = float(row.get("Baseline", 0))
                per_values["clustered"] = float(row.get("Clustered", 0))
    return per_values


def load_singletons(mapping_path):
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Find singleton clusters
    cluster_members = defaultdict(list)
    for phoneme, info in mapping.items():
        cluster_members[info["cluster_id"]].append(phoneme)

    singletons = set()
    singleton_clusters = {}
    for cid, members in cluster_members.items():
        if len(members) == 1:
            singletons.add(members[0])
            singleton_clusters[f"C{cid}"] = members[0]

    return singletons, singleton_clusters


def has_singleton_phonemes(phoneme_str, singletons, singleton_clusters):
    
    tokens = phoneme_str.split()
    for token in tokens:
        if token in singletons or token in singleton_clusters:
            return True
    return False


def main():
    print("=" * 60)
    print("TASK 9: PER ↔ MOS CORRELATION ANALYSIS")
    print("=" * 60)

    if not os.path.exists(MOS_SCORES_PATH):
        print(f"  ❌ MOS scores not found: {MOS_SCORES_PATH}")
        print("     Run Task 7 first!")
        return

    mos_data = load_mos_scores(MOS_SCORES_PATH)
    print(f"  Loaded {len(mos_data)} MOS scores")

    per_values = load_comparison_table(COMPARISON_CSV_PATH)
    if per_values:
        print(f"  Loaded PER values: baseline={per_values.get('baseline', 'N/A')}, "
              f"clustered={per_values.get('clustered', 'N/A')}")
    else:
        print("  ⚠️  No comparison_table.csv found — using placeholder PER values")
        print("     Run Task 3 first for real PER values!")
        per_values = {"baseline": 0.12, "clustered": 0.10}  # placeholders from README

    singletons, singleton_clusters = load_singletons(CLUSTER_MAPPING)
    print(f"  Singleton phonemes: {singletons}")

    stats = defaultdict(lambda: defaultdict(list))
    for d in mos_data:
        stats[d["language"]][d["condition"]].append(d["mos"])

    print("\n  Analyzing singleton impact on MOS...")

    with_singletons = []
    without_singletons = []

    for d in mos_data:
        if d["condition"] == "clustered":  # Only relevant for clustered
            if has_singleton_phonemes(d["phonemes"], singletons, singleton_clusters):
                with_singletons.append(d["mos"])
            else:
                without_singletons.append(d["mos"])

    if with_singletons:
        print(f"    Samples with singletons:    {len(with_singletons)} (mean MOS: {np.mean(with_singletons):.2f})")
    if without_singletons:
        print(f"    Samples without singletons: {len(without_singletons)} (mean MOS: {np.mean(without_singletons):.2f})")

    # VISUALIZATION
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Colors
    LANG_COLORS = {"hi": "#E85D75", "gu": "#4A90D9", "mr": "#50C878"}
    COND_COLORS = {"baseline": "#4A90D9", "clustered": "#E85D75"}

    ax1 = axes[0]

    for lang in sorted(stats.keys()):
        for cond in ["baseline", "clustered"]:
            mos_list = stats[lang].get(cond, [])
            if mos_list:
                per = per_values.get(cond, 0)
                # Scatter with jitter for visibility
                jitter_per = per + np.random.normal(0, 0.003, len(mos_list))
                ax1.scatter(jitter_per, mos_list,
                           color=LANG_COLORS.get(lang, "gray"),
                           marker='o' if cond == "baseline" else 's',
                           s=80, alpha=0.7,
                           label=f"{lang.upper()} ({cond})")

    # Add trend line (if we have enough points)
    all_per = []
    all_mos = []
    for cond in ["baseline", "clustered"]:
        per = per_values.get(cond, 0)
        for lang in stats:
            for mos_val in stats[lang].get(cond, []):
                all_per.append(per)
                all_mos.append(mos_val)

    if len(all_per) >= 4:
        z = np.polyfit(all_per, all_mos, 1)
        p = np.poly1d(z)
        per_range = np.linspace(min(all_per) - 0.01, max(all_per) + 0.01, 50)
        ax1.plot(per_range, p(per_range), "--", color="gray", alpha=0.8, linewidth=2, label="Trend")

    ax1.set_xlabel("PER (Phoneme Error Rate)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("MOS Score", fontsize=12, fontweight='bold')
    ax1.set_title("PER vs. MOS", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 5.5)
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(alpha=0.3)

    ax2 = axes[1]

    box_data = []
    box_labels = []
    box_colors = []

    if with_singletons:
        box_data.append(with_singletons)
        box_labels.append("With\nSingletons")
        box_colors.append("#FF9F43")
    if without_singletons:
        box_data.append(without_singletons)
        box_labels.append("Without\nSingletons")
        box_colors.append("#50C878")

    if not box_data:
        # If no clustered data, show baseline vs clustered overall
        for lang in sorted(stats.keys()):
            for cond in ["baseline", "clustered"]:
                mos_list = stats[lang].get(cond, [])
                if mos_list:
                    box_data.append(mos_list)
                    box_labels.append(f"{lang.upper()}\n{cond}")
                    box_colors.append(COND_COLORS.get(cond, "gray"))

    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                         widths=0.5, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='gold', markersize=8),
                         medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel("MOS Score", fontsize=12, fontweight='bold')
    ax2.set_title("Singleton Phoneme Impact on MOS", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 5.5)
    ax2.grid(axis='y', alpha=0.3)

    # ── Plot 3: Per-language PER vs MOS bar ──
    ax3 = axes[2]

    x = np.arange(len(stats))
    width = 0.35
    lang_labels = sorted(stats.keys())

    baseline_mos = [np.mean(stats[l].get("baseline", [0])) for l in lang_labels]
    clustered_mos = [np.mean(stats[l].get("clustered", [0])) for l in lang_labels]

    bars1 = ax3.bar(x - width/2, baseline_mos, width, label="Baseline",
                     color="#4A90D9", alpha=0.85, edgecolor="white")
    bars2 = ax3.bar(x + width/2, clustered_mos, width, label="Clustered",
                     color="#E85D75", alpha=0.85, edgecolor="white")

    # Annotate with PER values
    baseline_per = per_values.get("baseline", "?")
    clustered_per = per_values.get("clustered", "?")

    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'PER:{baseline_per:.3f}' if isinstance(baseline_per, float) else f'PER:{baseline_per}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color="#4A90D9")
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'PER:{clustered_per:.3f}' if isinstance(clustered_per, float) else f'PER:{clustered_per}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color="#E85D75")

    ax3.set_xlabel("Language", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Mean MOS", fontsize=12, fontweight='bold')
    ax3.set_title("MOS by Language with PER Annotations", fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.upper() for l in lang_labels])
    ax3.set_ylim(0, 5.5)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    fig.suptitle("PER ↔ MOS Correlation Analysis — Multilingual G2P",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_SCATTER), exist_ok=True)
    plt.savefig(OUTPUT_SCATTER, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved correlation plots: {OUTPUT_SCATTER}")

    # CORRELATION METRICS
    correlation_results = {}

    if len(all_per) >= 3 and len(set(all_per)) > 1:
        try:
            from scipy import stats as scipy_stats
            pearson_r, pearson_p = scipy_stats.pearsonr(all_per, all_mos)
            spearman_r, spearman_p = scipy_stats.spearmanr(all_per, all_mos)
            correlation_results = {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
            print(f"\n  Correlation Metrics:")
            print(f"    Pearson r:  {pearson_r:.4f}  (p={pearson_p:.4f})")
            print(f"    Spearman ρ: {spearman_r:.4f}  (p={spearman_p:.4f})")
        except ImportError:
            print("  ℹ️  Install scipy for correlation metrics: pip install scipy")
    else:
        print("  ℹ️  Not enough data points for correlation analysis")

    # GENERATE REPORT
    report = f

    for lang in sorted(stats.keys()):
        b_mos = stats[lang].get("baseline", [])
        c_mos = stats[lang].get("clustered", [])
        b_mean = np.mean(b_mos) if b_mos else 0
        c_mean = np.mean(c_mos) if c_mos else 0
        delta = c_mean - b_mean
        report += f"| {lang.upper()} | {b_mean:.2f} | {c_mean:.2f} | {delta:+.2f} |\n"

    # Correlation section
    report += 

    if correlation_results:
        report += f
        if correlation_results['pearson_r'] < -0.3:
            report += "**Finding:** There is a negative correlation between PER and MOS — lower PER tends to produce higher MOS scores, confirming that G2P accuracy positively impacts perceived speech quality.\n"
        elif correlation_results['pearson_r'] > 0.3:
            report += "**Finding:** Unexpectedly, there is a positive correlation — samples with higher PER also have higher MOS. This may indicate that the TTS model compensates for G2P errors.\n"
        else:
            report += "**Finding:** No strong correlation between PER and MOS was observed. This suggests that TTS quality is influenced by factors beyond G2P accuracy alone (e.g., prosody, speaker quality, audio processing).\n"
    else:
        report += "Insufficient data for statistical correlation analysis. Run Tasks 2-3 and 6-7 to generate PER and MOS data.\n"

    # Singleton section
    report += 

    if with_singletons and without_singletons:
        sing_mean = np.mean(with_singletons)
        no_sing_mean = np.mean(without_singletons)
        delta = sing_mean - no_sing_mean

        report += f
        if abs(delta) < 0.2:
            report += "**Finding:** Singleton phonemes do not significantly impact MOS scores. The dedicated cluster IDs for rare phonemes appear to preserve synthesis quality.\n"
        elif delta < -0.2:
            report += f"**Finding:** Samples with singleton phonemes score {abs(delta):.2f} points lower on average. Consider merging singleton clusters with nearest neighbors in future iterations.\n"
        else:
            report += f"**Finding:** Samples with singleton phonemes actually score slightly higher ({delta:+.2f}). This may be due to these phonemes appearing in simpler or shorter words.\n"
    else:
        report += "Insufficient singleton vs. non-singleton samples for analysis. This will be populated after Task 6 generates audio with varied phoneme content.\n"

    # Key findings
    report += f

    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Saved correlation report: {OUTPUT_REPORT}")
    print(f"\n{'='*60}")
    print("TASK 9 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
