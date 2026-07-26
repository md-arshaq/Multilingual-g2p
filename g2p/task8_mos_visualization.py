

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

# PATHS
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

SCORES_PATH = os.path.join(PROJECT_DIR, "results", "mos_scores.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "results", "mos_comparison.png")


def load_scores(csv_path):
    
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            auto_mos = row.get("auto_mos", "")
            human_mos = row.get("human_mos", "")

            # Prefer human MOS if available, else use auto
            if human_mos and human_mos != "None" and human_mos != "":
                mos = float(human_mos)
                mos_type = "human"
            elif auto_mos and auto_mos != "None" and auto_mos != "":
                mos = float(auto_mos)
                mos_type = "auto"
            else:
                continue

            data.append({
                "language": row.get("language", ""),
                "condition": row.get("condition", ""),
                "mos": mos,
                "mos_type": mos_type,
            })
    return data


def main():
    print("=" * 60)
    print("TASK 8: MOS VISUALIZATION")
    print("=" * 60)

    if not os.path.exists(SCORES_PATH):
        print(f"  ❌ Scores file not found: {SCORES_PATH}")
        print("     Run Task 7 first!")
        return

    # Load data
    data = load_scores(SCORES_PATH)
    if not data:
        print("  ❌ No valid MOS scores found in CSV!")
        return

    print(f"  Loaded {len(data)} MOS scores")

    # Organize by language and condition
    scores = defaultdict(lambda: defaultdict(list))
    for d in data:
        scores[d["language"]][d["condition"]].append(d["mos"])

    languages = sorted(scores.keys())
    conditions = ["baseline", "clustered"]

    # Colors
    BASELINE_COLOR  = "#4A90D9"   # Blue
    CLUSTERED_COLOR = "#E85D75"   # Red/Pink

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    x = np.arange(len(languages))
    width = 0.35

    baseline_means = []
    baseline_stds  = []
    clustered_means = []
    clustered_stds  = []

    for lang in languages:
        b_scores = scores[lang].get("baseline", [0])
        c_scores = scores[lang].get("clustered", [0])
        baseline_means.append(np.mean(b_scores))
        baseline_stds.append(np.std(b_scores))
        clustered_means.append(np.mean(c_scores))
        clustered_stds.append(np.std(c_scores))

    bars1 = ax1.bar(x - width/2, baseline_means, width,
                     yerr=baseline_stds, capsize=5,
                     label="Baseline", color=BASELINE_COLOR, alpha=0.85,
                     edgecolor="white", linewidth=1.5)
    bars2 = ax1.bar(x + width/2, clustered_means, width,
                     yerr=clustered_stds, capsize=5,
                     label="Clustered", color=CLUSTERED_COLOR, alpha=0.85,
                     edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, mean in zip(bars1, baseline_means):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, mean in zip(bars2, clustered_means):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_xlabel("Language", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Mean MOS Score", fontsize=12, fontweight='bold')
    ax1.set_title("Mean Opinion Score: Baseline vs. Clustered", fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([lang.upper() for lang in languages], fontsize=11)
    ax1.set_ylim(0, 5.5)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='Neutral (3.0)')

    ax2 = axes[1]

    box_data = []
    box_labels = []
    box_colors = []

    for lang in languages:
        b_scores = scores[lang].get("baseline", [])
        c_scores = scores[lang].get("clustered", [])
        if b_scores:
            box_data.append(b_scores)
            box_labels.append(f"{lang.upper()}\nBaseline")
            box_colors.append(BASELINE_COLOR)
        if c_scores:
            box_data.append(c_scores)
            box_labels.append(f"{lang.upper()}\nClustered")
            box_colors.append(CLUSTERED_COLOR)

    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                         widths=0.6, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='gold', markersize=8),
                         medianprops=dict(color='black', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, alpha=0.5))

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel("MOS Score", fontsize=12, fontweight='bold')
    ax2.set_title("MOS Distribution: Baseline vs. Clustered", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 5.5)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BASELINE_COLOR, alpha=0.7, label='Baseline'),
        Patch(facecolor=CLUSTERED_COLOR, alpha=0.7, label='Clustered'),
    ]
    ax2.legend(handles=legend_elements, fontsize=11, loc='upper right')

    mos_type = data[0]["mos_type"] if data else "auto"
    fig.suptitle(
        f"Multilingual G2P — MOS Comparison ({'Automated' if mos_type == 'auto' else 'Human'} Scores)\n"
        f"Hindi · Gujarati · Marathi",
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved MOS comparison chart: {OUTPUT_PATH}")

    # ── Statistical test (if enough samples) ─
    try:
        from scipy import stats as scipy_stats

        all_baseline = []
        all_clustered = []
        for lang in languages:
            all_baseline.extend(scores[lang].get("baseline", []))
            all_clustered.extend(scores[lang].get("clustered", []))

        if len(all_baseline) >= 3 and len(all_clustered) >= 3:
            # Mann-Whitney U test (non-parametric, doesn't assume normality)
            u_stat, p_value = scipy_stats.mannwhitneyu(all_baseline, all_clustered, alternative='two-sided')
            print(f"\n  Statistical Test (Mann-Whitney U):")
            print(f"    U-statistic: {u_stat:.2f}")
            print(f"    p-value:     {p_value:.4f}")
            if p_value < 0.05:
                print(f"    Result:      Significant difference (p < 0.05)")
            else:
                print(f"    Result:      No significant difference (p ≥ 0.05)")

            # If paired data available (same number of samples)
            if len(all_baseline) == len(all_clustered):
                t_stat, t_p = scipy_stats.ttest_rel(all_baseline, all_clustered)
                print(f"\n  Paired t-test:")
                print(f"    t-statistic: {t_stat:.2f}")
                print(f"    p-value:     {t_p:.4f}")
    except ImportError:
        print("\n  ℹ️  Install scipy for statistical tests: pip install scipy")

    print(f"\n{'='*60}")
    print("TASK 8 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
