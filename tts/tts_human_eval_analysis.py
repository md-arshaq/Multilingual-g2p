#!/usr/bin/env python3
"""
Phase 8c: Human evaluation analysis.

After collecting scores from 5 listeners, analyze:
  - Per-condition means (naturalness, intelligibility)
  - Wilcoxon signed-rank test for paired differences
  - p-value and effect direction
  - No significance claim if p >= 0.05

Usage:
    python tts/tts_human_eval_analysis.py \
        --eval_dir results/tts_hindi_female/human_eval_package \
        [--results_dir results/tts_hindi_female]
"""

import os
import sys
import csv
import json
import argparse
import math
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_EVAL_DIR = os.path.join(
    PROJECT_DIR, "results", "tts_hindi_female", "human_eval_package"
)
DEFAULT_RESULTS = os.path.join(PROJECT_DIR, "results", "tts_hindi_female")


def load_mapping(eval_dir):
    """Load the confidential condition mapping."""
    mapping_path = os.path.join(eval_dir, "_CONFIDENTIAL_mapping.csv")
    if not os.path.exists(mapping_path):
        print(f"ERROR: Mapping file not found: {mapping_path}")
        return None

    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = int(row["pair_id"])
            mapping[pair_id] = {
                "system_A_is": row["system_A_is"],
                "system_B_is": row["system_B_is"],
            }
    return mapping


def load_listener_scores(eval_dir):
    """Load scores from all listener evaluation forms."""
    all_scores = []

    for fname in sorted(os.listdir(eval_dir)):
        if fname.startswith("evaluation_form_listener_") and fname.endswith(".csv"):
            fpath = os.path.join(eval_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip rows without scores
                    if not row.get("naturalness_A") and not row.get("naturalness_B"):
                        continue
                    all_scores.append(row)

    return all_scores


def analyze_scores(scores, mapping):
    """
    Analyze human evaluation scores.
    Returns analysis results dict.
    """
    import numpy as np

    # Unblind scores using mapping
    baseline_naturalness = []
    clustered_naturalness = []
    baseline_intelligibility = []
    clustered_intelligibility = []
    preferences = {"baseline": 0, "clustered": 0, "no_preference": 0}

    paired_naturalness = []  # (baseline - clustered) per pair
    paired_intelligibility = []

    for row in scores:
        pair_id = int(row["pair_id"])
        if pair_id not in mapping:
            continue

        m = mapping[pair_id]

        try:
            nat_a = float(row["naturalness_A"]) if row.get("naturalness_A") else None
            nat_b = float(row["naturalness_B"]) if row.get("naturalness_B") else None
            int_a = float(row["intelligibility_A"]) if row.get("intelligibility_A") else None
            int_b = float(row["intelligibility_B"]) if row.get("intelligibility_B") else None
        except ValueError:
            continue

        # Map A/B back to baseline/clustered
        if m["system_A_is"] == "baseline":
            b_nat, c_nat = nat_a, nat_b
            b_int, c_int = int_a, int_b
        else:
            b_nat, c_nat = nat_b, nat_a
            b_int, c_int = int_b, int_a

        if b_nat is not None:
            baseline_naturalness.append(b_nat)
        if c_nat is not None:
            clustered_naturalness.append(c_nat)
        if b_int is not None:
            baseline_intelligibility.append(b_int)
        if c_int is not None:
            clustered_intelligibility.append(c_int)

        if b_nat is not None and c_nat is not None:
            paired_naturalness.append(b_nat - c_nat)
        if b_int is not None and c_int is not None:
            paired_intelligibility.append(b_int - c_int)

        # Preference
        pref = row.get("preference", "").strip().upper()
        if pref == "A":
            pref_condition = m["system_A_is"]
        elif pref == "B":
            pref_condition = m["system_B_is"]
        else:
            pref_condition = "no_preference"

        preferences[pref_condition] = preferences.get(pref_condition, 0) + 1

    results = {
        "baseline_naturalness": baseline_naturalness,
        "clustered_naturalness": clustered_naturalness,
        "baseline_intelligibility": baseline_intelligibility,
        "clustered_intelligibility": clustered_intelligibility,
        "paired_naturalness": paired_naturalness,
        "paired_intelligibility": paired_intelligibility,
        "preferences": preferences,
    }

    return results


def compute_stats(values):
    """Compute mean, std, 95% CI."""
    import numpy as np
    if not values:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    n = len(arr)
    se = std / math.sqrt(n) if n > 1 else 0
    t_val = 2.0  # approximate for small samples
    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci_low": round(mean - t_val * se, 3),
        "ci_high": round(mean + t_val * se, 3),
        "n": n,
    }


def wilcoxon_test(diffs):
    """Wilcoxon signed-rank test on paired differences."""
    try:
        from scipy.stats import wilcoxon
        import numpy as np

        diffs = np.array(diffs)
        # Remove zeros (ties)
        nonzero = diffs[diffs != 0]

        if len(nonzero) < 5:
            return {"statistic": None, "p_value": None,
                    "note": "Too few non-zero differences for Wilcoxon test"}

        stat, p_value = wilcoxon(nonzero)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "n_nonzero": len(nonzero),
        }
    except ImportError:
        return {"statistic": None, "p_value": None,
                "note": "scipy not installed"}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8c: Human evaluation analysis"
    )
    parser.add_argument(
        "--eval_dir", type=str, default=DEFAULT_EVAL_DIR,
        help="Directory containing evaluation forms and mapping"
    )
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_RESULTS,
        help="Directory to save analysis results"
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("PHASE 8c: HUMAN EVALUATION ANALYSIS")
    print("=" * 60)

    # Load data
    mapping = load_mapping(args.eval_dir)
    if mapping is None:
        sys.exit(1)

    scores = load_listener_scores(args.eval_dir)
    print(f"  Loaded {len(scores)} evaluation responses")
    print(f"  Mapping for {len(mapping)} pairs")

    if not scores:
        print("\n  No evaluation scores found. Collect listener data first.")
        sys.exit(1)

    # Analyze
    results = analyze_scores(scores, mapping)

    # Statistics
    b_nat_stats = compute_stats(results["baseline_naturalness"])
    c_nat_stats = compute_stats(results["clustered_naturalness"])
    b_int_stats = compute_stats(results["baseline_intelligibility"])
    c_int_stats = compute_stats(results["clustered_intelligibility"])

    # Wilcoxon tests
    nat_wilcoxon = wilcoxon_test(results["paired_naturalness"])
    int_wilcoxon = wilcoxon_test(results["paired_intelligibility"])

    # Generate report
    report_lines = [
        "# Human Evaluation Report",
        "",
        "## Naturalness Scores (1-5)",
        "",
        "| Condition | Mean | Std | 95% CI | N |",
        "|-----------|------|-----|--------|---|",
        f"| Baseline | {b_nat_stats['mean']} | {b_nat_stats['std']} | "
        f"[{b_nat_stats['ci_low']}, {b_nat_stats['ci_high']}] | {b_nat_stats['n']} |",
        f"| Clustered | {c_nat_stats['mean']} | {c_nat_stats['std']} | "
        f"[{c_nat_stats['ci_low']}, {c_nat_stats['ci_high']}] | {c_nat_stats['n']} |",
        "",
        "### Wilcoxon Signed-Rank Test (Naturalness)",
        "",
    ]

    if nat_wilcoxon.get("p_value") is not None:
        p = nat_wilcoxon["p_value"]
        direction = "baseline > clustered" if sum(results["paired_naturalness"]) > 0 else "clustered > baseline"
        significance = "statistically significant" if p < 0.05 else "NOT statistically significant"
        report_lines.extend([
            f"- **Test statistic:** {nat_wilcoxon['statistic']:.4f}",
            f"- **p-value:** {p:.6f}",
            f"- **Effect direction:** {direction}",
            f"- **Conclusion:** The difference is **{significance}** (p {'<' if p < 0.05 else '>='} 0.05)",
            "",
        ])
    else:
        report_lines.extend([
            f"- {nat_wilcoxon.get('note', 'Test not available')}",
            "",
        ])

    report_lines.extend([
        "## Intelligibility Scores (1-5)",
        "",
        "| Condition | Mean | Std | 95% CI | N |",
        "|-----------|------|-----|--------|---|",
        f"| Baseline | {b_int_stats['mean']} | {b_int_stats['std']} | "
        f"[{b_int_stats['ci_low']}, {b_int_stats['ci_high']}] | {b_int_stats['n']} |",
        f"| Clustered | {c_int_stats['mean']} | {c_int_stats['std']} | "
        f"[{c_int_stats['ci_low']}, {c_int_stats['ci_high']}] | {c_int_stats['n']} |",
        "",
        "### Wilcoxon Signed-Rank Test (Intelligibility)",
        "",
    ])

    if int_wilcoxon.get("p_value") is not None:
        p = int_wilcoxon["p_value"]
        direction = "baseline > clustered" if sum(results["paired_intelligibility"]) > 0 else "clustered > baseline"
        significance = "statistically significant" if p < 0.05 else "NOT statistically significant"
        report_lines.extend([
            f"- **Test statistic:** {int_wilcoxon['statistic']:.4f}",
            f"- **p-value:** {p:.6f}",
            f"- **Effect direction:** {direction}",
            f"- **Conclusion:** The difference is **{significance}** (p {'<' if p < 0.05 else '>='} 0.05)",
            "",
        ])
    else:
        report_lines.extend([
            f"- {int_wilcoxon.get('note', 'Test not available')}",
            "",
        ])

    # Preferences
    prefs = results["preferences"]
    report_lines.extend([
        "## Pairwise Preferences",
        "",
        f"- Baseline preferred: {prefs.get('baseline', 0)}",
        f"- Clustered preferred: {prefs.get('clustered', 0)}",
        f"- No preference: {prefs.get('no_preference', 0)}",
        "",
        "---",
        "",
        "> **Note:** This evaluation used 5 native Hindi listeners. "
        "Results should be considered preliminary given the small listener pool "
        "and limited dataset (1.75 hours of female speech).",
    ])

    # Save report
    report_path = os.path.join(args.results_dir, "human_eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("PHASE 8c COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Naturalness: Baseline={b_nat_stats['mean']}, Clustered={c_nat_stats['mean']}")
    print(f"  Intelligibility: Baseline={b_int_stats['mean']}, Clustered={c_int_stats['mean']}")
    if nat_wilcoxon.get("p_value"):
        print(f"  Naturalness Wilcoxon p={nat_wilcoxon['p_value']:.6f}")
    if int_wilcoxon.get("p_value"):
        print(f"  Intelligibility Wilcoxon p={int_wilcoxon['p_value']:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
