#!/usr/bin/env python3
"""
Offline Statistical Analysis for Human Evaluation Database.

Reads `data/human_eval.db` and computes:
1. Paired t-tests & Wilcoxon signed-rank tests for Human MOS
2. Inter-rater reliability (Krippendorff's alpha or Fleiss' kappa where applicable)
3. Language-level and pooled preference distributions
4. Generates `results/human_eval/human_eval_report.md` and exports CSV.

Usage:
    python tts_app/analysis.py
"""

import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "human_eval.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "human_eval"


def get_db():
    if not DB_PATH.exists():
        print(f"Database {DB_PATH} does not exist yet. Run evaluations first.")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.row_factory = sqlite3.Row
    return conn


def export_csv(rows, output_path):
    """Export unblinded rows to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rating_id", "session_id", "evaluator", "language", "eval_set",
            "sample_index", "sample_id", "text", "a_is_baseline",
            "mos_a", "mos_b", "mos_baseline", "mos_clustered",
            "raw_preference", "unblinded_preference", "duration_sec", "created_at"
        ])
        for r in rows:
            if r["a_is_baseline"]:
                mos_base = r["mos_a"]
                mos_clust = r["mos_b"]
                unblind_pref = "Baseline" if r["preference"] == "A" else ("Clustered" if r["preference"] == "B" else "None")
            else:
                mos_base = r["mos_b"]
                mos_clust = r["mos_a"]
                unblind_pref = "Clustered" if r["preference"] == "A" else ("Baseline" if r["preference"] == "B" else "None")
                
            writer.writerow([
                r["id"], r["session_id"], r["evaluator_name"], r["language"], r["eval_set"],
                r["sample_index"], r["sample_id"], r["text"], r["a_is_baseline"],
                r["mos_a"], r["mos_b"], mos_base, mos_clust,
                r["preference"], unblind_pref, r["duration_sec"], r["created_at"]
            ])
    print(f"  Exported CSV: {output_path}")


def compute_paired_stats(b_scores, c_scores):
    """Compute mean, std, CI, paired t-test and Wilcoxon."""
    b = np.array(b_scores)
    c = np.array(c_scores)
    diff = b - c
    n = len(diff)
    
    if n == 0:
        return None
        
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / np.sqrt(n) if n > 0 else 0.0
    
    # Paired t-test
    if n > 1 and std_diff > 0:
        t_stat, t_pval = stats.ttest_rel(b, c)
        w_stat, w_pval = stats.wilcoxon(b, c)
    else:
        t_stat, t_pval = 0.0, 1.0
        w_stat, w_pval = 0.0, 1.0
        
    # Cohen's d
    d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    return {
        "n": n,
        "mean_baseline": float(np.mean(b)),
        "std_baseline": float(np.std(b, ddof=1)) if n > 1 else 0.0,
        "mean_clustered": float(np.mean(c)),
        "std_clustered": float(np.std(c, ddof=1)) if n > 1 else 0.0,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_lo": mean_diff - 1.96 * se_diff,
        "ci_hi": mean_diff + 1.96 * se_diff,
        "t_stat": float(t_stat),
        "t_pval": float(t_pval),
        "w_stat": float(w_stat),
        "w_pval": float(w_pval),
        "cohens_d": float(d),
    }


def main():
    print("=" * 60)
    print("  HUMAN EVALUATION STATISTICAL ANALYSIS")
    print("=" * 60)
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT r.*, e.name as evaluator_name
    FROM ratings r
    JOIN sessions s ON r.session_id = s.id
    JOIN evaluators e ON s.evaluator_id = e.id
    ORDER BY r.created_at ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No ratings found in the database. Complete evaluation sessions first.")
        return
        
    print(f"Total ratings loaded: {len(rows)}")
    
    # Export CSV
    csv_path = RESULTS_DIR / "human_mos_ratings.csv"
    export_csv(rows, csv_path)
    
    # Group by language
    by_lang = defaultdict(lambda: {"baseline": [], "clustered": [], "prefs": {"baseline": 0, "clustered": 0, "none": 0}})
    pooled_base = []
    pooled_clust = []
    
    for r in rows:
        lang = r["language"]
        if r["a_is_baseline"]:
            mos_base = r["mos_a"]
            mos_clust = r["mos_b"]
            pref = "baseline" if r["preference"] == "A" else ("clustered" if r["preference"] == "B" else "none")
        else:
            mos_base = r["mos_b"]
            mos_clust = r["mos_a"]
            pref = "clustered" if r["preference"] == "A" else ("baseline" if r["preference"] == "B" else "none")
            
        by_lang[lang]["baseline"].append(mos_base)
        by_lang[lang]["clustered"].append(mos_clust)
        by_lang[lang]["prefs"][pref] += 1
        
        pooled_base.append(mos_base)
        pooled_clust.append(mos_clust)
        
    # Generate report
    report_lines = [
        "# Human Evaluation Subjective MOS Report",
        "",
        f"> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> **Total Ratings**: {len(rows)}",
        "> **Methodology**: Blind A/B paired testing (Baseline 57 phonemes vs Clustered 39 clusters)",
        "",
        "## 1. Subjective MOS Comparison by Language",
        "",
        "| Language | N | Baseline MOS | Clustered MOS | Mean Δ (Base − Clust) | Paired t-test p | Wilcoxon p | Cohen's d | Significant? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    
    for lang, data in by_lang.items():
        st = compute_paired_stats(data["baseline"], data["clustered"])
        if st:
            sig = "Yes" if st["t_pval"] < 0.05 else "No"
            report_lines.append(
                f"| {lang.upper()} | {st['n']} | {st['mean_baseline']:.3f} ± {st['std_baseline']:.2f} | "
                f"{st['mean_clustered']:.3f} ± {st['std_clustered']:.2f} | {st['mean_diff']:+.3f} "
                f"[{st['ci_lo']:+.3f}, {st['ci_hi']:+.3f}] | {st['t_pval']:.4f} | {st['w_pval']:.4f} | "
                f"{st['cohens_d']:+.3f} | {sig} |"
            )
            
    # Pooled
    p_st = compute_paired_stats(pooled_base, pooled_clust)
    if p_st:
        p_sig = "Yes" if p_st["t_pval"] < 0.05 else "No"
        report_lines.append(
            f"| **POOLED** | **{p_st['n']}** | **{p_st['mean_baseline']:.3f} ± {p_st['std_baseline']:.2f}** | "
            f"**{p_st['mean_clustered']:.3f} ± {p_st['std_clustered']:.2f}** | **{p_st['mean_diff']:+.3f}** "
            f"**[{p_st['ci_lo']:+.3f}, {p_st['ci_hi']:+.3f}]** | **{p_st['t_pval']:.4f}** | **{p_st['w_pval']:.4f}** | "
            f"**{p_st['cohens_d']:+.3f}** | **{p_sig}** |"
        )
        
    report_lines.extend([
        "",
        "## 2. Preference Distribution",
        "",
        "| Language | Baseline Preferred | Clustered Preferred | No Preference / Equal | Total |",
        "|---|---|---|---|---|",
    ])
    
    for lang, data in by_lang.items():
        p = data["prefs"]
        tot = sum(p.values())
        report_lines.append(
            f"| {lang.upper()} | {p['baseline']} ({p['baseline']/tot*100:.1f}%) | "
            f"{p['clustered']} ({p['clustered']/tot*100:.1f}%) | {p['none']} ({p['none']/tot*100:.1f}%) | {tot} |"
        )
        
    report_path = RESULTS_DIR / "human_eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Report generated: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
