#!/usr/bin/env python3
"""
Phase 8a: Automated MOS evaluation of synthesized speech.

Computes automated/predicted MOS scores using UTMOS, SpeechMOS, or a 
librosa-based heuristic. Results are clearly labeled as automated (not human) MOS.

Reports: per-condition mean, std, 95% CI, sample count, per-sentence paired difference.

Usage:
    python tts/tts_automated_eval.py [--samples_dir samples/tts_hindi_female]
"""

import os
import sys
import csv
import json
import glob
import argparse
import math
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_SAMPLES = os.path.join(PROJECT_DIR, "samples", "tts_hindi_female")
DEFAULT_RESULTS = os.path.join(PROJECT_DIR, "results", "tts_hindi_female")


def score_with_utmos(wav_paths):
    """Try speechmos DNSMOS for MOS prediction."""
    scores = {}

    # Try speechmos (DNSMOS)
    try:
        from speechmos import dnsmos
        import librosa
        import numpy as np

        for i, wav_path in enumerate(wav_paths):
            try:
                y, sr = librosa.load(wav_path, sr=16000)
                # DNSMOS requires audio in [-1, 1]
                peak = np.max(np.abs(y))
                if peak > 1.0:
                    y = y / peak
                result = dnsmos.run(y, sr)
                # result is a dict: ovrl_mos, sig_mos, bak_mos, p808_mos
                scores[wav_path] = float(result["ovrl_mos"])
            except Exception as e:
                print(f"    WARN: scoring failed for {os.path.basename(wav_path)}: {e}")
                scores[wav_path] = None
            if (i + 1) % 20 == 0:
                print(f"    Scored {i + 1}/{len(wav_paths)}...")
        return scores, "dnsmos"
    except ImportError:
        pass

    return scores, None


def score_with_librosa_heuristic(wav_paths):
    """
    Librosa-based audio quality heuristic.
    NOT a real MOS score — this is a signal-level quality proxy.
    """
    import numpy as np

    scores = {}
    try:
        import librosa
    except ImportError:
        return scores, None

    for wav_path in wav_paths:
        try:
            y, sr = librosa.load(wav_path, sr=16000)
            if len(y) == 0:
                scores[wav_path] = None
                continue

            # Feature extraction for quality estimation
            duration = len(y) / sr
            rms = float(np.sqrt(np.mean(y ** 2)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

            # Spectral features
            spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

            # MFCC stability (lower variance = more stable/natural)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

            # Heuristic MOS approximation (scaled 1-5)
            # Based on signal quality features
            score = 3.0  # baseline

            # Penalize very low or very high RMS (silence or distortion)
            if rms > 0.01 and rms < 0.5:
                score += 0.3
            elif rms < 0.005:
                score -= 0.5

            # Reward reasonable spectral centroid (not too dull, not too harsh)
            if 1000 < spec_cent < 4000:
                score += 0.3

            # Reward spectral bandwidth (natural speech has moderate bandwidth)
            if 1000 < spec_bw < 3000:
                score += 0.2

            # Penalize very high MFCC variance (unstable/glitchy audio)
            if mfcc_var < 50:
                score += 0.3
            elif mfcc_var > 200:
                score -= 0.3

            # Penalize very short or very long clips
            if duration < 0.5:
                score -= 0.5
            elif duration > 30:
                score -= 0.2

            # Clamp to [1, 5]
            score = max(1.0, min(5.0, score))
            scores[wav_path] = round(score, 3)

        except Exception as e:
            scores[wav_path] = None

    return scores, "librosa_heuristic"


def compute_statistics(scores):
    """Compute mean, std, 95% CI for a list of scores."""
    import numpy as np

    valid = [s for s in scores if s is not None]
    if not valid:
        return {"mean": None, "std": None, "ci_95_low": None, "ci_95_high": None,
                "count": 0}

    arr = np.array(valid)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    n = len(arr)

    # 95% CI using t-distribution approximation
    # For n > 30, z=1.96 is fine; for smaller n, use approximate t-value
    if n > 1:
        se = std / math.sqrt(n)
        t_val = 1.96 if n > 30 else 2.0  # approximate
        ci_low = mean - t_val * se
        ci_high = mean + t_val * se
    else:
        ci_low = ci_high = mean

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "count": n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8a: Automated MOS evaluation"
    )
    parser.add_argument(
        "--lang", type=str, choices=["hi", "mr", "gu"], default="hi",
        help="Language code ('hi', 'mr', or 'gu', default: 'hi')"
    )
    parser.add_argument(
        "--samples_dir", type=str, default=None,
        help="Directory containing synthesized audio"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory to save results"
    )
    args = parser.parse_args()

    lang_dir_names = {"hi": "tts_hindi_female", "mr": "tts_marathi_female", "gu": "tts_gujarati_female"}
    lang_dir_name = lang_dir_names.get(args.lang, f"tts_{args.lang}_female")
    if args.samples_dir is None:
        args.samples_dir = os.path.join(PROJECT_DIR, "samples", lang_dir_name)
    if args.results_dir is None:
        args.results_dir = os.path.join(PROJECT_DIR, "results", lang_dir_name)

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print(f"PHASE 8a: AUTOMATED MOS EVALUATION ({args.lang.upper()})")
    print("=" * 60)
    print("\n  NOTE: These are AUTOMATED/PREDICTED MOS scores,")
    print("  NOT human MOS scores. They serve as supporting evidence only.")

    # Discover audio files
    print(f"\n--- Discovering audio files ---")
    eval_sets = {}

    for set_name in ["held_out", "unseen"]:
        set_dir = os.path.join(args.samples_dir, set_name)
        if not os.path.isdir(set_dir):
            print(f"  {set_name}: directory not found ({set_dir})")
            continue

        baseline_files = sorted(glob.glob(os.path.join(set_dir, "baseline_*.wav")))
        clustered_files = sorted(glob.glob(os.path.join(set_dir, "clustered_*.wav")))

        eval_sets[set_name] = {
            "baseline": baseline_files,
            "clustered": clustered_files,
        }
        print(f"  {set_name}: {len(baseline_files)} baseline, "
              f"{len(clustered_files)} clustered")

    all_wav_paths = []
    for set_data in eval_sets.values():
        all_wav_paths.extend(set_data["baseline"])
        all_wav_paths.extend(set_data["clustered"])

    if not all_wav_paths:
        print("\n  No audio files found. Run Phase 7 (inference) first.")
        sys.exit(1)

    # Score all files
    print(f"\n--- Scoring {len(all_wav_paths)} audio files ---")

    # Try UTMOS first
    scores, method = score_with_utmos(all_wav_paths)

    # Fallback to librosa heuristic
    if method is None:
        print("  speechmos not available, using librosa heuristic")
        scores, method = score_with_librosa_heuristic(all_wav_paths)

    if method is None:
        print("  ERROR: No scoring method available.")
        print("  Install: pip install librosa")
        sys.exit(1)

    print(f"  Scoring method: {method}")
    scored = sum(1 for v in scores.values() if v is not None)
    print(f"  Successfully scored: {scored}/{len(all_wav_paths)}")

    # Analyze results per condition and set
    print(f"\n--- Results ---")

    all_results = []
    report_lines = [
        "# Automated MOS Evaluation Report",
        "",
        f"> **Method:** {method}",
        "> **WARNING:** These are AUTOMATED/PREDICTED MOS scores, NOT human MOS.",
        "> They serve as supporting evidence only.",
        "",
    ]

    for set_name, set_data in eval_sets.items():
        report_lines.append(f"## {set_name.replace('_', ' ').title()} Set")
        report_lines.append("")

        for condition in ["baseline", "clustered"]:
            files = set_data[condition]
            condition_scores = [scores.get(f) for f in files]
            stats = compute_statistics(condition_scores)

            print(f"  {set_name}/{condition}:")
            print(f"    Mean: {stats['mean']} +/- {stats['std']}")
            print(f"    95% CI: [{stats['ci_95_low']}, {stats['ci_95_high']}]")
            print(f"    N: {stats['count']}")

            report_lines.append(f"### {condition.title()}")
            report_lines.append(f"- **Mean MOS:** {stats['mean']}")
            report_lines.append(f"- **Std Dev:** {stats['std']}")
            report_lines.append(f"- **95% CI:** [{stats['ci_95_low']}, {stats['ci_95_high']}]")
            report_lines.append(f"- **Sample Count:** {stats['count']}")
            report_lines.append("")

            # Per-file scores
            for f in files:
                idx = os.path.basename(f).replace(f"{condition}_", "").replace(".wav", "")
                all_results.append({
                    "set": set_name,
                    "condition": condition,
                    "index": idx,
                    "filename": os.path.basename(f),
                    "automated_mos": scores.get(f),
                    "scoring_method": method,
                })

        # Compute paired differences
        baseline_scores = {os.path.basename(f).replace("baseline_", ""): scores.get(f)
                           for f in set_data["baseline"]}
        clustered_scores = {os.path.basename(f).replace("clustered_", ""): scores.get(f)
                            for f in set_data["clustered"]}

        paired_diffs = []
        for key in baseline_scores:
            b = baseline_scores.get(key)
            c = clustered_scores.get(key)
            if b is not None and c is not None:
                paired_diffs.append(b - c)

        if paired_diffs:
            import numpy as np
            diffs = np.array(paired_diffs)
            report_lines.append("### Paired Difference (Baseline - Clustered)")
            report_lines.append(f"- **Mean Difference:** {np.mean(diffs):.4f}")
            report_lines.append(f"- **Std Dev:** {np.std(diffs, ddof=1):.4f}")
            report_lines.append(f"- **Paired Samples:** {len(diffs)}")
            report_lines.append("")

    # Save CSV
    csv_path = os.path.join(args.results_dir, "automated_mos.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  CSV saved: {csv_path}")

    # Save report
    report_path = os.path.join(args.results_dir, "automated_mos_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Report saved: {report_path}")

    print(f"\n{'=' * 60}")
    print("PHASE 8a COMPLETE")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
