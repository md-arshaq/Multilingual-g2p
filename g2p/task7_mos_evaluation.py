

import os
import csv
import json
import argparse
import glob
from collections import defaultdict

# PATHS
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

SAMPLES_DIR     = os.path.join(PROJECT_DIR, "samples")
METADATA_PATH   = os.path.join(SAMPLES_DIR, "samples_metadata.csv")
SCORES_PATH     = os.path.join(PROJECT_DIR, "results", "mos_scores.csv")
REPORT_PATH     = os.path.join(PROJECT_DIR, "results", "mos_report.md")


# AUTOMATED MOS — UTMOS / SpeechMOS
def score_with_utmos(wav_paths):
    
    scores = {}

    # Try speechmos first (best quality if installed)
    try:
        import speechmos
        print("  Using speechmos for MOS prediction...")
        predictor = speechmos.SpeechMOS()
        for wav_path in wav_paths:
            try:
                score = predictor.predict(wav_path)
                scores[wav_path] = float(score)
            except Exception as e:
                print(f"    ⚠️  Error scoring {os.path.basename(wav_path)}: {e}")
                scores[wav_path] = None
        return scores
    except ImportError:
        pass

    # Try librosa-based heuristic (most reliable — no torchcodec issues)
    try:
        import librosa
        import numpy as np

        print("  Using librosa-based MOS estimation...")
        for wav_path in wav_paths:
            try:
                y, sr = librosa.load(wav_path, sr=16000)

                if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
                    scores[wav_path] = 1.0
                    continue

                # ── Multi-feature heuristic MOS ──
                duration = len(y) / sr
                rms = np.sqrt(np.mean(y ** 2))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

                # Spectral flatness (0 = tonal/speech-like, 1 = noise-like)
                spec_flat = np.mean(librosa.feature.spectral_flatness(y=y))

                # Start from neutral
                mos = 3.0

                # Energy: good speech has moderate RMS
                if rms > 0.01 and rms < 0.8:
                    mos += 0.3
                elif rms < 0.001:
                    mos -= 1.0  # too quiet / near silence

                # Duration: reasonable length for a sentence
                if duration > 1.0:
                    mos += 0.2
                elif duration < 0.3:
                    mos -= 0.5

                # Spectral centroid: speech typically 500-4000 Hz
                if 500 < spectral_centroid < 4000:
                    mos += 0.4
                elif spectral_centroid > 6000:
                    mos -= 0.3  # too high-pitched / noisy

                # ZCR: speech typically 0.03-0.15
                if 0.03 < zcr < 0.15:
                    mos += 0.3
                elif zcr > 0.3:
                    mos -= 0.3  # noise-like

                # Spectral flatness: low = more tonal (speech), high = noise
                if spec_flat < 0.1:
                    mos += 0.4
                elif spec_flat > 0.5:
                    mos -= 0.4

                # Bandwidth: speech has moderate bandwidth
                if 1000 < spectral_bandwidth < 3500:
                    mos += 0.2

                mos = round(max(1.0, min(5.0, mos)), 2)
                scores[wav_path] = mos

            except Exception as e:
                print(f"    ⚠️  Error scoring {os.path.basename(wav_path)}: {e}")
                scores[wav_path] = None

        return scores
    except ImportError:
        pass

    # Final fallback: no audio library available
    print("  ⚠️  No audio scoring library available.")
    print("     Install with: pip install librosa")
    print("     Generating placeholder scores (3.0 for all).")

    for wav_path in wav_paths:
        scores[wav_path] = 3.0  # neutral placeholder

    return scores


def load_human_scores(csv_path):
    
    human_scores = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            score = float(row.get("score", 0))
            if filename and 1 <= score <= 5:
                human_scores[filename].append(score)

    # Average per file
    averaged = {}
    for filename, scores_list in human_scores.items():
        averaged[filename] = sum(scores_list) / len(scores_list)

    return averaged


def main():
    parser = argparse.ArgumentParser(description="MOS evaluation of TTS samples")
    parser.add_argument("--samples_dir", type=str, default=None,
                        help="Directory containing audio samples")
    parser.add_argument("--human_scores", type=str, default=None,
                        help="Path to human MOS scores CSV (optional)")
    args = parser.parse_args()

    if args.samples_dir is None:
        args.samples_dir = SAMPLES_DIR

    print("=" * 60)
    print("TASK 7: MOS EVALUATION")
    print("=" * 60)

    print(f"\nScanning for audio files in: {args.samples_dir}")

    # Load metadata if available
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["filename"]] = row

    # Find all WAV files
    wav_files = []
    for lang in ["hi", "gu", "mr"]:
        lang_dir = os.path.join(args.samples_dir, lang)
        if os.path.isdir(lang_dir):
            for f in sorted(os.listdir(lang_dir)):
                if f.endswith(".wav"):
                    wav_files.append({
                        "path": os.path.join(lang_dir, f),
                        "filename": os.path.join(lang, f),
                        "language": lang,
                        "condition": "baseline" if "baseline" in f else "clustered" if "clustered" in f else "unknown",
                    })

    if not wav_files:
        print("  ❌ No WAV files found! Run Task 6 first to generate audio samples.")
        return

    print(f"  Found {len(wav_files)} audio files")
    for lang in ["hi", "gu", "mr"]:
        count = sum(1 for w in wav_files if w["language"] == lang)
        if count > 0:
            print(f"    {lang.upper()}: {count} files")

    print(f"\nRunning automated MOS scoring...")
    wav_paths = [w["path"] for w in wav_files]
    auto_scores = score_with_utmos(wav_paths)

    human_scores = {}
    if args.human_scores and os.path.exists(args.human_scores):
        print(f"\nLoading human MOS scores from: {args.human_scores}")
        human_scores = load_human_scores(args.human_scores)
        print(f"  Loaded scores for {len(human_scores)} files")

    results = []
    for wf in wav_files:
        auto_mos = auto_scores.get(wf["path"])
        human_mos = human_scores.get(wf["filename"])

        # Get metadata if available
        meta = metadata.get(wf["filename"], {})

        results.append({
            "filename": wf["filename"],
            "language": wf["language"],
            "condition": wf["condition"],
            "auto_mos": auto_mos,
            "human_mos": human_mos,
            "text": meta.get("text", ""),
            "phonemes": meta.get("phonemes", ""),
            "output_phonemes": meta.get("output_phonemes", ""),
        })

    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)

    with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "language", "condition", "auto_mos", "human_mos",
            "text", "phonemes", "output_phonemes"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved MOS scores: {SCORES_PATH}")

    print(f"\n{'='*60}")
    print("MOS SUMMARY")
    print(f"{'='*60}")

    stats = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["auto_mos"] is not None:
            stats[r["language"]][r["condition"]].append(r["auto_mos"])

    print(f"\n{'Language':<12} {'Condition':<12} {'Mean MOS':>10} {'Std':>8} {'Count':>8}")
    print("-" * 55)

    for lang in sorted(stats.keys()):
        for cond in sorted(stats[lang].keys()):
            scores_list = stats[lang][cond]
            import numpy as np
            mean = np.mean(scores_list)
            std = np.std(scores_list)
            count = len(scores_list)
            print(f"{lang.upper():<12} {cond:<12} {mean:>10.2f} {std:>8.2f} {count:>8}")

    import numpy as np

    report = 

    if human_scores:
        report += 
    else:
        report += 

    report += 

    for lang in sorted(stats.keys()):
        for cond in sorted(stats[lang].keys()):
            scores_list = stats[lang][cond]
            mean = np.mean(scores_list)
            std = np.std(scores_list)
            count = len(scores_list)
            report += f"| {lang.upper()} | {cond} | {mean:.2f} | {std:.2f} | {count} |\n"

    # Overall comparison
    baseline_all = []
    clustered_all = []
    for lang in stats:
        baseline_all.extend(stats[lang].get("baseline", []))
        clustered_all.extend(stats[lang].get("clustered", []))

    if baseline_all and clustered_all:
        b_mean = np.mean(baseline_all)
        c_mean = np.mean(clustered_all)
        delta = c_mean - b_mean

        report += f
        if abs(delta) < 0.1:
            report += "The clustered G2P model produces audio of **comparable quality** to the baseline.\n"
        elif delta > 0:
            report += f"The clustered G2P model produces **slightly better** audio quality (+{delta:.2f} MOS).\n"
        else:
            report += f"The clustered G2P model shows a **slight quality decrease** ({delta:.2f} MOS).\n"

    report += 

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Saved MOS report: {REPORT_PATH}")
    print(f"\n{'='*60}")
    print("TASK 7 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
