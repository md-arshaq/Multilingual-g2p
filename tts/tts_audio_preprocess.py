#!/usr/bin/env python3
"""
Phase 3: Audio preprocessing for the TTS experiment.

For each selected sample:
  - Convert to mono WAV at 22,050 Hz
  - Trim leading/trailing silence
  - Apply consistent loudness normalization
  - Reject processing failures (empty, NaN, clipping)

Both baseline and clustered models use identical processed audio.

Usage:
    python tts/tts_audio_preprocess.py [--manifest data/tts_hindi_female/manifest.csv]
"""

import os
import sys
import csv
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST = os.path.join(PROJECT_DIR, "data", "tts_hindi_female", "manifest.csv")
DEFAULT_RAW_DIR = os.path.join(PROJECT_DIR, "data", "tts_hindi_female", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "tts_hindi_female", "processed")

TARGET_SR = 22050
TRIM_TOP_DB = 25
TARGET_LUFS = -23.0  # Target loudness in LUFS (EBU R128)
MAX_AMPLITUDE = 0.99  # Clipping threshold


def rms_normalize(audio, target_db=-20.0):
    """RMS-based loudness normalization as fallback."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return audio * (target_rms / rms)


def process_audio(input_path, output_path, target_sr=TARGET_SR):
    """
    Process a single audio file:
      1. Load and resample to target_sr
      2. Convert to mono
      3. Trim silence
      4. Normalize loudness
      5. Validate output

    Returns: (success, duration_sec, error_msg)
    """
    import librosa
    import soundfile as sf

    try:
        # Load and resample
        y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    except Exception as e:
        return False, 0.0, f"load_error: {e}"

    if len(y) == 0:
        return False, 0.0, "empty_after_load"

    # Check for NaN
    if np.any(np.isnan(y)):
        return False, 0.0, "contains_nan_after_load"

    # Trim silence
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    except Exception as e:
        return False, 0.0, f"trim_error: {e}"

    if len(y_trimmed) == 0:
        return False, 0.0, "empty_after_trim"

    # Loudness normalization
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(target_sr)
        loudness = meter.integrated_loudness(y_trimmed)
        if np.isinf(loudness) or np.isnan(loudness):
            # Fallback to RMS normalization
            y_normalized = rms_normalize(y_trimmed)
        else:
            y_normalized = pyln.normalize.loudness(y_trimmed, loudness, TARGET_LUFS)
    except ImportError:
        # pyloudnorm not available, use RMS normalization
        y_normalized = rms_normalize(y_trimmed)
    except Exception:
        # Any other error, use RMS
        y_normalized = rms_normalize(y_trimmed)

    # Post-normalization validation
    if np.any(np.isnan(y_normalized)):
        return False, 0.0, "nan_after_normalization"

    # Check clipping
    max_amp = np.max(np.abs(y_normalized))
    if max_amp > MAX_AMPLITUDE:
        # Soft clip: scale down to prevent clipping
        y_normalized = y_normalized * (MAX_AMPLITUDE / max_amp)

    if len(y_normalized) == 0:
        return False, 0.0, "empty_after_normalization"

    duration = len(y_normalized) / target_sr

    # Write output
    try:
        sf.write(output_path, y_normalized, target_sr)
    except Exception as e:
        return False, 0.0, f"write_error: {e}"

    return True, duration, ""


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Audio preprocessing for TTS experiment"
    )
    parser.add_argument(
        "--lang", type=str, default="hi", choices=["hi", "mr", "gu"],
        help="Language code ('hi', 'mr', or 'gu', default: 'hi')"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help="Directory containing raw audio files"
    )
    parser.add_argument(
        "--processed_dir", type=str, default=None,
        help="Output directory for processed audio"
    )
    args = parser.parse_args()

    lang_dir_names = {"hi": "tts_hindi_female", "mr": "tts_marathi_female", "gu": "tts_gujarati_female"}
    lang_dir_name = lang_dir_names.get(args.lang, f"tts_{args.lang}_female")
    base_dir = os.path.join(PROJECT_DIR, "data", lang_dir_name)
    if args.manifest is None:
        args.manifest = os.path.join(base_dir, "manifest.csv")
    if args.raw_dir is None:
        args.raw_dir = os.path.join(base_dir, "raw")
    if args.processed_dir is None:
        args.processed_dir = os.path.join(base_dir, "processed")

    os.makedirs(args.processed_dir, exist_ok=True)

    print("=" * 60)
    print(f"PHASE 3: AUDIO PREPROCESSING ({args.lang.upper()})")
    print("=" * 60)

    # Check dependencies
    try:
        import librosa
        import soundfile as sf
        print(f"  librosa version: {librosa.__version__}")
    except ImportError:
        print("ERROR: librosa and soundfile are required.")
        print("  Install with: pip install librosa soundfile")
        sys.exit(1)

    try:
        import pyloudnorm
        print("  pyloudnorm is available")
        loudnorm_available = True
    except ImportError:
        print("  pyloudnorm not available; using RMS normalization as fallback")
        loudnorm_available = False

    # Load manifest
    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    selected = [r for r in manifest if r["selected"] == "1"]
    print(f"  Selected samples to process: {len(selected)}")

    # Process audio
    print(f"\n--- Processing audio ---")
    print(f"  Input:  {args.raw_dir}")
    print(f"  Output: {args.processed_dir}")
    print(f"  Target: {TARGET_SR} Hz, mono, trimmed, loudness-normalized")

    success_count = 0
    fail_count = 0
    total_duration = 0.0
    failures = []

    for i, row in enumerate(manifest):
        if row["selected"] != "1":
            continue

        filename = row.get("filename", "")
        if not filename:
            failures.append((row["row_id"], "no_filename"))
            row["selected"] = "0"
            row["exclusion_reason"] = "no_filename"
            fail_count += 1
            continue

        input_path = os.path.join(args.raw_dir, filename)
        output_path = os.path.join(args.processed_dir, filename)

        if not os.path.exists(input_path):
            failures.append((filename, "file_not_found"))
            row["selected"] = "0"
            row["exclusion_reason"] = "raw_file_not_found"
            fail_count += 1
            continue

        success, duration, error = process_audio(input_path, output_path)

        if success:
            success_count += 1
            total_duration += duration
            # Update duration with processed duration
            row["duration_sec"] = str(round(duration, 3))
        else:
            failures.append((filename, error))
            row["selected"] = "0"
            row["exclusion_reason"] = f"preprocessing_failed:{error}"
            fail_count += 1

        if (i + 1) % 200 == 0 or i == len(manifest) - 1:
            print(f"  [{success_count + fail_count}/{len(selected)}] "
                  f"OK: {success_count}, Failed: {fail_count}")

    # Verify processed files
    print(f"\n--- Verifying processed files ---")
    final_selected = [r for r in manifest if r["selected"] == "1"]
    verified = 0
    for row in final_selected:
        proc_path = os.path.join(args.processed_dir, row["filename"])
        if os.path.exists(proc_path) and os.path.getsize(proc_path) > 0:
            verified += 1
        else:
            row["selected"] = "0"
            row["exclusion_reason"] = "processed_file_missing"

    # Save updated manifest
    print(f"\n--- Saving updated manifest ---")
    fieldnames = list(manifest[0].keys())
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    # Log failures
    if failures:
        fail_log = os.path.join(os.path.dirname(args.manifest), "preprocessing_failures.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            for filename, error in failures:
                f.write(f"{filename}\t{error}\n")
        print(f"  Failures logged to: {fail_log}")

    # Final summary
    final_selected = [r for r in manifest if r["selected"] == "1"]
    final_hours = sum(float(r["duration_sec"]) for r in final_selected) / 3600

    print(f"\n{'=' * 60}")
    print("PHASE 3 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Input samples:      {len(selected)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Processing failures:    {fail_count}")
    print(f"  Verified on disk:       {verified}")
    print(f"  Final selected:         {len(final_selected)}")
    print(f"  Final duration:         {final_hours:.4f} hours")
    print(f"  Normalization method:   {'pyloudnorm (LUFS)' if loudnorm_available else 'RMS'}")
    print(f"  Processed audio:        {args.processed_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
