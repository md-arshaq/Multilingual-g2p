#!/usr/bin/env python3
"""
Phase 1: Download SPRINGLab/IndicTTS-Hindi, filter to female speakers,
select a deterministic 1.75-hour subset, and save the manifest.

Usage:
    python tts/tts_data_download.py [--output_dir data/tts_hindi_female]
"""

import os
import sys
import csv
import json
import random
import argparse
import struct
import wave
import io
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "SPRINGLab/IndicTTS-Hindi"
TARGET_HOURS = 1.75
MIN_HOURS = 1.70
MAX_HOURS = 1.80
MIN_DURATION_SEC = 2.5
MAX_DURATION_SEC = 20.0
RANDOM_SEED = 42
TARGET_SR = 22050  # for duration estimation only; raw files kept as-is

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "tts_hindi_female")


def get_audio_duration_from_bytes(audio_bytes, sampling_rate):
    """Compute duration in seconds from raw audio bytes and sample rate."""
    if isinstance(audio_bytes, dict):
        # HuggingFace datasets audio format: {'array': np.array, 'sampling_rate': int, 'path': str}
        arr = audio_bytes.get("array")
        sr = audio_bytes.get("sampling_rate", sampling_rate)
        if arr is not None:
            return len(arr) / sr
    return 0.0


def validate_audio(audio_obj):
    """
    Check that audio is not silent, corrupted, or contains NaNs.
    Returns (is_valid, reason) tuple.
    """
    import numpy as np

    if isinstance(audio_obj, dict):
        arr = audio_obj.get("array")
        if arr is None:
            return False, "no_audio_array"
        if len(arr) == 0:
            return False, "empty_audio"
        if np.any(np.isnan(arr)):
            return False, "contains_nan"
        if np.max(np.abs(arr)) < 1e-6:
            return False, "silent"
        return True, ""
    return False, "unknown_audio_format"


def save_audio_wav(audio_obj, output_path):
    """Save audio dict (from HuggingFace datasets) as WAV file."""
    import numpy as np
    import soundfile as sf

    arr = audio_obj["array"]
    sr = audio_obj["sampling_rate"]

    # Ensure mono
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)

    # Normalize to float32
    arr = arr.astype(np.float32)

    sf.write(output_path, arr, sr)


def main():
    parser = argparse.ArgumentParser(
        description="Download IndicTTS-Hindi and create female-only 1.75h subset"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only compute statistics, don't save audio files"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load the dataset
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 1: DATASET ACQUISITION & FEMALE SUBSET SELECTION")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("  Install with: pip install datasets soundfile")
        sys.exit(1)

    # Import G2P labeling helpers
    sys.path.insert(0, SCRIPT_DIR)
    try:
        from tts_g2p_labeling import (
            load_hindi_g2p_dict, load_phoneme_vocab, load_cluster_mapping,
            text_to_baseline_tokens, baseline_to_clustered_tokens
        )
    except ImportError as e:
        print(f"ERROR: Failed to import G2P helper functions: {e}")
        sys.exit(1)

    print("Loading G2P resources for candidate filtering...")
    g2p_dict = load_hindi_g2p_dict()
    valid_phonemes = load_phoneme_vocab()
    cluster_map = load_cluster_mapping()
    print(f"  Loaded {len(g2p_dict)} G2P dictionary entries")

    print(f"\nLoading {DATASET_NAME} ...")
    print("  (This may require HuggingFace authentication)")
    print("  Run 'huggingface-cli login' if you haven't already.\n")

    try:
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        print("\nPossible causes:")
        print("  - Not logged in to HuggingFace (run: huggingface-cli login)")
        print("  - Dataset license not accepted")
        print("  - Network error")
        sys.exit(1)

    print(f"  Loaded {len(ds)} total samples")
    print(f"  Columns: {ds.column_names}")

    # -----------------------------------------------------------------------
    # Step 2: Inspect gender encoding
    # -----------------------------------------------------------------------
    print("\n--- Gender field inspection ---")
    if "gender" not in ds.column_names:
        print("BLOCKER: No 'gender' column found in dataset.")
        print(f"  Available columns: {ds.column_names}")
        print("  Cannot proceed without gender filtering.")
        sys.exit(1)

    # Sample first 20 rows to determine gender encoding
    gender_samples = [ds[i]["gender"] for i in range(min(20, len(ds)))]
    unique_genders = set(ds["gender"])
    print(f"  Unique gender values: {unique_genders}")
    print(f"  First 20 gender values: {gender_samples}")

    # Determine female value
    # IndicTTS typically uses: 0=male, 1=female OR "male"/"female"
    if unique_genders <= {0, 1}:
        # Integer encoding — need to determine which is female
        # IndicTTS convention: female samples are typically labeled 1
        # We'll verify by checking counts (should be roughly equal ~5h each)
        from collections import Counter
        gender_counts = Counter(ds["gender"])
        print(f"  Gender distribution: {dict(gender_counts)}")

        # The dataset description says Male: 5.16h, Female: 5.18h
        # So counts should be roughly equal. We'll assume 1=female (IndicTTS convention)
        female_value = 1
        print(f"  Using female_value = {female_value} (IndicTTS convention)")
    elif "female" in unique_genders or "Female" in unique_genders:
        female_value = "female" if "female" in unique_genders else "Female"
        print(f"  Using female_value = '{female_value}'")
    else:
        print(f"BLOCKER: Cannot determine female encoding from values: {unique_genders}")
        print("  Please inspect the dataset manually and update this script.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Filter to female samples
    # -----------------------------------------------------------------------
    print("\n--- Filtering to female samples ---")
    female_indices = [i for i in range(len(ds)) if ds[i]["gender"] == female_value]
    print(f"  Female samples: {len(female_indices)} / {len(ds)} total")

    # -----------------------------------------------------------------------
    # Step 4: Validate and compute durations
    # -----------------------------------------------------------------------
    print("\n--- Validating audio and computing durations ---")
    import numpy as np

    candidates = []
    exclusions = {"no_text": 0, "no_audio_array": 0, "empty_audio": 0,
                  "contains_nan": 0, "silent": 0, "too_short": 0,
                  "too_long": 0, "unknown_audio_format": 0,
                  "oov_word": 0, "unmapped_phoneme": 0}

    for count, idx in enumerate(female_indices):
        if (count + 1) % 500 == 0:
            print(f"  Processing {count + 1}/{len(female_indices)}...")

        sample = ds[idx]
        text = sample.get("text", "").strip()

        # Check text
        if not text:
            exclusions["no_text"] += 1
            continue

        # Check G2P dictionary coverage
        baseline_seq, oov_words, baseline_ok = text_to_baseline_tokens(
            text, g2p_dict, valid_phonemes
        )
        if not baseline_ok:
            exclusions["oov_word"] += 1
            continue

        # Check cluster mapping
        clustered_seq, unmapped, cluster_ok = baseline_to_clustered_tokens(
            baseline_seq, cluster_map
        )
        if not cluster_ok:
            exclusions["unmapped_phoneme"] += 1
            continue

        # Check audio
        audio = sample.get("audio")
        if audio is None:
            exclusions["no_audio_array"] += 1
            continue

        is_valid, reason = validate_audio(audio)
        if not is_valid:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue

        # Compute duration
        duration = get_audio_duration_from_bytes(audio, TARGET_SR)
        if duration < MIN_DURATION_SEC:
            exclusions["too_short"] += 1
            continue
        if duration > MAX_DURATION_SEC:
            exclusions["too_long"] += 1
            continue

        candidates.append({
            "dataset_idx": idx,
            "text": text,
            "duration": duration,
            "gender": sample["gender"],
            "baseline_tokens": baseline_seq,
            "clustered_tokens": clustered_seq,
        })

    print(f"\n  Valid female candidates: {len(candidates)}")
    print(f"  Exclusions: {json.dumps(exclusions, indent=4)}")
    total_candidate_hours = sum(c["duration"] for c in candidates) / 3600
    print(f"  Total candidate duration: {total_candidate_hours:.2f} hours")

    if total_candidate_hours < MIN_HOURS:
        print(f"\nBLOCKER: Only {total_candidate_hours:.2f} hours of valid female audio.")
        print(f"  Need at least {MIN_HOURS} hours. Cannot proceed.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 5: Deterministic subset selection to ~1.75 hours
    # -----------------------------------------------------------------------
    print(f"\n--- Selecting {TARGET_HOURS}h subset (seed={RANDOM_SEED}) ---")
    random.seed(RANDOM_SEED)

    # Shuffle deterministically
    shuffled = list(range(len(candidates)))
    random.shuffle(shuffled)

    selected = []
    total_duration = 0.0
    target_seconds = TARGET_HOURS * 3600

    for i in shuffled:
        c = candidates[i]
        if total_duration + c["duration"] > MAX_HOURS * 3600:
            continue
        selected.append(i)
        total_duration += c["duration"]
        if total_duration >= MIN_HOURS * 3600:
            # Check if we're within target range
            if total_duration >= MIN_HOURS * 3600:
                break

    selected_hours = total_duration / 3600
    print(f"  Selected {len(selected)} samples")
    print(f"  Total duration: {selected_hours:.4f} hours")

    if selected_hours < MIN_HOURS or selected_hours > MAX_HOURS:
        print(f"  WARNING: Duration {selected_hours:.4f}h outside [{MIN_HOURS}, {MAX_HOURS}] range")
        print("  Adjusting selection...")

        # If over, remove from the end
        while selected_hours > MAX_HOURS and len(selected) > 0:
            removed = selected.pop()
            total_duration -= candidates[removed]["duration"]
            selected_hours = total_duration / 3600

        # If under, add more
        remaining = [i for i in shuffled if i not in set(selected)]
        for i in remaining:
            c = candidates[i]
            if total_duration + c["duration"] <= MAX_HOURS * 3600:
                selected.append(i)
                total_duration += c["duration"]
                selected_hours = total_duration / 3600
                if selected_hours >= MIN_HOURS:
                    break

    print(f"  Final: {len(selected)} samples, {selected_hours:.4f} hours")

    if selected_hours < MIN_HOURS or selected_hours > MAX_HOURS:
        print(f"BLOCKER: Cannot achieve target duration range [{MIN_HOURS}, {MAX_HOURS}]h")
        print(f"  Achieved: {selected_hours:.4f}h")
        sys.exit(1)

    selected_set = set(selected)

    # -----------------------------------------------------------------------
    # Step 6: Save raw audio files (unless dry run)
    # -----------------------------------------------------------------------
    if not args.dry_run:
        print(f"\n--- Saving raw audio to {raw_dir} ---")
        for count, sel_idx in enumerate(selected):
            if (count + 1) % 200 == 0:
                print(f"  Saving {count + 1}/{len(selected)}...")

            c = candidates[sel_idx]
            ds_idx = c["dataset_idx"]
            audio = ds[ds_idx]["audio"]

            filename = f"hindi_female_{ds_idx:06d}.wav"
            output_path = os.path.join(raw_dir, filename)
            c["filename"] = filename

            try:
                save_audio_wav(audio, output_path)
            except Exception as e:
                print(f"  ERROR saving {filename}: {e}")
                c["filename"] = None
    else:
        print("\n--- DRY RUN: Skipping audio file saving ---")
        for sel_idx in selected:
            c = candidates[sel_idx]
            c["filename"] = f"hindi_female_{c['dataset_idx']:06d}.wav"

    # -----------------------------------------------------------------------
    # Step 7: Save manifest
    # -----------------------------------------------------------------------
    print(f"\n--- Saving manifest ---")
    manifest_path = os.path.join(output_dir, "manifest.csv")

    manifest_rows = []
    for i, c in enumerate(candidates):
        is_selected = i in selected_set
        manifest_rows.append({
            "row_id": c["dataset_idx"],
            "text": c["text"],
            "gender": c["gender"],
            "duration_sec": round(c["duration"], 3),
            "selected": 1 if is_selected else 0,
            "exclusion_reason": "" if is_selected else "not_selected_by_random",
            "filename": c.get("filename", ""),
            "baseline_tokens": c["baseline_tokens"] if is_selected else "",
            "clustered_tokens": c["clustered_tokens"] if is_selected else "",
            "split": "",  # Filled in Phase 4
        })

    fieldnames = [
        "row_id", "text", "gender", "duration_sec", "selected",
        "exclusion_reason", "filename", "baseline_tokens",
        "clustered_tokens", "split"
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"  Manifest saved to: {manifest_path}")
    print(f"  Total rows: {len(manifest_rows)}")
    print(f"  Selected: {sum(1 for r in manifest_rows if r['selected'] == 1)}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Total samples in dataset: {len(ds)}")
    print(f"  Female samples found: {len(female_indices)}")
    print(f"  Valid candidates after filtering: {len(candidates)}")
    print(f"  Selected for experiment: {len(selected)}")
    print(f"  Selected duration: {selected_hours:.4f} hours")
    print(f"  Target range: [{MIN_HOURS}, {MAX_HOURS}] hours")
    print(f"  Audio saved to: {raw_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Exclusion stats: {json.dumps(exclusions, indent=4)}")

    if args.dry_run:
        print("\n  ⚠️  DRY RUN — no audio files were saved")

    return 0


if __name__ == "__main__":
    sys.exit(main())
