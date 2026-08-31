#!/usr/bin/env python3
"""
Phase 8b: Prepare blinded human evaluation package.

Creates:
  - Randomized audio with anonymous condition IDs (System-A / System-B)
  - Evaluation form template (CSV)
  - Instructions document for evaluators
  - Each pair randomizes which condition appears first

Usage:
    python tts/tts_human_eval_prep.py [--samples_dir samples/tts_hindi_female]
"""

import os
import sys
import csv
import json
import random
import shutil
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_SAMPLES = os.path.join(PROJECT_DIR, "samples", "tts_hindi_female")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "results", "tts_hindi_female", "human_eval_package")
RANDOM_SEED = 42


def prepare_eval_package(samples_dir, output_dir):
    """
    Create blinded evaluation package:
      1. Copy and rename audio files with anonymous IDs
      2. Randomize presentation order per pair
      3. Generate evaluation form template
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    random.seed(RANDOM_SEED)

    # Collect pairs from both eval sets
    pairs = []

    for set_name in ["held_out", "unseen"]:
        set_dir = os.path.join(samples_dir, set_name)
        if not os.path.isdir(set_dir):
            continue

        baseline_files = sorted(
            [f for f in os.listdir(set_dir) if f.startswith("baseline_") and f.endswith(".wav")]
        )

        for bf in baseline_files:
            idx = bf.replace("baseline_", "").replace(".wav", "")
            cf = f"clustered_{idx}.wav"

            baseline_path = os.path.join(set_dir, bf)
            clustered_path = os.path.join(set_dir, cf)

            if os.path.exists(baseline_path) and os.path.exists(clustered_path):
                pairs.append({
                    "set": set_name,
                    "index": idx,
                    "baseline_src": baseline_path,
                    "clustered_src": clustered_path,
                })

    if not pairs:
        print("  No paired audio files found.")
        return []

    print(f"  Found {len(pairs)} paired samples")

    # Randomize global pair order
    random.shuffle(pairs)

    # Create blinded files and mapping
    mapping = []
    form_rows = []

    for pair_idx, pair in enumerate(pairs, 1):
        # Randomize which condition is A vs B for this pair
        if random.random() < 0.5:
            system_a = "baseline"
            system_b = "clustered"
            src_a = pair["baseline_src"]
            src_b = pair["clustered_src"]
        else:
            system_a = "clustered"
            system_b = "baseline"
            src_a = pair["clustered_src"]
            src_b = pair["baseline_src"]

        # Copy with anonymous names
        file_a = f"pair_{pair_idx:03d}_system_A.wav"
        file_b = f"pair_{pair_idx:03d}_system_B.wav"

        shutil.copy2(src_a, os.path.join(audio_dir, file_a))
        shutil.copy2(src_b, os.path.join(audio_dir, file_b))

        # Record mapping (NOT shared with evaluators)
        mapping.append({
            "pair_id": pair_idx,
            "set": pair["set"],
            "original_index": pair["index"],
            "system_A_is": system_a,
            "system_B_is": system_b,
            "file_A": file_a,
            "file_B": file_b,
        })

        # Evaluation form row
        form_rows.append({
            "pair_id": pair_idx,
            "file_A": file_a,
            "file_B": file_b,
            "naturalness_A": "",
            "naturalness_B": "",
            "intelligibility_A": "",
            "intelligibility_B": "",
            "preference": "",  # A, B, or No Preference
            "comments": "",
        })

    return mapping, form_rows


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8b: Prepare blinded human evaluation package"
    )
    parser.add_argument(
        "--samples_dir", type=str, default=DEFAULT_SAMPLES,
        help="Directory containing synthesized audio"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT,
        help="Output directory for evaluation package"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 8b: HUMAN EVALUATION PACKAGE PREPARATION")
    print("=" * 60)

    # Prepare package
    result = prepare_eval_package(args.samples_dir, args.output_dir)

    if not result:
        print("\n  No paired audio found. Run Phase 7 first.")
        sys.exit(1)

    mapping, form_rows = result

    # Save mapping (for experimenter only — NOT shared with evaluators)
    mapping_path = os.path.join(args.output_dir, "_CONFIDENTIAL_mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0].keys()))
        writer.writeheader()
        writer.writerows(mapping)
    print(f"\n  Mapping (CONFIDENTIAL): {mapping_path}")

    # Save evaluation form template (one per listener)
    for listener_id in range(1, 6):  # 5 listeners
        form_path = os.path.join(
            args.output_dir, f"evaluation_form_listener_{listener_id}.csv"
        )
        # Deep copy form rows
        listener_rows = []
        for row in form_rows:
            r = dict(row)
            r["listener_id"] = f"L{listener_id}"
            listener_rows.append(r)

        fieldnames = ["listener_id"] + list(form_rows[0].keys())
        with open(form_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(listener_rows)

    print(f"  Evaluation forms: 5 listener forms created")

    # Save instructions
    instructions = """# Human Evaluation Instructions

## Overview
You will evaluate pairs of Hindi speech audio samples. For each pair, you will
hear System A and System B. Your task is to rate each system and optionally
indicate your preference.

## Rating Scales

### Naturalness (1-5)
Rate how natural the speech sounds:
- 5: Completely natural, indistinguishable from human speech
- 4: Mostly natural, with minor artifacts
- 3: Somewhat natural, noticeable synthesis artifacts
- 2: Unnatural, clearly synthesized with significant issues
- 1: Very unnatural, barely intelligible or severely distorted

### Intelligibility (1-5)
Rate how easy it is to understand what is being said:
- 5: Perfectly clear, every word is easily understood
- 4: Mostly clear, one or two unclear parts
- 3: Somewhat clear, several parts are difficult to understand
- 2: Mostly unclear, only a few words are intelligible
- 1: Completely unintelligible

### Preference (Optional)
After listening to both systems in a pair:
- Write "A" if you prefer System A
- Write "B" if you prefer System B
- Write "No Preference" if they sound equally good

## Important Rules
1. Listen to each sample at least TWICE before rating
2. Use headphones in a quiet environment
3. Rate each sample independently — do not compare across pairs
4. Fill in ALL naturalness and intelligibility scores
5. The "preference" and "comments" fields are optional
6. Do not discuss your ratings with other listeners

## How to Fill the Form
1. Open the CSV file assigned to you (evaluation_form_listener_X.csv)
2. For each row, listen to the two audio files indicated
3. Enter your scores (1-5) in the appropriate columns
4. Save the file and return it to the experimenter

Thank you for your participation!
"""

    instructions_path = os.path.join(args.output_dir, "INSTRUCTIONS.md")
    with open(instructions_path, "w", encoding="utf-8") as f:
        f.write(instructions)
    print(f"  Instructions: {instructions_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("PHASE 8b COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total pairs: {len(mapping)}")
    print(f"  Audio files: {args.output_dir}/audio/")
    print(f"  Evaluation forms: 5 listener forms")
    print(f"  Instructions: {instructions_path}")
    print(f"\n  Distribute the following to each listener:")
    print(f"    1. The audio/ directory")
    print(f"    2. Their specific evaluation_form_listener_X.csv")
    print(f"    3. The INSTRUCTIONS.md")
    print(f"\n  DO NOT share _CONFIDENTIAL_mapping.csv with listeners!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
