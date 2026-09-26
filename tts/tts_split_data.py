#!/usr/bin/env python3
"""
Phase 4: Data splitting and VITS metadata file generation.

Creates:
  - Immutable 90/5/5 train/val/test split (seed=42)
  - VITS-format metadata files (pipe-delimited) for both baseline and clustered
  - Split ID lists for reproducibility verification

Usage:
    python tts/tts_split_data.py [--manifest data/tts_hindi_female/manifest.csv]
    python tts/tts_split_data.py --verify  # Validate split integrity
"""

import os
import sys
import csv
import json
import random
import argparse
from collections import Counter
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST = os.path.join(PROJECT_DIR, "data", "tts_hindi_female", "manifest.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "tts_hindi_female")

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42


def normalized_text_key(text):
    """Return a stable grouping key so identical utterances cannot leak splits."""
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold()


def create_split(manifest, output_dir):
    """
    Split selected samples into train/val/test.
    Returns updated manifest with split assignments.
    """
    selected = [r for r in manifest if r["selected"] == "1"]
    n = len(selected)

    if n == 0:
        print("ERROR: No selected samples to split")
        return manifest

    # Group duplicate normalized text before splitting. A row-level random split
    # can put the same sentence into train and test, inflating test results.
    groups = {}
    for row in selected:
        groups.setdefault(normalized_text_key(row["text"]), []).append(row)

    group_items = list(groups.items())
    random.Random(RANDOM_SEED).shuffle(group_items)
    target_counts = {
        "train": round(n * TRAIN_RATIO),
        "val": round(n * VAL_RATIO),
    }
    target_counts["test"] = n - target_counts["train"] - target_counts["val"]
    assigned_counts = Counter()
    split_map = {}

    for _, group_rows in group_items:
        group_size = len(group_rows)
        # Prefer the split furthest below its target after accounting for the
        # whole group. This keeps every duplicate text in one partition.
        split_name = max(
            target_counts,
            key=lambda name: (target_counts[name] - assigned_counts[name], name),
        )
        for row in group_rows:
            split_map[row["row_id"]] = split_name
        assigned_counts[split_name] += group_size

    print(f"  Unique normalized texts: {len(groups)} (from {n} selected rows)")

    # Update manifest
    for row in manifest:
        if row["row_id"] in split_map:
            row["split"] = split_map[row["row_id"]]

    # Save split ID lists
    for split_name in ["train", "val", "test"]:
        ids = [r["row_id"] for r in manifest if r.get("split") == split_name]
        list_path = os.path.join(output_dir, f"{split_name}_ids.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for id_ in ids:
                f.write(f"{id_}\n")
        print(f"  {split_name}: {len(ids)} samples → {list_path}")

    return manifest


def generate_vits_metadata(manifest, output_dir):
    """
    Generate VITS-format metadata files (pipe-delimited, LJSpeech-style).

    Creates for each split:
      - metadata_baseline.csv: filename|baseline_token_sequence
      - metadata_clustered.csv: filename|clustered_token_sequence
    """
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_rows = [r for r in manifest
                      if r.get("split") == split_name and r["selected"] == "1"]

        if not split_rows:
            print(f"  WARNING: No samples for split '{split_name}'")
            continue

        # Baseline metadata
        baseline_path = os.path.join(split_dir, "metadata_baseline.csv")
        with open(baseline_path, "w", newline="", encoding="utf-8") as f:
            for r in split_rows:
                filename = os.path.splitext(r["filename"])[0]  # Remove .wav
                f.write(f"{filename}|{r['baseline_tokens']}\n")

        # Clustered metadata
        clustered_path = os.path.join(split_dir, "metadata_clustered.csv")
        with open(clustered_path, "w", newline="", encoding="utf-8") as f:
            for r in split_rows:
                filename = os.path.splitext(r["filename"])[0]
                f.write(f"{filename}|{r['clustered_tokens']}\n")

        print(f"  {split_name}: {len(split_rows)} samples → "
              f"{baseline_path}, {clustered_path}")


def verify_split(manifest, output_dir):
    """Verify split integrity and fairness constraints."""
    print("\n--- Split Verification ---")
    errors = []

    selected = [r for r in manifest if r["selected"] == "1"]
    total_duration = sum(float(r["duration_sec"]) for r in selected) / 3600

    # Check duration range (relaxed to support trimmed files)
    print(f"  Total duration: {total_duration:.4f} hours")
    if total_duration < 1.50 or total_duration > 1.80:
        errors.append(f"Duration {total_duration:.4f}h outside [1.50, 1.80] range")

    # Check all selected have valid labels
    for r in selected:
        if not r.get("baseline_tokens"):
            errors.append(f"Row {r['row_id']}: missing baseline_tokens")
        if not r.get("clustered_tokens"):
            errors.append(f"Row {r['row_id']}: missing clustered_tokens")

    # Check split assignment
    split_counts = Counter(r.get("split", "") for r in selected)
    print(f"  Split distribution: {dict(split_counts)}")

    for split_name in ["train", "val", "test"]:
        if split_name not in split_counts:
            errors.append(f"Missing split: {split_name}")

    # Check no overlap between splits
    split_ids = {}
    for split_name in ["train", "val", "test"]:
        ids = set(r["row_id"] for r in selected if r.get("split") == split_name)
        split_ids[split_name] = ids

    for s1 in split_ids:
        for s2 in split_ids:
            if s1 >= s2:
                continue
            overlap = split_ids[s1] & split_ids[s2]
            if overlap:
                errors.append(f"Overlap between {s1} and {s2}: {len(overlap)} samples")

    # Check that duplicate text did not leak across partitions.
    split_texts = {
        split_name: {
            normalized_text_key(r["text"])
            for r in selected if r.get("split") == split_name
        }
        for split_name in ["train", "val", "test"]
    }
    for i, s1 in enumerate(["train", "val", "test"]):
        for s2 in ["train", "val", "test"][i + 1:]:
            overlap = split_texts[s1] & split_texts[s2]
            if overlap:
                errors.append(
                    f"Text leakage between {s1} and {s2}: {len(overlap)} normalized texts"
                )

    # Check WAV filename consistency between baseline and clustered metadata
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split_name)
        baseline_meta = os.path.join(split_dir, "metadata_baseline.csv")
        clustered_meta = os.path.join(split_dir, "metadata_clustered.csv")

        if not os.path.exists(baseline_meta) or not os.path.exists(clustered_meta):
            errors.append(f"Missing metadata file for {split_name}")
            continue

        with open(baseline_meta, "r", encoding="utf-8") as f:
            baseline_filenames = [line.split("|")[0] for line in f if line.strip()]
        with open(clustered_meta, "r", encoding="utf-8") as f:
            clustered_filenames = [line.split("|")[0] for line in f if line.strip()]

        if baseline_filenames != clustered_filenames:
            errors.append(f"{split_name}: baseline and clustered metadata have "
                          f"different filenames ({len(baseline_filenames)} vs "
                          f"{len(clustered_filenames)})")
        else:
            print(f"  {split_name}: [OK] Baseline and clustered metadata match "
                  f"({len(baseline_filenames)} files)")

    if errors:
        print(f"\n  [FAIL] VERIFICATION FAILED:")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print(f"\n  [OK] ALL CHECKS PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Data splitting and VITS metadata generation"
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
        "--output_dir", type=str, default=None,
        help="Output directory for split files and metadata"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only run verification checks (no splitting)"
    )
    args = parser.parse_args()

    lang_dir_names = {"hi": "tts_hindi_female", "mr": "tts_marathi_female", "gu": "tts_gujarati_female"}
    lang_dir_name = lang_dir_names.get(args.lang, f"tts_{args.lang}_female")
    base_dir = os.path.join(PROJECT_DIR, "data", lang_dir_name)
    if args.manifest is None:
        args.manifest = os.path.join(base_dir, "manifest.csv")
    if args.output_dir is None:
        args.output_dir = base_dir

    print("=" * 60)
    print(f"PHASE 4: DATA SPLIT & METADATA GENERATION ({args.lang.upper()})")
    print("=" * 60)

    # Load manifest
    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    selected = [r for r in manifest if r["selected"] == "1"]
    print(f"  Total rows: {len(manifest)}")
    print(f"  Selected: {len(selected)}")
    total_hours = sum(float(r["duration_sec"]) for r in selected) / 3600
    print(f"  Total duration: {total_hours:.4f} hours")

    if args.verify:
        ok = verify_split(manifest, args.output_dir)
        sys.exit(0 if ok else 1)

    # Create splits
    print(f"\n--- Creating splits (seed={RANDOM_SEED}) ---")
    print(f"  Ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    manifest = create_split(manifest, args.output_dir)

    # Generate VITS metadata
    print(f"\n--- Generating VITS metadata ---")
    generate_vits_metadata(manifest, args.output_dir)

    # Save updated manifest
    print(f"\n--- Saving updated manifest ---")
    fieldnames = list(manifest[0].keys())
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"  Updated: {args.manifest}")

    # Run verification
    ok = verify_split(manifest, args.output_dir)

    # Final summary
    print(f"\n{'=' * 60}")
    print("PHASE 4 COMPLETE")
    print(f"{'=' * 60}")

    for split_name in ["train", "val", "test"]:
        split_rows = [r for r in manifest
                      if r.get("split") == split_name and r["selected"] == "1"]
        split_duration = sum(float(r["duration_sec"]) for r in split_rows) / 3600
        print(f"  {split_name}: {len(split_rows)} samples, {split_duration:.3f} hours")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
