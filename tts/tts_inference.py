#!/usr/bin/env python3
"""
Phase 7: Inference — generate paired baseline/clustered audio for evaluation.

Produces two evaluation sets:
  1. Held-out test set: the reserved 5% from the split
  2. Unseen generalization set: 20 new Hindi sentences

For each sentence, both baseline and clustered synthesis are produced.

Usage:
    python tts/tts_inference.py \
        --baseline_model models/tts_hindi_female/baseline/best_model.pth \
        --clustered_model models/tts_hindi_female/clustered/best_model.pth \
        [--output_dir samples/tts_hindi_female]
"""

import os
import sys
import csv
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_MANIFEST = os.path.join(PROJECT_DIR, "data", "tts_hindi_female", "manifest.csv")
DEFAULT_UNSEEN = os.path.join(PROJECT_DIR, "configs", "tts", "tts_unseen_sentences.json")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "samples", "tts_hindi_female")
G2P_HI_PATH = os.path.join(PROJECT_DIR, "data", "g2p_hi.txt")
G2P_MULTI_PATH = os.path.join(PROJECT_DIR, "data", "multilingual_g2p_dataset.txt")
CLUSTER_MAPPING_PATH = os.path.join(PROJECT_DIR, "g2p", "phoneme_cluster_mapping.json")

# Import labeling functions
sys.path.insert(0, SCRIPT_DIR)
from tts_g2p_labeling import (
    load_hindi_g2p_dict, load_cluster_mapping, load_phoneme_vocab,
    normalize_hindi_text, text_to_baseline_tokens, baseline_to_clustered_tokens
)
from vits_tokenizer import (
    BASELINE_PHONEMES, CLUSTER_TOKENS, assert_tokenizer_round_trip,
    build_vocab, patch_tokenizer,
)

WB_TOKEN = "<wb>"


def synthesize_vits(model, text_tokens, output_path):
    """
    Synthesize audio from token sequence using a loaded VITS model.

    Args:
        model: loaded TTS model object
        text_tokens: space-separated token string
        output_path: path to save WAV output

    Returns:
        success (bool)
    """
    try:
        # Coqui TTS inference — text is our pre-tokenized sequence
        model.tts_to_file(
            text=text_tokens,
            file_path=output_path,
        )
        return True
    except Exception as e:
        print(f"    ERROR: Synthesis failed: {e}")
        return False


def load_patched_tts_model(model_path, config_path, vocab, smoke_sequence):
    """Load a VITS checkpoint and restore its strict whitespace tokenizer.

    The tokenizer patch used in training is runtime-only; Coqui's generic
    loader cannot infer multi-character token boundaries from config.json.
    Applying the same shared patch here prevents character-level inference.
    """
    from TTS.api import TTS

    model = TTS(model_path=model_path, config_path=config_path)
    try:
        tokenizer = model.synthesizer.tts_model.tokenizer
    except AttributeError as exc:
        raise RuntimeError("Loaded model does not expose a Coqui VITS tokenizer") from exc

    patch_tokenizer(tokenizer, vocab)
    ids = assert_tokenizer_round_trip(tokenizer, smoke_sequence)
    print(f"  Tokenizer restored: {smoke_sequence!r} -> {ids}")
    return model


def generate_held_out(manifest_path, baseline_model, clustered_model, output_dir):
    """Generate paired synthesis for held-out test samples."""
    print("\n--- Held-out test set ---")

    held_out_dir = os.path.join(output_dir, "held_out")
    os.makedirs(held_out_dir, exist_ok=True)

    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    test_samples = [r for r in manifest
                    if r.get("split") == "test" and r["selected"] == "1"]
    print(f"  Test samples: {len(test_samples)}")

    metadata = []
    for i, row in enumerate(test_samples):
        sample_id = row["row_id"]
        text = row["text"]
        baseline_tokens = row["baseline_tokens"]
        clustered_tokens = row["clustered_tokens"]

        # Baseline synthesis
        baseline_path = os.path.join(held_out_dir, f"baseline_{i+1:03d}.wav")
        baseline_ok = False
        if baseline_model is not None:
            baseline_ok = synthesize_vits(baseline_model, baseline_tokens, baseline_path)
        else:
            print(f"    [SKIP] No baseline model loaded")

        # Clustered synthesis
        clustered_path = os.path.join(held_out_dir, f"clustered_{i+1:03d}.wav")
        clustered_ok = False
        if clustered_model is not None:
            clustered_ok = synthesize_vits(clustered_model, clustered_tokens, clustered_path)
        else:
            print(f"    [SKIP] No clustered model loaded")

        status = "OK" if (baseline_ok and clustered_ok) else "PARTIAL"
        print(f"  [{i+1}/{len(test_samples)}] {status}: {text[:40]}...")

        metadata.append({
            "set": "held_out",
            "sample_id": sample_id,
            "index": i + 1,
            "text": text,
            "baseline_tokens": baseline_tokens,
            "clustered_tokens": clustered_tokens,
            "baseline_wav": os.path.basename(baseline_path) if baseline_ok else "",
            "clustered_wav": os.path.basename(clustered_path) if clustered_ok else "",
        })

    return metadata


def generate_unseen(unseen_path, baseline_model, clustered_model, output_dir,
                    g2p_dict, cluster_map, valid_phonemes):
    """Generate paired synthesis for unseen generalization sentences."""
    print("\n--- Unseen generalization set ---")

    unseen_dir = os.path.join(output_dir, "unseen")
    os.makedirs(unseen_dir, exist_ok=True)

    with open(unseen_path, "r", encoding="utf-8") as f:
        unseen_data = json.load(f)

    sentences = unseen_data["sentences"]
    print(f"  Unseen sentences: {len(sentences)}")

    metadata = []
    for s in sentences:
        idx = s["id"]
        text = s["text"]

        # G2P pipeline: text -> baseline tokens
        baseline_tokens, oov, baseline_ok = text_to_baseline_tokens(
            text, g2p_dict, valid_phonemes
        )

        if not baseline_ok:
            print(f"  [{idx:02d}] SKIP (OOV: {oov}): {text[:40]}...")
            metadata.append({
                "set": "unseen",
                "sample_id": idx,
                "index": idx,
                "text": text,
                "baseline_tokens": "",
                "clustered_tokens": "",
                "baseline_wav": "",
                "clustered_wav": "",
                "error": f"OOV: {', '.join(oov[:5])}",
            })
            continue

        # Derive clustered tokens
        clustered_tokens, unmapped, cluster_ok = baseline_to_clustered_tokens(
            baseline_tokens, cluster_map
        )

        if not cluster_ok:
            print(f"  [{idx:02d}] SKIP (unmapped: {unmapped}): {text[:40]}...")
            metadata.append({
                "set": "unseen",
                "sample_id": idx,
                "index": idx,
                "text": text,
                "baseline_tokens": baseline_tokens,
                "clustered_tokens": "",
                "baseline_wav": "",
                "clustered_wav": "",
                "error": f"Unmapped: {', '.join(unmapped)}",
            })
            continue

        # Log G2P output (required by prompt)
        print(f"  [{idx:02d}] G2P output:")
        print(f"       Text:     {text}")
        print(f"       Baseline: {baseline_tokens[:60]}...")
        print(f"       Clustered: {clustered_tokens[:60]}...")

        # Synthesize
        baseline_path = os.path.join(unseen_dir, f"baseline_{idx:02d}.wav")
        clustered_path = os.path.join(unseen_dir, f"clustered_{idx:02d}.wav")

        baseline_ok = False
        if baseline_model is not None:
            baseline_ok = synthesize_vits(baseline_model, baseline_tokens, baseline_path)

        clustered_ok = False
        if clustered_model is not None:
            clustered_ok = synthesize_vits(clustered_model, clustered_tokens, clustered_path)

        metadata.append({
            "set": "unseen",
            "sample_id": idx,
            "index": idx,
            "text": text,
            "baseline_tokens": baseline_tokens,
            "clustered_tokens": clustered_tokens,
            "baseline_wav": os.path.basename(baseline_path) if baseline_ok else "",
            "clustered_wav": os.path.basename(clustered_path) if clustered_ok else "",
            "error": "",
        })

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Inference for evaluation"
    )
    parser.add_argument(
        "--baseline_model", type=str, default=None,
        help="Path to trained baseline VITS model checkpoint"
    )
    parser.add_argument(
        "--clustered_model", type=str, default=None,
        help="Path to trained clustered VITS model checkpoint"
    )
    parser.add_argument(
        "--baseline_config", type=str, default=None,
        help="Path to baseline VITS config"
    )
    parser.add_argument(
        "--clustered_config", type=str, default=None,
        help="Path to clustered VITS config"
    )
    parser.add_argument(
        "--manifest", type=str, default=DEFAULT_MANIFEST,
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--unseen", type=str, default=DEFAULT_UNSEEN,
        help="Path to unseen sentences JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT,
        help="Output directory for synthesized audio"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run G2P pipeline only, don't synthesize"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PHASE 7: INFERENCE & EVALUATION DATA GENERATION")
    print("=" * 60)

    # Load G2P resources
    print("\nLoading G2P resources...")
    g2p_dict = load_hindi_g2p_dict()
    cluster_map = load_cluster_mapping()
    valid_phonemes = load_phoneme_vocab()
    print(f"  G2P dictionary: {len(g2p_dict)} words")
    print(f"  Cluster mapping: {len(cluster_map)} phonemes")

    # Load TTS models
    baseline_model = None
    clustered_model = None

    if not args.dry_run:
        if args.baseline_model:
            try:
                print(f"\nLoading baseline model: {args.baseline_model}")
                baseline_model = load_patched_tts_model(
                    args.baseline_model,
                    args.baseline_config,
                    build_vocab(BASELINE_PHONEMES),
                    "b aa r ax t",
                )
                print("  Baseline model loaded")
            except Exception as e:
                print(f"  WARNING: Could not load baseline model: {e}")

        if args.clustered_model:
            try:
                print(f"\nLoading clustered model: {args.clustered_model}")
                clustered_model = load_patched_tts_model(
                    args.clustered_model,
                    args.clustered_config,
                    build_vocab(CLUSTER_TOKENS),
                    "C0 C10 C38",
                )
                print("  Clustered model loaded")
            except Exception as e:
                print(f"  WARNING: Could not load clustered model: {e}")

        if baseline_model is None and clustered_model is None:
            print("\n  WARNING: No models loaded. Running in dry-run mode.")
            args.dry_run = True

    # Generate held-out test set
    held_out_meta = generate_held_out(
        args.manifest, baseline_model, clustered_model, args.output_dir
    )

    # Generate unseen set
    unseen_meta = generate_unseen(
        args.unseen, baseline_model, clustered_model, args.output_dir,
        g2p_dict, cluster_map, valid_phonemes
    )

    # Save inference metadata
    all_meta = held_out_meta + unseen_meta
    meta_path = os.path.join(args.output_dir, "inference_metadata.csv")
    if all_meta:
        fieldnames = list(all_meta[0].keys())
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_meta)
        print(f"\n  Inference metadata saved: {meta_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("PHASE 7 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Held-out samples: {len(held_out_meta)}")
    print(f"  Unseen sentences: {len(unseen_meta)}")

    unseen_ok = sum(1 for m in unseen_meta if m.get("baseline_tokens"))
    unseen_skip = sum(1 for m in unseen_meta if not m.get("baseline_tokens"))
    print(f"  Unseen tokenizable: {unseen_ok}/{len(unseen_meta)}")
    if unseen_skip:
        print(f"  Unseen skipped (OOV): {unseen_skip}")

    if args.dry_run:
        print("\n  DRY RUN: No audio files generated. Use --baseline_model and")
        print("  --clustered_model to synthesize audio.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
