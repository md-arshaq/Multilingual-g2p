#!/usr/bin/env python3
"""
Phase 2: G2P labeling for the selected Hindi TTS subset.

For each selected sample in the manifest:
  1. Normalize text (strip punctuation, preserve spoken words)
  2. Look up each word in the Hindi G2P dictionary
  3. Derive clustered tokens via phoneme_cluster_mapping.json
  4. Exclude utterances with OOV words or unmappable phonemes
  5. Update the manifest with token sequences

Usage:
    python tts/tts_g2p_labeling.py [--manifest data/tts_hindi_female/manifest.csv]
"""

import os
import sys
import csv
import json
import re
import argparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

G2P_HI_PATH = os.path.join(PROJECT_DIR, "data", "g2p_hi.txt")
G2P_MULTI_PATH = os.path.join(PROJECT_DIR, "data", "multilingual_g2p_dataset.txt")
CLUSTER_MAPPING_PATH = os.path.join(PROJECT_DIR, "g2p", "phoneme_cluster_mapping.json")
PHONEME_VOCAB_PATH = os.path.join(PROJECT_DIR, "g2p", "phoneme_vocab.json")
DEFAULT_MANIFEST = os.path.join(PROJECT_DIR, "data", "tts_hindi_female", "manifest.csv")

# Word boundary token
WB_TOKEN = "<wb>"


def load_hindi_g2p_dict():
    """
    Load the Hindi G2P dictionary from both g2p_hi.txt and 
    multilingual_g2p_dataset.txt (with <HI> prefix).
    Returns dict: word (str) -> phoneme_sequence (str, space-separated)
    """
    g2p_dict = {}

    # Load from g2p_hi.txt (word\tphonemes)
    if os.path.exists(G2P_HI_PATH):
        with open(G2P_HI_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    word = parts[0].strip()
                    phonemes = parts[1].strip()
                    if word and phonemes:
                        g2p_dict[word] = phonemes

    # Also load Hindi entries from multilingual dataset
    if os.path.exists(G2P_MULTI_PATH):
        with open(G2P_MULTI_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                left = parts[0].strip()
                phonemes = parts[1].strip()
                # Extract language tag and word
                if left.startswith("<") and ">" in left:
                    end = left.index(">") + 1
                    lang_tag = left[1:end - 1].upper()
                    word = left[end:].strip()
                    if lang_tag == "HI" and word and phonemes:
                        # Don't overwrite existing entries
                        if word not in g2p_dict:
                            g2p_dict[word] = phonemes

    return g2p_dict


def load_cluster_mapping():
    """
    Load phoneme -> cluster token mapping.
    Returns dict: phoneme (str) -> cluster_token (str, e.g. 'C0')
    """
    with open(CLUSTER_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {p: f"C{info['cluster_id']}" for p, info in mapping.items()}


def load_phoneme_vocab():
    """Load the 57-phoneme vocabulary to validate phoneme tokens."""
    with open(PHONEME_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return set(vocab["phoneme_to_id"].keys())


def normalize_hindi_text(text):
    """
    Conservative text normalization for Hindi:
    - Remove punctuation (danda, double danda, comma, period, question mark, etc.)
    - Remove extra whitespace
    - Preserve spoken Hindi words only
    """
    # Remove common Hindi/Unicode punctuation
    # Danda (।), double danda (॥), other punctuation
    text = re.sub(r'[।॥,\.!?\-;:\'"()\[\]{}<>«»""''…–—/\\|@#$%^&*+=~`]', ' ', text)

    # Remove digits (both Devanagari and ASCII)
    text = re.sub(r'[0-9०-९]', ' ', text)

    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_to_baseline_tokens(text, g2p_dict, valid_phonemes):
    """
    Convert Hindi text to baseline phoneme tokens using dictionary lookup.

    Returns:
        (token_sequence, oov_words, all_valid)
        - token_sequence: str, space-separated tokens with <wb> boundaries
        - oov_words: list of words not found in dictionary
        - all_valid: True if all words found and all phonemes valid
    """
    normalized = normalize_hindi_text(text)
    words = normalized.split()

    if not words:
        return "", [], False

    word_phonemes = []
    oov_words = []
    all_valid = True

    for word in words:
        if word in g2p_dict:
            phonemes_str = g2p_dict[word]
            phonemes = phonemes_str.split()

            # Validate all phonemes are in our vocabulary
            invalid = [p for p in phonemes if p not in valid_phonemes]
            if invalid:
                oov_words.append(f"{word}(invalid_phonemes:{','.join(invalid)})")
                all_valid = False
                continue

            word_phonemes.append(phonemes)
        else:
            oov_words.append(word)
            all_valid = False

    if not all_valid or not word_phonemes:
        return "", oov_words, False

    # Build token sequence with word boundaries
    token_parts = []
    for i, phonemes in enumerate(word_phonemes):
        token_parts.extend(phonemes)
        if i < len(word_phonemes) - 1:
            token_parts.append(WB_TOKEN)

    return " ".join(token_parts), oov_words, True


def baseline_to_clustered_tokens(baseline_tokens, cluster_map):
    """
    Convert baseline phoneme token sequence to clustered token sequence.
    Preserves <wb> boundary tokens.

    Returns:
        (clustered_sequence, unmapped_phonemes, all_valid)
    """
    if not baseline_tokens:
        return "", [], False

    tokens = baseline_tokens.split()
    clustered = []
    unmapped = []

    for t in tokens:
        if t == WB_TOKEN:
            clustered.append(WB_TOKEN)
        elif t in cluster_map:
            clustered.append(cluster_map[t])
        else:
            unmapped.append(t)

    all_valid = len(unmapped) == 0
    return " ".join(clustered), unmapped, all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: G2P label preparation for TTS experiment"
    )
    parser.add_argument(
        "--manifest", type=str, default=DEFAULT_MANIFEST,
        help="Path to manifest CSV from Phase 1"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2: G2P LABEL PREPARATION")
    print("=" * 60)

    # Load resources
    print("\nLoading G2P dictionary...")
    g2p_dict = load_hindi_g2p_dict()
    print(f"  Loaded {len(g2p_dict)} Hindi word→phoneme entries")

    print("Loading cluster mapping...")
    cluster_map = load_cluster_mapping()
    unique_clusters = sorted(set(cluster_map.values()), key=lambda x: int(x[1:]))
    print(f"  {len(cluster_map)} phonemes → {len(unique_clusters)} clusters")
    print(f"  Clusters: {', '.join(unique_clusters)}")

    print("Loading phoneme vocabulary...")
    valid_phonemes = load_phoneme_vocab()
    print(f"  {len(valid_phonemes)} valid phonemes")

    # Load manifest
    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    selected = [r for r in manifest if r["selected"] == "1"]
    print(f"  Total rows: {len(manifest)}")
    print(f"  Selected for labeling: {len(selected)}")

    # Process each selected sample
    print("\n--- Processing G2P labels ---")
    stats = {
        "total_selected": len(selected),
        "labeled_ok": 0,
        "excluded_oov": 0,
        "excluded_cluster_fail": 0,
        "excluded_empty": 0,
        "total_words": 0,
        "oov_words": 0,
        "dict_hits": 0,
    }

    all_oov = []

    for i, row in enumerate(manifest):
        if row["selected"] != "1":
            continue

        text = row["text"]

        # Get baseline tokens
        baseline_seq, oov_words, baseline_ok = text_to_baseline_tokens(
            text, g2p_dict, valid_phonemes
        )

        word_count = len(normalize_hindi_text(text).split())
        stats["total_words"] += word_count

        if not baseline_ok:
            row["selected"] = "0"
            row["exclusion_reason"] = f"oov_words:{','.join(oov_words[:5])}"
            stats["excluded_oov"] += 1
            stats["oov_words"] += len(oov_words)
            stats["dict_hits"] += word_count - len(oov_words)
            all_oov.extend(oov_words)
            continue

        stats["dict_hits"] += word_count

        # Get clustered tokens
        clustered_seq, unmapped, cluster_ok = baseline_to_clustered_tokens(
            baseline_seq, cluster_map
        )

        if not cluster_ok:
            row["selected"] = "0"
            row["exclusion_reason"] = f"unmapped_phonemes:{','.join(unmapped)}"
            stats["excluded_cluster_fail"] += 1
            continue

        if not baseline_seq.strip() or not clustered_seq.strip():
            row["selected"] = "0"
            row["exclusion_reason"] = "empty_token_sequence"
            stats["excluded_empty"] += 1
            continue

        # Store tokens
        row["baseline_tokens"] = baseline_seq
        row["clustered_tokens"] = clustered_seq
        stats["labeled_ok"] += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(manifest)} "
                  f"(labeled: {stats['labeled_ok']}, "
                  f"excluded: {stats['excluded_oov'] + stats['excluded_cluster_fail']})")

    # Recompute duration of final selected set
    final_selected = [r for r in manifest if r["selected"] == "1"]
    final_duration = sum(float(r["duration_sec"]) for r in final_selected) / 3600

    # Save updated manifest
    print(f"\n--- Saving updated manifest ---")
    fieldnames = list(manifest[0].keys())
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"  Updated: {args.manifest}")

    # Save OOV word list for reference
    if all_oov:
        oov_path = os.path.join(os.path.dirname(args.manifest), "oov_words.txt")
        with open(oov_path, "w", encoding="utf-8") as f:
            for w in sorted(set(all_oov)):
                f.write(w + "\n")
        print(f"  OOV words saved to: {oov_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total selected (input): {stats['total_selected']}")
    print(f"  Successfully labeled:   {stats['labeled_ok']}")
    print(f"  Excluded (OOV):         {stats['excluded_oov']}")
    print(f"  Excluded (cluster):     {stats['excluded_cluster_fail']}")
    print(f"  Excluded (empty):       {stats['excluded_empty']}")
    print(f"  Final duration:         {final_duration:.4f} hours")
    print(f"  Dictionary coverage:    {stats['dict_hits']}/{stats['total_words']} words "
          f"({100 * stats['dict_hits'] / max(stats['total_words'], 1):.1f}%)")
    print(f"  Unique OOV words:       {len(set(all_oov))}")

    if final_duration < 1.70:
        print(f"\n  ⚠️  WARNING: Final duration {final_duration:.4f}h is below 1.70h minimum")
        print(f"  The experiment may need a larger initial selection or OOV handling")

    return 0


if __name__ == "__main__":
    sys.exit(main())
