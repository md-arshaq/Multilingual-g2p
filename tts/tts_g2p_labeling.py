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
G2P_MR_PATH = os.path.join(PROJECT_DIR, "data", "g2p_mr.txt")
G2P_GU_PATH = os.path.join(PROJECT_DIR, "data", "g2p_gu.txt")
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


def load_marathi_g2p_dict():
    """
    Load the Marathi G2P dictionary from both g2p_mr.txt and 
    multilingual_g2p_dataset.txt (with <MR> prefix).
    Returns dict: word (str) -> phoneme_sequence (str, space-separated)
    """
    g2p_dict = {}

    # Load from g2p_mr.txt (word\tphonemes)
    if os.path.exists(G2P_MR_PATH):
        with open(G2P_MR_PATH, "r", encoding="utf-8") as f:
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

    # Also load Marathi entries from multilingual dataset
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
                if left.startswith("<") and ">" in left:
                    end = left.index(">") + 1
                    lang_tag = left[1:end - 1].upper()
                    word = left[end:].strip()
                    if lang_tag == "MR" and word and phonemes:
                        if word not in g2p_dict:
                            g2p_dict[word] = phonemes

    return g2p_dict


def load_gujarati_g2p_dict():
    """
    Load the Gujarati G2P dictionary from both g2p_gu.txt and 
    multilingual_g2p_dataset.txt (with <GU> prefix).
    Returns dict: word (str) -> phoneme_sequence (str, space-separated)
    """
    g2p_dict = {}

    # Load from g2p_gu.txt (word\tphonemes)
    if os.path.exists(G2P_GU_PATH):
        with open(G2P_GU_PATH, "r", encoding="utf-8") as f:
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

    # Also load Gujarati entries from multilingual dataset
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
                if left.startswith("<") and ">" in left:
                    end = left.index(">") + 1
                    lang_tag = left[1:end - 1].upper()
                    word = left[end:].strip()
                    if lang_tag == "GU" and word and phonemes:
                        if word not in g2p_dict:
                            g2p_dict[word] = phonemes

    return g2p_dict


def load_g2p_dict(lang="hi"):
    """Dispatcher to load G2P dictionary for specified language."""
    if lang.lower() == "mr":
        return load_marathi_g2p_dict()
    elif lang.lower() == "gu":
        return load_gujarati_g2p_dict()
    return load_hindi_g2p_dict()


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


def normalize_text(text, lang="hi"):
    """
    Conservative text normalization for Indic scripts (Hindi, Marathi, Gujarati):
    - Remove punctuation (danda, double danda, comma, period, question mark, etc.)
    - Remove digits (both ASCII and Indic: 0-9, ०-९, ૦-૯)
    - Remove zero-width characters
    - Collapse extra whitespace
    - Preserve spoken words only
    """
    text = re.sub(r'[।॥,\.!?\-;:\'"()\[\]{}<>«»""''…–—/\\|@#$%^&*+=~`]', ' ', text)
    text = re.sub(r'[0-9०-९૦-૯]', ' ', text)
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_hindi_text(text):
    """Backward compatibility alias for normalize_text."""
    return normalize_text(text)


def devanagari_word_to_phonemes(word, valid_phonemes):
    """Fallback Devanagari character-level phonemization for OOV words."""
    vowel_map = {
        'अ': ['a'], 'आ': ['aa'], 'इ': ['i'], 'ई': ['ii'], 'उ': ['u'], 'ऊ': ['uu'], 'ऋ': ['rq'], 'ए': ['ee'], 'ऐ': ['ei'], 'ओ': ['o'], 'औ': ['ou'],
        'ॲ': ['ae'], 'ऑ': ['o']
    }
    consonant_map = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh', 'ञ': 'nj',
        'ट': 'tx', 'ठ': 'txh', 'ડ': 'dx', 'ढ': 'dxh', 'ण': 'nx',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'w',
        'श': 'sh', 'ष': 'sx', 'स': 's', 'ह': 'h', 'ड़': 'dxq', 'ढ़': 'dxhq', 'फ़': 'f', 'ज़': 'z', 'ख़': 'khq', 'ग़': 'gq', 'क़': 'kq',
        'ळ': 'lx'
    }
    matra_map = {
        'ा': ['aa'], 'િ': ['i'], 'ी': ['ii'], 'ु': ['u'], 'ू': ['uu'], 'ृ': ['rq'], 'े': ['ee'], 'ै': ['ei'], 'ो': ['o'], 'ौ': ['ou'],
        'ॅ': ['ae'], 'ॉ': ['o']
    }

    phonemes = []
    i = 0
    chars = list(word)
    while i < len(chars):
        c = chars[i]
        if c in vowel_map:
            phonemes.extend(vowel_map[c])
            i += 1
        elif c in consonant_map:
            base_p = consonant_map[c]
            # Check next character for matra or halant
            if i + 1 < len(chars):
                nxt = chars[i + 1]
                if nxt == '्':
                    phonemes.append(base_p)
                    i += 2
                elif nxt in matra_map:
                    phonemes.append(base_p)
                    phonemes.extend(matra_map[nxt])
                    i += 2
                elif nxt in ('ं', 'ँ'):
                    phonemes.extend([base_p, 'a', 'mq'])
                    i += 2
                else:
                    phonemes.extend([base_p, 'a'])
                    i += 1
            else:
                # Word-final consonant (conservatively include schwa unless specified)
                phonemes.extend([base_p, 'a'])
                i += 1
        elif c in matra_map:
            phonemes.extend(matra_map[c])
            i += 1
        elif c in ('ं', 'ँ'):
            phonemes.append('mq')
            i += 1
        elif c == 'ः':
            phonemes.append('hq')
            i += 1
        else:
            i += 1

    invalid = [p for p in phonemes if p not in valid_phonemes]
    if invalid or not phonemes:
        return None
    return phonemes


def gujarati_word_to_phonemes(word, valid_phonemes):
    """Fallback Gujarati character-level phonemization for OOV words."""
    vowel_map = {
        'અ': ['a'], 'આ': ['aa'], 'ઇ': ['i'], 'ઈ': ['ii'], 'ઉ': ['u'], 'ઊ': ['uu'],
        'ઋ': ['rq'], 'એ': ['ee'], 'ઐ': ['ei'], 'ઓ': ['o'], 'ઔ': ['ou'],
        'ઍ': ['ae'], 'ઑ': ['o']
    }
    consonant_map = {
        'ક': 'k', 'ખ': 'kh', 'ગ': 'g', 'ઘ': 'gh', 'ઙ': 'ng',
        'ચ': 'c', 'છ': 'ch', 'જ': 'j', 'ઝ': 'jh', 'ઞ': 'nj',
        'ટ': 'tx', 'ઠ': 'txh', 'ડ': 'dx', 'ઢ': 'dxh', 'ણ': 'nx',
        'ત': 't', 'થ': 'th', 'દ': 'd', 'ધ': 'dh', 'ન': 'n',
        'પ': 'p', 'ફ': 'ph', 'બ': 'b', 'ભ': 'bh', 'મ': 'm',
        'ય': 'y', 'ર': 'r', 'લ': 'l', 'ળ': 'lx', 'વ': 'w',
        'શ': 'sh', 'ષ': 'sx', 'સ': 's', 'હ': 'h'
    }
    matra_map = {
        'ા': ['aa'], 'િ': ['i'], 'ી': ['ii'], 'ુ': ['u'], 'ૂ': ['uu'],
        'ૃ': ['rq'], 'ે': ['ee'], 'ૈ': ['ei'], 'ો': ['o'], 'ૌ': ['ou'],
        'ૅ': ['ae'], 'ૉ': ['o']
    }

    phonemes = []
    i = 0
    chars = list(word)
    while i < len(chars):
        c = chars[i]
        if c in vowel_map:
            phonemes.extend(vowel_map[c])
            i += 1
        elif c in consonant_map:
            base_p = consonant_map[c]
            if i + 1 < len(chars):
                nxt = chars[i + 1]
                if nxt == '્':
                    phonemes.append(base_p)
                    i += 2
                elif nxt in matra_map:
                    phonemes.append(base_p)
                    phonemes.extend(matra_map[nxt])
                    i += 2
                elif nxt in ('ં', 'ઁ'):
                    phonemes.extend([base_p, 'a', 'mq'])
                    i += 2
                else:
                    phonemes.extend([base_p, 'a'])
                    i += 1
            else:
                phonemes.extend([base_p, 'a'])
                i += 1
        elif c in matra_map:
            phonemes.extend(matra_map[c])
            i += 1
        elif c in ('ં', 'ઁ'):
            phonemes.append('mq')
            i += 1
        elif c == 'ઃ':
            phonemes.append('hq')
            i += 1
        else:
            i += 1

    invalid = [p for p in phonemes if p not in valid_phonemes]
    if invalid or not phonemes:
        return None
    return phonemes


def text_to_baseline_tokens(text, g2p_dict, valid_phonemes, lang="hi"):
    """
    Convert text (Hindi, Marathi, or Gujarati) to baseline phoneme tokens using dictionary lookup with fallback.
    """
    normalized = normalize_text(text, lang=lang)
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
                fb_phonemes = gujarati_word_to_phonemes(word, valid_phonemes) if lang == "gu" else devanagari_word_to_phonemes(word, valid_phonemes)
                if fb_phonemes:
                    word_phonemes.append(fb_phonemes)
                else:
                    oov_words.append(f"{word}(invalid_phonemes:{','.join(invalid)})")
                    all_valid = False
            else:
                word_phonemes.append(phonemes)
        else:
            fb_phonemes = gujarati_word_to_phonemes(word, valid_phonemes) if lang == "gu" else devanagari_word_to_phonemes(word, valid_phonemes)
            if fb_phonemes:
                word_phonemes.append(fb_phonemes)
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
        "--lang", type=str, default="hi", choices=["hi", "mr", "gu"],
        help="Language code ('hi', 'mr', or 'gu', default: 'hi')"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to manifest CSV from Phase 1 (defaults to data/tts_{language}_female/manifest.csv)"
    )
    args = parser.parse_args()

    lang_dir_names = {"hi": "tts_hindi_female", "mr": "tts_marathi_female", "gu": "tts_gujarati_female"}
    lang_dir_name = lang_dir_names.get(args.lang, f"tts_{args.lang}_female")
    if args.manifest is None:
        args.manifest = os.path.join(PROJECT_DIR, "data", lang_dir_name, "manifest.csv")

    print("=" * 60)
    print(f"PHASE 2: G2P LABEL PREPARATION ({args.lang.upper()})")
    print("=" * 60)

    # Load resources
    print(f"\nLoading G2P dictionary for {args.lang.upper()}...")
    g2p_dict = load_g2p_dict(args.lang)
    print(f"  Loaded {len(g2p_dict)} {args.lang.upper()} word→phoneme entries")

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
            text, g2p_dict, valid_phonemes, lang=args.lang
        )

        word_count = len(normalize_text(text, lang=args.lang).split())
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
