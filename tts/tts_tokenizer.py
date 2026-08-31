#!/usr/bin/env python3
"""
Phase 5: Custom tokenizer for VITS that treats phoneme/cluster tokens atomically.

Critical requirements:
  - Splits on whitespace to get tokens
  - Handles C10 as ONE token (not C + 1 + 0)
  - Handles <wb> as a boundary token
  - Provides encode/decode and vocabulary export

Usage:
    python tts/tts_tokenizer.py --test                # Run smoke tests
    python tts/tts_tokenizer.py --export baseline      # Export baseline vocab
    python tts/tts_tokenizer.py --export clustered     # Export clustered vocab
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PHONEME_VOCAB_PATH = os.path.join(PROJECT_DIR, "g2p", "phoneme_vocab.json")
CLUSTER_MAPPING_PATH = os.path.join(PROJECT_DIR, "g2p", "phoneme_cluster_mapping.json")

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
BLANK_TOKEN = "<blnk>"
WB_TOKEN = "<wb>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, BLANK_TOKEN, WB_TOKEN]


class TTSTokenizer:
    """
    Whitespace-based tokenizer for VITS that treats each space-separated
    token as an atomic unit.

    This bypasses any character-level or IPA-based phonemizer.
    Token examples:
      Baseline: 'b aa r ax t'  → ['b', 'aa', 'r', 'ax', 't']
      Clustered: 'C3 C0 C29 C0 C1' → ['C3', 'C0', 'C29', 'C0', 'C1']
    """

    def __init__(self, tokens, name="tts_tokenizer"):
        """
        Args:
            tokens: list of content tokens (phonemes or cluster IDs)
            name: identifier for this tokenizer
        """
        self.name = name

        # Build vocabulary: special tokens first, then content tokens
        self.token_to_id = {}
        self.id_to_token = {}

        idx = 0
        for t in SPECIAL_TOKENS:
            self.token_to_id[t] = idx
            self.id_to_token[idx] = t
            idx += 1

        for t in tokens:
            if t not in self.token_to_id:
                self.token_to_id[t] = idx
                self.id_to_token[idx] = t
                idx += 1

        self.vocab_size = len(self.token_to_id)
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.blank_id = self.token_to_id[BLANK_TOKEN]
        self.wb_id = self.token_to_id[WB_TOKEN]

    def encode(self, text, add_bos=True, add_eos=True):
        """
        Encode a whitespace-separated token string to a list of integer IDs.

        Args:
            text: space-separated tokens, e.g. 'b aa r <wb> ax t'
            add_bos: prepend BOS token
            add_eos: append EOS token

        Returns:
            list of integer IDs
        """
        tokens = text.strip().split()
        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            else:
                raise ValueError(
                    f"Unknown token '{t}' not in vocabulary. "
                    f"Available tokens: {sorted(self.token_to_id.keys())}"
                )

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, skip_special=True):
        """
        Decode a list of integer IDs back to token string.

        Args:
            ids: list of integer IDs
            skip_special: if True, skip pad/bos/eos/blank tokens

        Returns:
            space-separated token string
        """
        special = {self.pad_id, self.bos_id, self.eos_id, self.blank_id}
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                t = self.id_to_token[id_]
                if skip_special and id_ in special:
                    continue
                tokens.append(t)
        return " ".join(tokens)

    def get_vocab_dict(self):
        """Return the full token→ID mapping."""
        return dict(self.token_to_id)

    def get_content_tokens(self):
        """Return only content tokens (excluding special tokens)."""
        return [t for t in self.token_to_id if t not in SPECIAL_TOKENS]

    def export_json(self, path):
        """Export tokenizer as JSON file."""
        data = {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "special_tokens": {t: self.token_to_id[t] for t in SPECIAL_TOKENS},
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def get_characters_config(self):
        """
        Generate the Coqui TTS characters configuration dict.
        This is used in VITS config to define the custom vocabulary.
        """
        content_tokens = self.get_content_tokens()

        # For Coqui TTS, we need to define characters as a string
        # But since our tokens are multi-character, we need a different approach.
        # We'll use the Coqui TTS 'characters' dict format.
        return {
            "characters_class": "TTS.tts.utils.text.characters.Graphemes",
            "vocab_string": " ".join(content_tokens),
            "pad": PAD_TOKEN,
            "eos": EOS_TOKEN,
            "bos": BOS_TOKEN,
            "blank": BLANK_TOKEN,
            "characters": " ".join(content_tokens + [WB_TOKEN]),
            "punctuations": "",
            "phonemes": None,
            "is_unique": True,
            "is_sorted": True,
        }


def build_baseline_tokenizer():
    """Build tokenizer for the 57-phoneme baseline vocabulary."""
    with open(PHONEME_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    phonemes = sorted(vocab["phoneme_to_id"].keys())
    return TTSTokenizer(phonemes, name="baseline_57phoneme")


def build_clustered_tokenizer():
    """Build tokenizer for the 39-cluster vocabulary (C0–C38)."""
    with open(CLUSTER_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Extract unique cluster IDs
    cluster_ids = set()
    for info in mapping.values():
        cluster_ids.add(f"C{info['cluster_id']}")

    # Sort numerically
    clusters = sorted(cluster_ids, key=lambda x: int(x[1:]))
    return TTSTokenizer(clusters, name="clustered_39cluster")


def run_smoke_tests():
    """
    Run critical smoke tests as specified in the prompt.
    These verify correct tokenization behavior.
    """
    print("=" * 60)
    print("TOKENIZER SMOKE TESTS")
    print("=" * 60)

    all_passed = True

    # Test 1: Baseline tokenizer
    print("\n--- Test 1: Baseline tokenizer ---")
    baseline_tok = build_baseline_tokenizer()
    print(f"  Vocab size: {baseline_tok.vocab_size} "
          f"(57 phonemes + {len(SPECIAL_TOKENS)} special = {57 + len(SPECIAL_TOKENS)})")

    test_seq = "b aa r ax t"
    ids = baseline_tok.encode(test_seq, add_bos=False, add_eos=False)
    decoded = baseline_tok.decode(ids)
    print(f"  '{test_seq}' → {ids} → '{decoded}'")

    if len(ids) != 5:
        print(f"  ❌ FAIL: Expected 5 IDs, got {len(ids)}")
        all_passed = False
    else:
        print(f"  ✅ PASS: 5 tokens → 5 IDs")

    if decoded != test_seq:
        print(f"  ❌ FAIL: Round-trip failed: '{decoded}' != '{test_seq}'")
        all_passed = False
    else:
        print(f"  ✅ PASS: Round-trip encode/decode")

    # Test 2: Clustered tokenizer — CRITICAL: C10 must be ONE token
    print("\n--- Test 2: Clustered tokenizer (C10 atomic test) ---")
    clustered_tok = build_clustered_tokenizer()
    print(f"  Vocab size: {clustered_tok.vocab_size} "
          f"(39 clusters + {len(SPECIAL_TOKENS)} special = {39 + len(SPECIAL_TOKENS)})")

    test_seq = "C0 C10 C38"
    ids = clustered_tok.encode(test_seq, add_bos=False, add_eos=False)
    decoded = clustered_tok.decode(ids)
    print(f"  '{test_seq}' → {ids} → '{decoded}'")

    if len(ids) != 3:
        print(f"  ❌ FAIL: Expected 3 IDs, got {len(ids)}")
        print(f"    C10 was likely split into C + 1 + 0!")
        all_passed = False
    else:
        print(f"  ✅ PASS: 'C0 C10 C38' → exactly 3 IDs (C10 is atomic)")

    if decoded != test_seq:
        print(f"  ❌ FAIL: Round-trip failed: '{decoded}' != '{test_seq}'")
        all_passed = False
    else:
        print(f"  ✅ PASS: Round-trip encode/decode")

    # Test 3: Word boundary token
    print("\n--- Test 3: Word boundary token ---")
    test_seq_wb = "b aa r <wb> ax t"
    ids_wb = baseline_tok.encode(test_seq_wb, add_bos=False, add_eos=False)
    decoded_wb = baseline_tok.decode(ids_wb)
    print(f"  '{test_seq_wb}' → {ids_wb}")

    if len(ids_wb) != 6:
        print(f"  ❌ FAIL: Expected 6 IDs (5 phonemes + 1 <wb>), got {len(ids_wb)}")
        all_passed = False
    else:
        print(f"  ✅ PASS: <wb> handled as separate token")

    # Test 4: BOS/EOS
    print("\n--- Test 4: BOS/EOS wrapping ---")
    ids_wrapped = baseline_tok.encode("b aa", add_bos=True, add_eos=True)
    print(f"  'b aa' with BOS/EOS → {ids_wrapped}")

    if ids_wrapped[0] != baseline_tok.bos_id:
        print(f"  ❌ FAIL: First ID should be BOS ({baseline_tok.bos_id})")
        all_passed = False
    elif ids_wrapped[-1] != baseline_tok.eos_id:
        print(f"  ❌ FAIL: Last ID should be EOS ({baseline_tok.eos_id})")
        all_passed = False
    else:
        print(f"  ✅ PASS: BOS/EOS correctly applied")

    # Test 5: All 39 clusters present
    print("\n--- Test 5: Cluster vocabulary completeness ---")
    content = clustered_tok.get_content_tokens()
    expected_clusters = {f"C{i}" for i in range(39)}
    actual_clusters = set(content)
    missing = expected_clusters - actual_clusters
    extra = actual_clusters - expected_clusters - {WB_TOKEN}

    if missing:
        print(f"  ❌ FAIL: Missing clusters: {sorted(missing)}")
        all_passed = False
    elif extra:
        print(f"  ⚠️  WARNING: Extra tokens: {sorted(extra)}")
    else:
        print(f"  ✅ PASS: All C0–C38 present ({len(expected_clusters)} clusters)")

    # Test 6: All 57 phonemes present
    print("\n--- Test 6: Baseline vocabulary completeness ---")
    baseline_content = baseline_tok.get_content_tokens()
    with open(PHONEME_VOCAB_PATH, "r", encoding="utf-8") as f:
        expected_phonemes = set(json.load(f)["phoneme_to_id"].keys())
    actual_phonemes = set(baseline_content)
    missing_p = expected_phonemes - actual_phonemes
    if missing_p:
        print(f"  ❌ FAIL: Missing phonemes: {sorted(missing_p)}")
        all_passed = False
    else:
        print(f"  ✅ PASS: All 57 phonemes present")

    # Test 7: No internal phonemizer interference (simulated)
    print("\n--- Test 7: No phonemizer interference ---")
    # Verify that encoding + decoding preserves exact token sequence
    cluster_input = "C3 C0 C29 <wb> C0 C1"
    cluster_ids = clustered_tok.encode(cluster_input, add_bos=False, add_eos=False)
    cluster_output = clustered_tok.decode(cluster_ids)
    if cluster_output != cluster_input:
        print(f"  ❌ FAIL: Token sequence altered: '{cluster_output}' != '{cluster_input}'")
        all_passed = False
    else:
        print(f"  ✅ PASS: No alteration of cluster token sequence")

    # Summary
    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL SMOKE TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print(f"{'=' * 60}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Custom TTS tokenizer with smoke tests"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run smoke tests"
    )
    parser.add_argument(
        "--export", type=str, choices=["baseline", "clustered", "both"],
        help="Export tokenizer JSON (baseline, clustered, or both)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(PROJECT_DIR, "configs", "tts"),
        help="Output directory for exported files"
    )
    args = parser.parse_args()

    if args.test:
        ok = run_smoke_tests()
        sys.exit(0 if ok else 1)

    if args.export:
        os.makedirs(args.output_dir, exist_ok=True)

        if args.export in ["baseline", "both"]:
            tok = build_baseline_tokenizer()
            path = tok.export_json(
                os.path.join(args.output_dir, "tokenizer_baseline.json")
            )
            print(f"Baseline tokenizer exported: {path}")
            print(f"  Vocab size: {tok.vocab_size}")
            print(f"  Content tokens: {len(tok.get_content_tokens())}")

        if args.export in ["clustered", "both"]:
            tok = build_clustered_tokenizer()
            path = tok.export_json(
                os.path.join(args.output_dir, "tokenizer_clustered.json")
            )
            print(f"Clustered tokenizer exported: {path}")
            print(f"  Vocab size: {tok.vocab_size}")
            print(f"  Content tokens: {len(tok.get_content_tokens())}")

        return 0

    # Default: show info
    print("Baseline tokenizer:")
    baseline = build_baseline_tokenizer()
    print(f"  Vocab size: {baseline.vocab_size}")
    print(f"  Content tokens: {len(baseline.get_content_tokens())}")
    print(f"  Tokens: {baseline.get_content_tokens()}")

    print("\nClustered tokenizer:")
    clustered = build_clustered_tokenizer()
    print(f"  Vocab size: {clustered.vocab_size}")
    print(f"  Content tokens: {len(clustered.get_content_tokens())}")
    print(f"  Tokens: {clustered.get_content_tokens()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
