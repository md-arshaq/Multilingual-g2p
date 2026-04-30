# -*- coding: utf-8 -*-
"""
Week 3 — Task 1: Map G2P Output Phonemes → Parler-TTS Phoneme Set
==================================================================
Your G2P model outputs a custom phoneme set (e.g. a, aa, ii, uu, mq, txh…).
Parler-TTS uses IPA (International Phonetic Alphabet).

This script:
  1. Defines the full mapping from YOUR phoneme set → IPA
  2. Shows what Parler-TTS actually expects (text input, not raw IPA)
  3. Verifies every phoneme in your dataset is accounted for
  4. Prints a conversion table and any unmapped symbols

Run:
  python week3_task1_phoneme_mapping.py
"""

import json
import os
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PARLER-TTS EXPECTED FORMAT — IMPORTANT CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
"""
Parler-TTS (mini v1) does NOT accept raw IPA as input.
It accepts PLAIN TEXT, just like a normal TTS system.

Your pipeline will be:
  [Word] → [G2P model] → [your phonemes] → [IPA] → [display/debug only]
                                                   ↓
  [Word as plain text] ──────────────────────────→ [Parler-TTS]

So the G2P output is NOT fed into Parler-TTS directly.
Instead, Parler-TTS handles its OWN internal pronunciation.
Your G2P output serves a different purpose:
  - Driving a rule-based TTS (e.g. Festival, eSpeak)
  - Verifying/correcting Parler-TTS pronunciation
  - Building a custom frontend for a phoneme-based engine

For Hindi/Gujarati/Marathi in Parler-TTS, you pass the sentence in
Devanagari/Gujarati script. Parler-TTS has limited multilingual support —
see the test results in Task 3 to evaluate quality.

Below we map your phoneme set → IPA anyway, so you have it for
rule-based backends (eSpeak, Festival) or future use.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: YOUR G2P PHONEME SET → IPA MAPPING
# ─────────────────────────────────────────────────────────────────────────────

G2P_TO_IPA = {
    # ── Vowels ──────────────────────────────────────────────────────────────
    "a":   "ə",      # schwa / inherent vowel
    "aa":  "aː",     # long A  (आ)
    "i":   "ɪ",      # short I (इ)
    "ii":  "iː",     # long I  (ई)
    "u":   "ʊ",      # short U (उ)
    "uu":  "uː",     # long U  (ऊ)
    "e":   "eː",     # E       (ए)
    "ee":  "eː",     # long E  (ए / ऐ variant)
    "ai":  "ɛː",     # AI      (ऐ)
    "o":   "oː",     # O       (ओ)
    "oo":  "oː",     # long O  (ओ variant)
    "au":  "ɔː",     # AU      (औ)

    # ── Nasalized vowels ─────────────────────────────────────────────────────
    "mq":  "̃",       # nasalization marker (anusvara  ं / chandrabindu ँ)
    "q":   "̃",       # alternate nasalization in dataset
    "q ":  "̃",       # with trailing space variant

    # ── Stops — Unaspirated ──────────────────────────────────────────────────
    "k":   "k",
    "g":   "ɡ",
    "c":   "tʃ",     # च
    "j":   "dʒ",     # ज
    "t":   "t̪",      # dental T  (त)
    "d":   "d̪",      # dental D  (द)
    "p":   "p",
    "b":   "b",

    # ── Stops — Aspirated ────────────────────────────────────────────────────
    "kh":  "kʰ",     # ख
    "gh":  "ɡʱ",     # घ
    "ch":  "tʃʰ",    # छ
    "jh":  "dʒʱ",    # झ
    "th":  "t̪ʰ",     # dental TH (थ)
    "dh":  "d̪ʱ",     # dental DH (ध)
    "ph":  "pʰ",     # फ
    "bh":  "bʱ",     # भ

    # ── Retroflex — Unaspirated ───────────────────────────────────────────────
    "tx":  "ʈ",      # ट
    "dx":  "ɖ",      # ड
    "nx":  "ɳ",      # ण

    # ── Retroflex — Aspirated ─────────────────────────────────────────────────
    "txh": "ʈʰ",     # ठ
    "dxh": "ɖʱ",     # ढ

    # ── Retroflex flaps (used in medial position) ─────────────────────────────
    "dxq": "ɽ",      # ड़  (flap)
    "dxqh":"ɽʱ",     # ढ़  (aspirated flap)
    "lx":  "ɭ",      # retroflex L (Gujarati/Marathi ळ)

    # ── Nasals ────────────────────────────────────────────────────────────────
    "n":   "n",
    "m":   "m",
    "ng":  "ŋ",      # velar nasal

    # ── Fricatives ────────────────────────────────────────────────────────────
    "s":   "s",
    "z":   "z",
    "sh":  "ʃ",      # श
    "sx":  "ʂ",      # retroflex SH  ष
    "h":   "ɦ",      # voiced H  ह
    "f":   "f",

    # ── Approximants / Sonorants ──────────────────────────────────────────────
    "y":   "j",      # य
    "r":   "r",      # र  (trill/tap)
    "rq":  "ɾ",      # flap R variant
    "l":   "l",
    "v":   "ʋ",      # व
    "w":   "w",

    # ── Extra vowel / diphthong tokens found in Gujarati/Marathi data ───────
    "ei":  "eɪ",     # diphthong EI  (Gujarati)
    "ou":  "oʊ",     # diphthong OU  (Gujarati/Marathi)
    "ae":  "æ",      # front A       (loan words)
    "ax":  "ɐ",      # near-open central (Marathi schwa variant)

    # ── Extra consonant clusters in Gujarati/Marathi ──────────────────────
    "nj":  "ndʒ",    # prenasalized affricate  ञ
    "hq":  "ɦ̃",     # nasalized H  (Marathi)
    "dxhq":"ɖ̃ʱ",    # nasalized retroflex aspirated (Marathi)
    "kq":  "k̃",      # nasalized K  (rare)
    "khq": "kʰ̃",    # nasalized aspirated K
    "gq":  "ɡ̃",      # nasalized G  (rare)

    # ── Special / Punctuation ─────────────────────────────────────────────────
    "|":   ".",       # phrase boundary / full stop
    "ˈ":   "ˈ",      # primary stress (if present)
    "ˌ":   "ˌ",      # secondary stress (if present)
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CONVERT A PHONEME SEQUENCE TO IPA
# ─────────────────────────────────────────────────────────────────────────────

def g2p_to_ipa(phoneme_list):
    """
    Convert a list of G2P phoneme strings to a single IPA string.
    Unknown phonemes are kept as-is with a '?' prefix.
    """
    ipa_tokens = []
    for ph in phoneme_list:
        ph = ph.strip()
        if not ph:
            continue
        if ph in G2P_TO_IPA:
            ipa_tokens.append(G2P_TO_IPA[ph])
        else:
            ipa_tokens.append(f"?{ph}")   # flag unmapped
    return "".join(ipa_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SCAN DATASET — FIND ALL PHONEMES USED
# ─────────────────────────────────────────────────────────────────────────────

def scan_phonemes(dataset_path):
    """Read the dataset and collect every unique phoneme token used."""
    phoneme_counter = Counter()
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            phonemes = parts[1].strip().split()
            for ph in phonemes:
                phoneme_counter[ph] += 1
    return phoneme_counter


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MAIN — RUN VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATASET_PATH = "multilingual_g2p_dataset.txt"

    print("=" * 65)
    print("WEEK 3 — TASK 1: G2P PHONEME → IPA MAPPING")
    print("=" * 65)

    # ── 5a. Print the full mapping table ─────────────────────────────────────
    print("\n📋 YOUR PHONEME SET → IPA MAPPING TABLE")
    print(f"{'G2P Symbol':<14} {'IPA':<10} {'Example'}")
    print("-" * 50)
    examples = {
        "a": "schwa (अ)", "aa": "आ", "ii": "ई", "uu": "ऊ",
        "mq": "nasalization ँ/ं", "k": "क", "kh": "ख",
        "tx": "ट (retroflex)", "txh": "ठ (aspirated retroflex)",
        "dxq": "ड़ (flap)", "sh": "श", "sx": "ष", "lx": "ळ",
    }
    for g2p, ipa in sorted(G2P_TO_IPA.items()):
        ex = examples.get(g2p, "")
        print(f"  {g2p:<12} {ipa:<10} {ex}")

    # ── 5b. Scan dataset ──────────────────────────────────────────────────────
    if os.path.exists(DATASET_PATH):
        print(f"\n🔍 SCANNING DATASET: {DATASET_PATH}")
        phoneme_counts = scan_phonemes(DATASET_PATH)
        print(f"   Total unique phoneme tokens found: {len(phoneme_counts)}")

        mapped   = {ph: cnt for ph, cnt in phoneme_counts.items() if ph in G2P_TO_IPA}
        unmapped = {ph: cnt for ph, cnt in phoneme_counts.items() if ph not in G2P_TO_IPA}

        print(f"\n✅ MAPPED   ({len(mapped)} tokens):")
        for ph, cnt in sorted(mapped.items(), key=lambda x: -x[1])[:20]:
            print(f"   '{ph}'  →  '{G2P_TO_IPA[ph]}'   (appears {cnt:,}×)")

        if unmapped:
            print(f"\n⚠️  UNMAPPED ({len(unmapped)} tokens — ADD THESE TO G2P_TO_IPA):")
            for ph, cnt in sorted(unmapped.items(), key=lambda x: -x[1]):
                print(f"   '{ph}'   (appears {cnt:,}×)")
        else:
            print("\n✅ ALL phonemes in the dataset are mapped!")
    else:
        print(f"\n⚠️  Dataset not found at '{DATASET_PATH}'.")
        print("   Copy multilingual_g2p_dataset.txt into the same folder and re-run.")

    # ── 5c. Live conversion demo ──────────────────────────────────────────────
    print("\n🎯 LIVE CONVERSION DEMO")
    print("-" * 50)
    demo_words = [
        ("नमस्ते (namaste)",  ["n", "a", "m", "a", "s", "t", "ee"]),
        ("पानी (paanee)",     ["p", "aa", "n", "ii"]),
        ("भारत (Bharat)",     ["bh", "aa", "r", "a", "t", "a"]),
        ("ठंड (thand)",       ["txh", "a", "mq", "d", "a"]),
        ("ड़ (da flap)",       ["dxq", "a"]),
    ]
    for word, phonemes in demo_words:
        ipa = g2p_to_ipa(phonemes)
        print(f"  {word}")
        print(f"    G2P : {' '.join(phonemes)}")
        print(f"    IPA : [{ipa}]")
        print()

    # ── 5d. Parler-TTS format note ────────────────────────────────────────────
    print("=" * 65)
    print("📌 HOW TO FEED TEXT INTO PARLER-TTS (your TTS backend)")
    print("=" * 65)
    print("""
Parler-TTS expects PLAIN TEXT (Devanagari / Gujarati script), NOT IPA.

Example usage (Python):
------------------------------------------------------
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import torch, soundfile as sf

model_id = "parler-tts/parler-tts-mini-v1"
model     = ParlerTTSForConditionalGeneration.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Plain text — Devanagari script, no IPA needed
text        = "नमस्ते, आप कैसे हैं?"
description = "A female speaker delivers a clear Hindi sentence."

inputs = tokenizer(description, return_tensors="pt")
prompt = tokenizer(text, return_tensors="pt")

generation = model.generate(
    input_ids=inputs.input_ids,
    prompt_input_ids=prompt.input_ids
)
audio = generation.cpu().numpy().squeeze()
sf.write("output.wav", audio, model.config.sampling_rate)
------------------------------------------------------

Your G2P output is used to:
  • Verify / correct pronunciation in a rule-based backend
  • Drive eSpeak-NG or Festival with explicit phoneme strings
  • Build a custom phoneme-level TTS frontend
""")

    print("Task 1 complete. Run Task 2 next: python week3_task2_schwa_deletion.py")
