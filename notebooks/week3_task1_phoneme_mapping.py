# -*- coding: utf-8 -*-


import json
import os
from collections import Counter

# SECTION 1: PARLER-TTS EXPECTED FORMAT — IMPORTANT CONTEXT


# SECTION 2: YOUR G2P PHONEME SET → IPA MAPPING

G2P_TO_IPA = {
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

    "mq":  "̃",       # nasalization marker (anusvara  ं / chandrabindu ँ)
    "q":   "̃",       # alternate nasalization in dataset
    "q ":  "̃",       # with trailing space variant

    "k":   "k",
    "g":   "ɡ",
    "c":   "tʃ",     # च
    "j":   "dʒ",     # ज
    "t":   "t̪",      # dental T  (त)
    "d":   "d̪",      # dental D  (द)
    "p":   "p",
    "b":   "b",

    "kh":  "kʰ",     # ख
    "gh":  "ɡʱ",     # घ
    "ch":  "tʃʰ",    # छ
    "jh":  "dʒʱ",    # झ
    "th":  "t̪ʰ",     # dental TH (थ)
    "dh":  "d̪ʱ",     # dental DH (ध)
    "ph":  "pʰ",     # फ
    "bh":  "bʱ",     # भ

    "tx":  "ʈ",      # ट
    "dx":  "ɖ",      # ड
    "nx":  "ɳ",      # ण

    "txh": "ʈʰ",     # ठ
    "dxh": "ɖʱ",     # ढ

    "dxq": "ɽ",      # ड़  (flap)
    "dxqh":"ɽʱ",     # ढ़  (aspirated flap)
    "lx":  "ɭ",      # retroflex L (Gujarati/Marathi ळ)

    "n":   "n",
    "m":   "m",
    "ng":  "ŋ",      # velar nasal

    "s":   "s",
    "z":   "z",
    "sh":  "ʃ",      # श
    "sx":  "ʂ",      # retroflex SH  ष
    "h":   "ɦ",      # voiced H  ह
    "f":   "f",

    "y":   "j",      # य
    "r":   "r",      # र  (trill/tap)
    "rq":  "ɾ",      # flap R variant
    "l":   "l",
    "v":   "ʋ",      # व
    "w":   "w",

    "ei":  "eɪ",     # diphthong EI  (Gujarati)
    "ou":  "oʊ",     # diphthong OU  (Gujarati/Marathi)
    "ae":  "æ",      # front A       (loan words)
    "ax":  "ɐ",      # near-open central (Marathi schwa variant)

    "nj":  "ndʒ",    # prenasalized affricate  ञ
    "hq":  "ɦ̃",     # nasalized H  (Marathi)
    "dxhq":"ɖ̃ʱ",    # nasalized retroflex aspirated (Marathi)
    "kq":  "k̃",      # nasalized K  (rare)
    "khq": "kʰ̃",    # nasalized aspirated K
    "gq":  "ɡ̃",      # nasalized G  (rare)

    "|":   ".",       # phrase boundary / full stop
    "ˈ":   "ˈ",      # primary stress (if present)
    "ˌ":   "ˌ",      # secondary stress (if present)
}

# SECTION 3: CONVERT A PHONEME SEQUENCE TO IPA

def g2p_to_ipa(phoneme_list):
    
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


# SECTION 4: SCAN DATASET — FIND ALL PHONEMES USED

def scan_phonemes(dataset_path):
    
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


# SECTION 5: MAIN — RUN VERIFICATION

if __name__ == "__main__":

    DATASET_PATH = "multilingual_g2p_dataset.txt"

    print("=" * 65)
    print("WEEK 3 — TASK 1: G2P PHONEME → IPA MAPPING")
    print("=" * 65)

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

    print("=" * 65)
    print("📌 HOW TO FEED TEXT INTO PARLER-TTS (your TTS backend)")
    print("=" * 65)
    print()

    print("Task 1 complete. Run Task 2 next: python week3_task2_schwa_deletion.py")
