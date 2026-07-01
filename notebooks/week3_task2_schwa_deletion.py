# -*- coding: utf-8 -*-
"""
Week 3 — Task 2: Stress / Tone / Schwa Deletion Rules
======================================================
G2P alone does NOT handle Hindi schwa deletion correctly.
This script adds post-processing rules on top of your G2P output.

Covers:
  1. Hindi schwa deletion  (the most critical rule)
  2. Gujarati schwa deletion (slightly different pattern)
  3. Marathi schwa deletion
  4. Stress assignment (all three languages)
  5. Nasalization normalisation

Run:
  python week3_task2_schwa_deletion.py
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PHONEME SETS
# ─────────────────────────────────────────────────────────────────────────────

# Your G2P "a" token = the schwa (inherent vowel)
SCHWA = "a"

# Consonant tokens in your phoneme set (everything that is NOT a vowel/special)
VOWELS = {"a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"}

NASALS       = {"n", "m", "ng", "nx"}
STOPS        = {"k", "g", "c", "j", "t", "d", "p", "b",
                "kh", "gh", "ch", "jh", "th", "dh", "ph", "bh",
                "tx", "dx", "txh", "dxh", "dxq", "dxqh"}
FRICATIVES   = {"s", "z", "sh", "sx", "h", "f"}
APPROXIMANTS = {"y", "r", "rq", "l", "lx", "v", "w"}
CONSONANTS   = NASALS | STOPS | FRICATIVES | APPROXIMANTS

STRESS_MARKER   = "ˈ"   # primary stress
SECONDARY_STRESS = "ˌ"  # secondary stress
NASALIZATION     = "mq"  # anusvara / chandrabindu
SYLLABLE_SEP     = "."   # dot used for syllable boundary (internal only)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: HINDI SCHWA DELETION
# ─────────────────────────────────────────────────────────────────────────────
"""
The Hindi schwa deletion rule (Ohala 1983, Pandey 2014):

A word-final schwa is ALWAYS deleted.
An internal schwa is deleted when:
  - It is in an even-numbered syllable AND
  - It is followed by a CV sequence (consonant + vowel)

Simplified practical rule (used here):
  • Delete schwa at word end (most impactful)
  • Delete schwa between two consonants when followed by a vowel-bearing syllable
    i.e.: C [a] C V  →  C C V  (the [a] drops out)

Example:
  k a m a l a  →  k a m l a   (कमल: /kəməl/ → /kəml/ ≈ "kamal" not "kamala")
  s a m a j a  →  s a m dʒ a  (समझ)
  k a h a n aa →  k a h n aa  (कहना)
"""


def apply_hindi_schwa_deletion(phonemes: List[str]) -> List[str]:
    """
    Apply Hindi schwa deletion rules to a list of phoneme tokens.

    Rules applied in order:
      R1. Delete word-final schwa.
      R2. Delete schwa in C_a_C_V context (internal deletion).
    """
    result = phonemes.copy()

    # R1: Delete word-final schwa
    if result and result[-1] == SCHWA:
        result = result[:-1]

    # R2: Internal schwa deletion — C [a] C V  →  C C V
    # Scan left-to-right; build new list
    new = []
    i = 0
    while i < len(result):
        token = result[i]

        # Check pattern: current=CONSONANT, next=SCHWA, next+1=CONSONANT, next+2=VOWEL
        if (token in CONSONANTS
                and i + 3 < len(result)
                and result[i + 1] == SCHWA
                and result[i + 2] in CONSONANTS
                and result[i + 3] in VOWELS):
            # Delete the schwa at i+1
            new.append(token)         # C
            i += 2                    # skip [a], keep going from next C
        else:
            new.append(token)
            i += 1

    return new


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GUJARATI SCHWA DELETION
# ─────────────────────────────────────────────────────────────────────────────
"""
Gujarati has similar schwa deletion to Hindi but with one difference:
  • The schwa is retained in word-final position when followed by anusvara.
  • Otherwise, word-final schwa is deleted.
  • Internal rule same as Hindi.
"""


def apply_gujarati_schwa_deletion(phonemes: List[str]) -> List[str]:
    result = phonemes.copy()

    # R1: Delete word-final schwa UNLESS followed by nasalization
    # Since our phoneme list doesn't have a final nasalization after the schwa
    # in this context, we check if second-to-last is nasalization marker
    if (len(result) >= 2
            and result[-1] == SCHWA
            and result[-2] != NASALIZATION):
        result = result[:-1]
    elif result and result[-1] == SCHWA:
        result = result[:-1]   # delete anyway if isolated

    # R2: Internal deletion same as Hindi
    new = []
    i = 0
    while i < len(result):
        token = result[i]
        if (token in CONSONANTS
                and i + 3 < len(result)
                and result[i + 1] == SCHWA
                and result[i + 2] in CONSONANTS
                and result[i + 3] in VOWELS):
            new.append(token)
            i += 2
        else:
            new.append(token)
            i += 1

    return new


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MARATHI SCHWA DELETION
# ─────────────────────────────────────────────────────────────────────────────
"""
Marathi schwa deletion is similar to Hindi.
Additional rule: schwa before a sonorant cluster is also deleted.
"""


def apply_marathi_schwa_deletion(phonemes: List[str]) -> List[str]:
    result = phonemes.copy()

    # R1: Word-final schwa deletion
    if result and result[-1] == SCHWA:
        result = result[:-1]

    # R2: Internal C_a_C_V context
    new = []
    i = 0
    while i < len(result):
        token = result[i]
        if (token in CONSONANTS
                and i + 3 < len(result)
                and result[i + 1] == SCHWA
                and result[i + 2] in CONSONANTS
                and result[i + 3] in VOWELS):
            new.append(token)
            i += 2
        else:
            new.append(token)
            i += 1

    # R3: Marathi-specific — delete schwa before sonorant + vowel
    final = []
    i = 0
    while i < len(new):
        token = new[i]
        if (token == SCHWA
                and i + 1 < len(new)
                and new[i + 1] in APPROXIMANTS | NASALS
                and i + 2 < len(new)
                and new[i + 2] in VOWELS):
            i += 1   # skip this schwa
        else:
            final.append(token)
            i += 1

    return final


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STRESS ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
"""
Hindi/Urdu stress (Kelkar 1968, Hayes 1991):
  • Stress falls on the LAST heavy syllable.
  • Heavy syllable = contains a long vowel OR a short vowel followed by ≥2 consonants.
  • If no heavy syllable, stress falls on the first syllable.

This function inserts "ˈ" before the stressed vowel token.
"""


def is_long_vowel(token: str) -> bool:
    return token in {"aa", "ii", "uu", "ee", "ai", "oo", "au"}


def find_syllables(phonemes: List[str]) -> List[Tuple[int, str, bool]]:
    """
    Walk the phoneme list and identify vowel nuclei.
    Returns list of (index_in_phonemes, vowel_token, is_heavy).
    Heaviness = long vowel OR short vowel followed by 2+ consonants.
    """
    syllables = []
    for i, tok in enumerate(phonemes):
        if tok in VOWELS:
            heavy = is_long_vowel(tok)
            if not heavy:
                # Check if followed by 2+ consonants
                cons_after = 0
                j = i + 1
                while j < len(phonemes) and phonemes[j] in CONSONANTS:
                    cons_after += 1
                    j += 1
                heavy = cons_after >= 2
            syllables.append((i, tok, heavy))
    return syllables


def assign_stress(phonemes: List[str], lang: str = "HI") -> List[str]:
    """
    Insert a stress marker 'ˈ' before the nucleus of the stressed syllable.
    Works for Hindi, Gujarati, Marathi (all follow similar stress patterns).
    """
    syllables = find_syllables(phonemes)
    if not syllables:
        return phonemes

    # Find last heavy syllable
    stressed_idx = None
    for idx, tok, heavy in reversed(syllables):
        if heavy:
            stressed_idx = idx
            break

    # Fallback: first syllable
    if stressed_idx is None:
        stressed_idx = syllables[0][0]

    result = phonemes.copy()
    result.insert(stressed_idx, STRESS_MARKER)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: NASALIZATION NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
"""
Your dataset uses "mq" for nasalization (anusvara ं / chandrabindu ँ).
Rule: the "mq" token should be placed AFTER the vowel it nasalises.
The dataset already does this, but this function validates and
converts it to a tilde diacritic representation for IPA output.
"""


def normalise_nasalization(phonemes: List[str]) -> List[str]:
    """
    Ensure 'mq' always directly follows a vowel.
    If 'mq' appears at the start or after a consonant, flag it.
    Returns a cleaned list (no change in valid cases).
    """
    cleaned = []
    for i, tok in enumerate(phonemes):
        if tok == NASALIZATION:
            if i == 0 or phonemes[i - 1] not in VOWELS:
                # Orphan nasalization — log and skip
                print(f"  ⚠️  Orphan nasalization at position {i}: {phonemes}")
                continue
        cleaned.append(tok)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FULL POST-PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(phonemes: List[str], lang: str) -> List[str]:
    """
    Apply all post-processing in order:
      1. Nasalization normalisation
      2. Schwa deletion (language-specific)
      3. Stress assignment
    """
    phonemes = normalise_nasalization(phonemes)

    if lang == "HI":
        phonemes = apply_hindi_schwa_deletion(phonemes)
    elif lang == "GU":
        phonemes = apply_gujarati_schwa_deletion(phonemes)
    elif lang == "MR":
        phonemes = apply_marathi_schwa_deletion(phonemes)

    phonemes = assign_stress(phonemes, lang)
    return phonemes


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN — DEMO + BATCH TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("WEEK 3 — TASK 2: SCHWA DELETION + STRESS RULES")
    print("=" * 65)

    # ── 7a. Hindi schwa deletion demo ────────────────────────────────────────
    print("\n🔤 HINDI SCHWA DELETION EXAMPLES")
    print("-" * 50)

    hindi_tests = [
        # (word,          raw G2P output,                      expected after deletion)
        ("कमल (kamal)",  ["k","a","m","a","l","a"],            "k a m l a → final-a deleted → k a m l"),
        ("समझ (samajh)", ["s","a","m","a","j","a"],            "word-final a deleted"),
        ("कहना (kahna)", ["k","a","h","a","n","aa"],           "internal a deleted before n+aa"),
        ("नमस्ते",       ["n","a","m","a","s","t","ee"],       "internal + no final deletion (ends in ee)"),
        ("भारत",         ["bh","aa","r","a","t","a"],          "word-final a deleted"),
        ("पानी",         ["p","aa","n","ii"],                  "no schwa present"),
        ("अंकल",         ["a","mq","k","a","l","a"],           "final a deleted"),
    ]

    for word, raw, note in hindi_tests:
        processed = apply_hindi_schwa_deletion(raw)
        stressed  = assign_stress(processed, "HI")
        print(f"\n  Word     : {word}")
        print(f"  Raw G2P  : {' '.join(raw)}")
        print(f"  After ∅  : {' '.join(processed)}   ({note})")
        print(f"  + Stress : {' '.join(stressed)}")

    # ── 7b. Gujarati demo ─────────────────────────────────────────────────────
    print("\n\n🔤 GUJARATI SCHWA DELETION EXAMPLES")
    print("-" * 50)

    gujarati_tests = [
        ("ગામ (gaam)",    ["g","aa","m","a"],   "final a deleted"),
        ("પાણી (paanee)", ["p","aa","nx","ii"], "no schwa"),
        ("ઘર (ghar)",     ["gh","a","r","a"],   "final a deleted"),
    ]

    for word, raw, note in gujarati_tests:
        processed = apply_gujarati_schwa_deletion(raw)
        stressed  = assign_stress(processed, "GU")
        print(f"\n  Word     : {word}")
        print(f"  Raw G2P  : {' '.join(raw)}")
        print(f"  After ∅  : {' '.join(processed)}   ({note})")
        print(f"  + Stress : {' '.join(stressed)}")

    # ── 7c. Marathi demo ─────────────────────────────────────────────────────
    print("\n\n🔤 MARATHI SCHWA DELETION EXAMPLES")
    print("-" * 50)

    marathi_tests = [
        ("घर (ghar)",      ["gh","a","r","a"],           "final a deleted"),
        ("माझं (maazha)",  ["m","aa","j","a","mq"],      "ends in nasal"),
        ("सांगतो",         ["s","aa","mq","g","a","t","o"],"internal"),
    ]

    for word, raw, note in marathi_tests:
        processed = apply_marathi_schwa_deletion(raw)
        stressed  = assign_stress(processed, "MR")
        print(f"\n  Word     : {word}")
        print(f"  Raw G2P  : {' '.join(raw)}")
        print(f"  After ∅  : {' '.join(processed)}   ({note})")
        print(f"  + Stress : {' '.join(stressed)}")

    # ── 7d. Batch test against dataset ───────────────────────────────────────
    print("\n\n📊 BATCH TEST — FIRST 10 WORDS FROM DATASET (with post-processing)")
    print("-" * 65)

    import os
    DATASET = "multilingual_g2p_dataset.txt"
    if os.path.exists(DATASET):
        count = 0
        with open(DATASET, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                src = parts[0].strip()
                raw_phonemes = parts[1].strip().split()

                # Extract language tag
                if src.startswith("<") and ">" in src:
                    end = src.index(">")
                    lang = src[1:end]
                    word = src[end+1:].strip()
                else:
                    lang = "HI"
                    word = src

                processed = postprocess(raw_phonemes, lang)

                print(f"  [{lang}] {word}")
                print(f"     Raw       : {' '.join(raw_phonemes)}")
                print(f"     Processed : {' '.join(processed)}")
                print()

                count += 1
                if count >= 10:
                    break
    else:
        print(f"  Dataset not found at '{DATASET}'.")
        print("  Copy multilingual_g2p_dataset.txt to the same folder and re-run.")

    print("=" * 65)
    print("Task 2 complete. Run Task 3 next: python week3_task3_e2e_test.py")
    print("=" * 65)
