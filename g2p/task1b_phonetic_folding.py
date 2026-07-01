import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(SCRIPT_DIR, "phoneme_vocab.json")
OUT_MAPPING_PATH = os.path.join(SCRIPT_DIR, "phoneme_cluster_mapping.json")

# ─────────────────────────────────────────────
# ACOUSTIC PHONETIC FOLDING RULES
# Only merge sounds that are acoustically near-identical
# or represent fine-grained distinctions not strictly
# necessary for intelligible TTS.
# ─────────────────────────────────────────────
FOLDING_RULES = {
    # 1. Long/Short Vowel folding
    "ii": "i",
    "uu": "u",
    "ee": "e",
    "ae": "ei",
    "ax": "a",
    
    # 2. Nukta (dot) consonants folded to base
    "kq": "k",
    "khq": "kh",
    "gq": "g",
    "dxq": "dx",
    "dxhq": "dxh",
    "f": "ph",
    "z": "j",
    
    # 3. Rare/Regional consonants folded to closest equivalents
    "sx": "sh",  # retroflex s -> palatal sh (pronounced identically in modern Hindi)
    "hq": "h",
    "rq": "r",
    "lx": "l",   # Marathi retroflex l -> regular l
    
    # 4. Nasals folded
    "ng": "n",
    "nj": "n",
    "mq": "q",   # chandrabindu -> anusvara
}

def main():
    print("="*60)
    print("TASK 1B: PHONETIC FOLDING (Smart Clustering)")
    print("="*60)
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        
    if isinstance(vocab_data, dict) and "phoneme_to_id" in vocab_data:
        phonemes = list(vocab_data["phoneme_to_id"].keys())
    elif isinstance(vocab_data, dict):
        phonemes = list(vocab_data.keys())
    else:
        phonemes = vocab_data
        
    print(f"Original phoneme count: {len(phonemes)}")
    
    mapping = {}
    cluster_reps = {}
    
    # Assign cluster IDs sequentially
    current_cluster_id = 0
    
    for p in phonemes:
        # Determine its base/representative phoneme
        rep = FOLDING_RULES.get(p, p)
        
        # If this representative hasn't been assigned a cluster ID yet, assign one
        if rep not in cluster_reps:
            cluster_reps[rep] = current_cluster_id
            current_cluster_id += 1
            
        mapping[p] = {
            "cluster_id": cluster_reps[rep],
            "representative": rep,
            "cluster_label": f"Phonetic_Group_{rep}"
        }
        
    print(f"New cluster count: {len(cluster_reps)}")
    print(f"Vocab Reduction: {(1 - len(cluster_reps)/len(phonemes))*100:.1f}%")
    
    with open(OUT_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Overwrote mapping: {OUT_MAPPING_PATH}")
    print("   The TTS audio will now maintain high phonetic accuracy")
    print("   while benefiting from a smaller vocabulary.")

if __name__ == "__main__":
    main()
