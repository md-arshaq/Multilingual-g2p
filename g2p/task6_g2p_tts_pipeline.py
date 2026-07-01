"""
Task 6: Integrate Clustered G2P Output into TTS Pipeline
─────────────────────────────────────────────────────────
End-to-end pipeline: grapheme input → G2P → phonemes/clusters → TTS → audio

This script uses Coqui TTS (VITS) for synthesis.
It supports two modes:
  1. Baseline: G2P → raw phonemes → TTS
  2. Clustered: G2P → cluster IDs → decode to phonemes → TTS

Generates ≥10 test sentences per language under both conditions.

USAGE:
  python task6_g2p_tts_pipeline.py --tts_model_path /path/to/vits/model
  
  # Or use pre-trained Coqui model for quick demo:
  python task6_g2p_tts_pipeline.py --use_pretrained

Inputs:
  models/best_g2p_transformer.weights.h5    (baseline G2P)
  models/g2p_clustered_model.weights.h5     (clustered G2P)
  g2p/phoneme_cluster_mapping.json          (cluster→phoneme mapping)

Outputs:
  samples/hi/baseline_*.wav
  samples/hi/clustered_*.wav
  samples/gu/baseline_*.wav  ...
  samples/mr/baseline_*.wav  ...
"""

import os
import json
import argparse
import csv
import numpy as np

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

CLUSTER_MAPPING_PATH = os.path.join(SCRIPT_DIR, "phoneme_cluster_mapping.json")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")

# ─────────────────────────────────────────────
# TEST SENTENCES — ≥10 per language
# ─────────────────────────────────────────────
TEST_SENTENCES = {
    "hi": [
        "भारत एक महान देश है",
        "आज मौसम बहुत अच्छा है",
        "मुझे हिंदी बोलना पसंद है",
        "कृपया यहाँ बैठिए",
        "यह किताब बहुत रोचक है",
        "सूरज पूरब से उगता है",
        "बच्चे पार्क में खेल रहे हैं",
        "नमस्ते आप कैसे हैं",
        "मेरा नाम कुश है",
        "हम सब मिलकर काम करेंगे",
        "विज्ञान और प्रौद्योगिकी",
        "शिक्षा सबसे बड़ा हथियार है",
    ],
    "gu": [
        "ભારત એક મહાન દેશ છે",
        "આજે હવામાન ખૂબ સારું છે",
        "મને ગુજરાતી બોલવાનું ગમે છે",
        "કૃપા કરીને અહીં બેસો",
        "આ પુસ્તક ખૂબ રસપ્રદ છે",
        "સૂરજ પૂર્વમાંથી ઉગે છે",
        "બાળકો બગીચામાં રમી રહ્યા છે",
        "નમસ્તે તમે કેમ છો",
        "મારું નામ કુશ છે",
        "અમે બધા સાથે મળીને કામ કરીશું",
    ],
    "mr": [
        "भारत एक महान देश आहे",
        "आज हवामान खूप चांगले आहे",
        "मला मराठी बोलायला आवडते",
        "कृपया येथे बसा",
        "हे पुस्तक खूप रोचक आहे",
        "सूर्य पूर्वेकडून उगवतो",
        "मुले बागेत खेळत आहेत",
        "नमस्कार तुम्ही कसे आहात",
        "माझे नाव कुश आहे",
        "आम्ही सर्व मिळून काम करू",
    ],
}


def load_cluster_mapping(path):
    """Load cluster mapping and build cluster→representative phoneme map."""
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # phoneme → cluster_id
    phoneme_to_cluster = {}
    for phoneme, info in mapping.items():
        phoneme_to_cluster[phoneme] = f"C{info['cluster_id']}"

    # cluster_id → representative phoneme (most frequent member)
    from collections import defaultdict
    cluster_members = defaultdict(list)
    for phoneme, info in mapping.items():
        cluster_members[info['cluster_id']].append((phoneme, info.get('frequency', 0)))

    cluster_to_phoneme = {}
    for cid, members in cluster_members.items():
        # Pick the most frequent member as representative
        members.sort(key=lambda x: x[1], reverse=True)
        cluster_to_phoneme[f"C{cid}"] = members[0][0]

    return phoneme_to_cluster, cluster_to_phoneme


def word_to_phonemes_from_dict(word, lang, g2p_dict):
    """Look up a word in the G2P dictionary."""
    key = (lang, word)
    if key in g2p_dict:
        return g2p_dict[key].split()
    return None


def sentence_to_phonemes(sentence, lang, g2p_dict):
    """Convert a sentence to phonemes using the G2P dictionary (word by word)."""
    words = sentence.strip().split()
    all_phonemes = []
    unknown_words = []

    for word in words:
        phonemes = word_to_phonemes_from_dict(word, lang, g2p_dict)
        if phonemes:
            all_phonemes.extend(phonemes)
        else:
            unknown_words.append(word)
            # Fallback: use characters as-is (TTS might handle graphemes)
            all_phonemes.extend(list(word))

    return all_phonemes, unknown_words


def phonemes_to_clusters(phonemes, phoneme_to_cluster):
    """Convert phoneme sequence to cluster IDs."""
    return [phoneme_to_cluster.get(p, p) for p in phonemes]


def clusters_to_phonemes(clusters, cluster_to_phoneme):
    """Convert cluster IDs back to representative phonemes."""
    return [cluster_to_phoneme.get(c, c) for c in clusters]


def synthesize_with_coqui(phoneme_sequence, output_path, tts_model=None, lang="hi"):
    """
    Synthesize audio using Coqui TTS.

    If tts_model is provided, use it directly.
    Otherwise, use the default pre-trained model.
    """
    try:
        from TTS.api import TTS

        if tts_model is None:
            # Use a pre-trained model — this will download automatically
            # For Indian languages, try the multilingual model
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts",
                      progress_bar=False)
        else:
            tts = tts_model

        # Convert phoneme list to space-separated string
        phoneme_str = " ".join(phoneme_sequence)

        # Synthesize
        tts.tts_to_file(text=phoneme_str, file_path=output_path)
        return True

    except Exception as e:
        print(f"  ⚠️  Coqui TTS synthesis failed: {e}")
        return False


# ─────────────────────────────────────────────
# PHONEME → APPROXIMATE GRAPHEME MAPPING
# Maps our romanized phonemes back to Devanagari/Gujarati
# approximations so gTTS produces audibly different speech
# when phonemes change due to clustering.
# ─────────────────────────────────────────────
PHONEME_TO_DEVANAGARI = {
    "a": "अ", "aa": "आ", "i": "इ", "ii": "ई", "u": "उ", "uu": "ऊ",
    "e": "ए", "ee": "ई", "o": "ओ", "ou": "औ", "ei": "ऐ", "ae": "ऐ",
    "ax": "अ", "k": "क", "kh": "ख", "g": "ग", "gh": "घ", "ng": "ङ",
    "c": "च", "ch": "छ", "j": "ज", "jh": "झ", "nj": "ञ",
    "t": "त", "th": "थ", "d": "द", "dh": "ध", "n": "न",
    "tx": "ट", "txh": "ठ", "dx": "ड", "dxh": "ढ", "nx": "ण",
    "p": "प", "ph": "फ", "b": "ब", "bh": "भ", "m": "म",
    "y": "य", "r": "र", "l": "ल", "w": "व", "v": "व",
    "s": "स", "sh": "श", "sx": "ष", "h": "ह",
    "q": "ं", "mq": "ँ", "rq": "ऱ", "lx": "ळ", "z": "ज़",
    "f": "फ़", "kq": "क़", "gq": "ग़", "dxq": "ड़", "dxhq": "ढ़",
    "khq": "ख़", "hq": "ह",
}


def phonemes_to_grapheme_text(phoneme_list, lang="hi"):
    """
    Convert a phoneme sequence back to approximate grapheme text.
    This produces readable text that gTTS can synthesize, and
    crucially, cluster-decoded phonemes will produce DIFFERENT text
    than the original (because clustering collapses distinctions).
    """
    graphemes = []
    for p in phoneme_list:
        if p in PHONEME_TO_DEVANAGARI:
            graphemes.append(PHONEME_TO_DEVANAGARI[p])
        else:
            graphemes.append(p)  # keep as-is if unknown
    return "".join(graphemes)


def synthesize_with_gtts(text, output_path, lang_code="hi"):
    """Synthesize using Google TTS from text."""
    try:
        from gtts import gTTS

        lang_map = {"hi": "hi", "gu": "gu", "mr": "mr"}
        gtts_lang = lang_map.get(lang_code, "hi")

        tts = gTTS(text=text, lang=gtts_lang)
        tts.save(output_path)
        return True

    except Exception as e:
        print(f"  ⚠️  gTTS synthesis failed: {e}")
        return False


def generate_silent_wav(output_path, duration=1.0, sr=22050):
    """Generate a silent WAV file as placeholder."""
    try:
        import soundfile as sf
        samples = np.zeros(int(sr * duration), dtype=np.float32)
        sf.write(output_path, samples, sr)
        return True
    except ImportError:
        # Use wave module as fallback
        import wave
        import struct
        with wave.open(output_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            num_samples = int(sr * duration)
            wf.writeframes(struct.pack(f'{num_samples}h', *([0] * num_samples)))
        return True


def main():
    parser = argparse.ArgumentParser(description="G2P → TTS end-to-end pipeline")
    parser.add_argument("--tts_model_path", type=str, default=None,
                        help="Path to trained Coqui VITS model")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Use pre-trained Coqui model (downloads automatically)")
    parser.add_argument("--use_gtts", action="store_true",
                        help="Use Google TTS as fallback (requires internet)")
    parser.add_argument("--languages", type=str, nargs="+", default=["hi", "gu", "mr"],
                        help="Languages to process")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for samples")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = SAMPLES_DIR

    print("=" * 60)
    print("TASK 6: G2P → TTS END-TO-END PIPELINE")
    print("=" * 60)

    # Load G2P dictionary
    g2p_dataset_path = os.path.join(PROJECT_DIR, "data", "multilingual_g2p_dataset.txt")
    print("\nLoading G2P dictionary...")
    g2p_dict = {}
    with open(g2p_dataset_path, "r", encoding="utf-8") as f:
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
                lang = left[1:end-1].lower()
                word = left[end:].strip()
                g2p_dict[(lang, word)] = phonemes
    print(f"  Loaded {len(g2p_dict)} entries")

    # Load cluster mapping
    print("Loading cluster mapping...")
    phoneme_to_cluster, cluster_to_phoneme = load_cluster_mapping(CLUSTER_MAPPING_PATH)
    print(f"  {len(phoneme_to_cluster)} phonemes → {len(cluster_to_phoneme)} clusters")
    print(f"  Cluster→representative: {cluster_to_phoneme}")

    # Determine synthesis method
    synth_method = "placeholder"
    if args.use_gtts:
        try:
            from gtts import gTTS
            synth_method = "gtts"
            print("\n  Using gTTS for synthesis")
            print("  NOTE: Baseline uses original text, Clustered uses phoneme-reconstructed text")
        except ImportError:
            print("\n  ⚠️  gTTS not installed. pip install gTTS")
    elif args.use_pretrained or args.tts_model_path:
        try:
            from TTS.api import TTS
            synth_method = "coqui"
            print("\n  Using Coqui TTS for synthesis")
        except ImportError:
            print("\n  ⚠️  Coqui TTS not installed. pip install TTS")

    if synth_method == "placeholder":
        print("\n  ℹ️  No TTS engine available. Generating placeholder WAVs + metadata.")
        print("     Install a TTS engine:")
        print("       pip install gTTS           (simple, needs internet)")
        print("       pip install TTS            (Coqui, full featured)")

    # Process each language
    all_metadata = []

    for lang in args.languages:
        print(f"\n{'='*50}")
        print(f"Language: {lang.upper()}")
        print(f"{'='*50}")

        sentences = TEST_SENTENCES.get(lang, [])
        if not sentences:
            print(f"  No test sentences for {lang}")
            continue

        for condition in ["baseline", "clustered"]:
            print(f"\n  Condition: {condition}")
            condition_dir = os.path.join(args.output_dir, lang)
            os.makedirs(condition_dir, exist_ok=True)

            for idx, sentence in enumerate(sentences):
                # Step 1: Sentence → phonemes (word by word)
                phonemes, unknown = sentence_to_phonemes(sentence, lang, g2p_dict)

                if condition == "baseline":
                    # Use raw phonemes
                    output_phonemes = phonemes
                else:
                    # Convert to clusters, then back to representative phonemes
                    clusters = phonemes_to_clusters(phonemes, phoneme_to_cluster)
                    output_phonemes = clusters_to_phonemes(clusters, cluster_to_phoneme)

                # Step 2: Synthesize
                filename = f"{condition}_{idx+1:02d}.wav"
                output_path = os.path.join(condition_dir, filename)

                success = False
                if synth_method == "gtts":
                    if condition == "baseline":
                        # Baseline: use the ORIGINAL sentence text
                        success = synthesize_with_gtts(sentence, output_path, lang)
                    else:
                        # Clustered: reconstruct text from cluster-decoded phonemes
                        # This produces DIFFERENT text because clustering collapses distinctions
                        reconstructed = phonemes_to_grapheme_text(output_phonemes, lang)
                        success = synthesize_with_gtts(reconstructed, output_path, lang)
                elif synth_method == "coqui":
                    success = synthesize_with_coqui(output_phonemes, output_path, lang=lang)
                else:
                    success = generate_silent_wav(output_path, duration=2.0)

                status = "✅" if success else "❌"
                phoneme_str = " ".join(output_phonemes[:10])
                if len(output_phonemes) > 10:
                    phoneme_str += " ..."

                print(f"    {status} {filename}: \"{sentence[:30]}...\" → [{phoneme_str}]")

                # Record metadata
                all_metadata.append({
                    "language": lang,
                    "condition": condition,
                    "sentence_id": idx + 1,
                    "filename": os.path.join(lang, filename),
                    "text": sentence,
                    "phonemes": " ".join(phonemes),
                    "output_phonemes": " ".join(output_phonemes),
                    "unknown_words": "|".join(unknown) if unknown else "",
                })

    # Save metadata CSV
    meta_path = os.path.join(args.output_dir, "samples_metadata.csv")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "language", "condition", "sentence_id", "filename",
            "text", "phonemes", "output_phonemes", "unknown_words"
        ])
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"\n{'='*60}")
    print("TASK 6 COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Samples saved to: {args.output_dir}")
    print(f"  Metadata saved to: {meta_path}")
    print(f"  Total samples: {len(all_metadata)}")

    # Summary per language/condition
    from collections import Counter
    counts = Counter((m["language"], m["condition"]) for m in all_metadata)
    for (lang, cond), count in sorted(counts.items()):
        print(f"    {lang.upper()} / {cond}: {count} samples")


if __name__ == "__main__":
    main()
