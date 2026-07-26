

import os
import csv
import json
import random
import argparse
import shutil
from collections import defaultdict

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


TARGET_SR = 22050
MONO = True
TRAIN_RATIO = 0.90
VAL_RATIO   = 0.05
TEST_RATIO  = 0.05

LANG_CODES = {
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
}

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

G2P_DATASET_PATH  = os.path.join(PROJECT_DIR, "data", "multilingual_g2p_dataset.txt")
CLUSTER_MAPPING   = os.path.join(SCRIPT_DIR, "phoneme_cluster_mapping.json")


def load_g2p_dictionary(dataset_path):
    g2p_dict = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
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
    return g2p_dict


def load_cluster_mapping(mapping_path):
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {p: f"C{info['cluster_id']}" for p, info in mapping.items()}


def discover_audio_files(input_dir, lang):
    lang_dir = os.path.join(input_dir, lang)
    if not os.path.isdir(lang_dir):
        print(f"  ⚠️  Directory not found: {lang_dir}")
        return []

    transcript_file = None
    for f in os.listdir(lang_dir):
        if f.endswith('.txt') and 'mono' in f.lower():
            transcript_file = os.path.join(lang_dir, f)
            break

    if not transcript_file:
        print(f"  ⚠️  No transcript file found in {lang_dir}")
        return []

    wav_dir = os.path.join(lang_dir, 'wav')
    if not os.path.isdir(wav_dir):
        print(f"  ⚠️  No wav/ directory found in {lang_dir}")
        return []

    pairs = []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            txt_filename = parts[0].strip()
            text = parts[1].strip()
            base = txt_filename.replace('.txt', '')
            wav_filename = base + '.wav'
            wav_path = os.path.join(wav_dir, wav_filename)
            if os.path.exists(wav_path):
                pairs.append({
                    'wav_path': wav_path,
                    'text': text,
                    'filename': os.path.splitext(wav_filename)[0]
                })

    return pairs


def normalize_audio(wav_path, output_path, target_sr=22050):
    if not HAS_AUDIO:
        shutil.copy2(wav_path, output_path)
        return

    try:
        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        sf.write(output_path, y_trimmed, target_sr)
    except Exception as e:
        print(f"  ⚠️  Error processing {wav_path}: {e}")
        shutil.copy2(wav_path, output_path)


def split_data(pairs, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05, seed=42):
    random.seed(seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    train = [pairs[i] for i in indices[:train_end]]
    val   = [pairs[i] for i in indices[train_end:val_end]]
    test  = [pairs[i] for i in indices[val_end:]]

    return train, val, test


def create_split_directory(pairs, output_dir, lang, split_name, g2p_dict, cluster_map, normalize=True):
    
    split_dir = os.path.join(output_dir, lang, split_name)
    wavs_dir  = os.path.join(split_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    metadata = []
    for pair in pairs:
        filename = pair["filename"]
        text     = pair["text"]
        wav_src  = pair["wav_path"]
        wav_dst  = os.path.join(wavs_dir, filename + ".wav")

        if normalize:
            normalize_audio(wav_src, wav_dst, TARGET_SR)
        else:
            shutil.copy2(wav_src, wav_dst)

        words = text.strip().split()
        phoneme_parts = []
        cluster_parts = []
        for word in words:
            key = (lang, word)
            if key in g2p_dict:
                phonemes = g2p_dict[key]
                phoneme_parts.append(phonemes)
                cluster_tokens = []
                for p in phonemes.split():
                    if p in cluster_map:
                        cluster_tokens.append(cluster_map[p])
                    else:
                        cluster_tokens.append(p)
                cluster_parts.append(" ".join(cluster_tokens))
            else:
                phoneme_parts.append(word)
                cluster_parts.append(word)

        phoneme_seq = " | ".join(phoneme_parts)
        cluster_seq = " | ".join(cluster_parts)

        metadata.append({
            "filename": filename,
            "text": text,
            "phonemes": phoneme_seq,
            "clusters": cluster_seq,
        })

    meta_path = os.path.join(split_dir, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for m in metadata:
            writer.writerow([m["filename"], m["text"], m["phonemes"]])

    #cluster metadata
    cluster_meta_path = os.path.join(split_dir, "metadata_clustered.csv")
    with open(cluster_meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for m in metadata:
            writer.writerow([m["filename"], m["text"], m["clusters"]])

    return len(metadata)


def main():
    parser = argparse.ArgumentParser(description="Prepare TTS training data in LJ-Speech format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to IndicTTS data root (should have hi/, gu/, mr/ subdirectories)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/tts/ in project root)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip audio normalization (just copy files)")
    parser.add_argument("--languages", type=str, nargs="+", default=["hi", "gu", "mr"],
                        help="Languages to process (default: hi gu mr)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_DIR, "data", "tts")

    if not HAS_AUDIO and not args.no_normalize:
        print("⚠️  librosa/soundfile not installed. Audio normalization disabled.")
        print("   Install with: pip install librosa soundfile")
        print("   Falling back to file copy.\n")

    # Load G2P dictionary for phoneme alignment
    print("Loading G2P dictionary...")
    g2p_dict = load_g2p_dictionary(G2P_DATASET_PATH)
    print(f"  Loaded {len(g2p_dict)} word→phoneme entries")

    # Load cluster mapping
    print("Loading cluster mapping...")
    cluster_map = load_cluster_mapping(CLUSTER_MAPPING)
    print(f"  Loaded mapping for {len(cluster_map)} phonemes")

    print(f"\nInput directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Languages:        {args.languages}")
    print(f"Target SR:        {TARGET_SR} Hz")
    print(f"Normalize audio:  {not args.no_normalize}")

    # Process each language
    total_stats = {}
    for lang in args.languages:
        print(f"\n{'='*50}")
        print(f"Processing: {lang.upper()} ({LANG_CODES.get(lang, 'Unknown')})")
        print(f"{'='*50}")

        # Discover audio files
        pairs = discover_audio_files(args.input_dir, lang)
        if not pairs:
            print(f"  ❌ No audio+transcript pairs found for {lang}")
            print(f"     Looked in: {os.path.join(args.input_dir, lang)}")
            continue

        print(f"  Found {len(pairs)} audio+transcript pairs")

        # Split
        train, val, test = split_data(pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")

        # Create directories and process
        normalize = not args.no_normalize
        n_train = create_split_directory(train, args.output_dir, lang, "train", g2p_dict, cluster_map, normalize)
        n_val   = create_split_directory(val,   args.output_dir, lang, "val",   g2p_dict, cluster_map, normalize)
        n_test  = create_split_directory(test,  args.output_dir, lang, "test",  g2p_dict, cluster_map, normalize)

        total_stats[lang] = {"train": n_train, "val": n_val, "test": n_test, "total": len(pairs)}
        print(f"  ✅ {lang.upper()} complete: {n_train} train / {n_val} val / {n_test} test")

    # Summary
    print(f"\n{'='*60}")
    print("TTS DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput: {args.output_dir}")
    for lang, stats in total_stats.items():
        print(f"  {lang.upper()}: {stats['total']} total → {stats['train']} train / {stats['val']} val / {stats['test']} test")
    print(f"\nEach split directory contains:")
    print(f"  metadata.csv           — filename|text|phoneme_sequence")
    print(f"  metadata_clustered.csv — filename|text|cluster_sequence")
    print(f"  wavs/                  — normalized audio files ({TARGET_SR} Hz, mono)")


if __name__ == "__main__":
    main()
