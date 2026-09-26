#!/usr/bin/env python3
"""Pre-training validation: GPU, dependencies, data, and tokenizer checks."""
import sys, os

print("=" * 60)
print("PRE-TRAINING VALIDATION REPORT")
print("=" * 60)

errors = []

# === 1. GPU Check ===
print("\n--- 1. GPU & CUDA ---")
import torch
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"  VRAM        : {vram_gb} GB")
    print(f"  cuDNN       : {torch.backends.cudnn.version()}")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    errors.append("CUDA is NOT available — training will fail")

# === 2. Dependencies ===
print("\n--- 2. Key Dependencies ---")
import importlib.metadata
deps = ["coqui-tts", "coqui-tts-trainer", "transformers", "numpy", "scipy",
        "librosa", "soundfile", "pyloudnorm", "tensorboard", "protobuf"]
for pkg in deps:
    try:
        ver = importlib.metadata.version(pkg)
        print(f"  {pkg:25s} {ver}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {pkg:25s} *** MISSING ***")
        errors.append(f"Missing package: {pkg}")

# === 3. TTS imports ===
print("\n--- 3. Coqui TTS Imports ---")
try:
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.configs.shared_configs import CharactersConfig, BaseDatasetConfig
    from TTS.config.shared_configs import BaseAudioConfig
    from TTS.tts.models.vits import Vits
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs
    print("  All imports successful")
except Exception as e:
    print(f"  IMPORT ERROR: {e}")
    errors.append(f"Import error: {e}")

# === 4. Tokenizer ===
print("\n--- 4. Tokenizer Validation ---")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
try:
    from vits_tokenizer import BASELINE_PHONEMES, CLUSTER_TOKENS, build_vocab, patch_tokenizer
    bv = build_vocab(BASELINE_PHONEMES)
    cv = build_vocab(CLUSTER_TOKENS)
    print(f"  Baseline vocab : {len(bv)} tokens ({len(BASELINE_PHONEMES)} phonemes + 5 special)")
    print(f"  Clustered vocab: {len(cv)} tokens ({len(CLUSTER_TOKENS)} clusters + 5 special)")

    # Quick tokenization test
    class DummyTok:
        characters = None
    tok = patch_tokenizer(DummyTok(), bv)
    ids = tok.text_to_ids("b aa r ax t")
    assert len(ids) == 5, f"Expected 5 IDs, got {len(ids)}"
    print(f"  Baseline test  : 'b aa r ax t' -> {ids} [OK]")

    tok2 = patch_tokenizer(DummyTok(), cv)
    ids2 = tok2.text_to_ids("C0 C10 C38")
    assert len(ids2) == 3, f"Expected 3 IDs, got {len(ids2)}"
    print(f"  Cluster test   : 'C0 C10 C38' -> {ids2} [OK]")
except Exception as e:
    print(f"  TOKENIZER ERROR: {e}")
    errors.append(f"Tokenizer error: {e}")

# === 5. Data Files ===
print("\n--- 5. Data Files ---")
DATA_DIR = "/mnt/d/tts_data/hi"

# Check processed WAVs
proc_dir = os.path.join(DATA_DIR, "processed")
if os.path.isdir(proc_dir):
    wavs = [f for f in os.listdir(proc_dir) if f.endswith(".wav")]
    print(f"  Processed WAVs : {len(wavs)}")
    if len(wavs) < 900:
        errors.append(f"Only {len(wavs)} processed WAVs found (expected ~927)")
else:
    errors.append("processed/ directory missing")

# Check metadata files
for split in ["train", "val", "test"]:
    for kind in ["metadata_baseline.csv", "metadata_clustered.csv"]:
        p = os.path.join(DATA_DIR, split, kind)
        if os.path.exists(p):
            with open(p) as f:
                lines = [l for l in f if l.strip()]
            print(f"  {split}/{kind:30s} {len(lines)} lines")
        else:
            print(f"  {split}/{kind:30s} *** MISSING ***")
            errors.append(f"Missing {split}/{kind}")

# Check that WAV files referenced in metadata actually exist
print("\n--- 6. WAV File Cross-Check ---")
missing_wavs = 0
for split in ["train", "val", "test"]:
    p = os.path.join(DATA_DIR, split, "metadata_baseline.csv")
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem = line.split("|")[0].strip()
            wav = os.path.join(DATA_DIR, "processed", stem + ".wav")
            if not os.path.exists(wav):
                missing_wavs += 1
                if missing_wavs <= 5:
                    print(f"  MISSING: {wav}")
if missing_wavs == 0:
    print(f"  All referenced WAVs exist [OK]")
else:
    errors.append(f"{missing_wavs} WAV files referenced in metadata are missing")
    print(f"  {missing_wavs} WAVs missing!")

# === 7. Output directory ===
print("\n--- 7. Output Directory ---")
MODELS_DIR = "/mnt/d/tts_models/hi"
os.makedirs(MODELS_DIR, exist_ok=True)
test_file = os.path.join(MODELS_DIR, ".write_test")
try:
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print(f"  {MODELS_DIR} writable [OK]")
except Exception as e:
    print(f"  WRITE ERROR: {e}")
    errors.append(f"Cannot write to {MODELS_DIR}")

# === 8. protobuf version check ===
print("\n--- 8. Protobuf Compatibility ---")
try:
    pb_ver = importlib.metadata.version("protobuf")
    major = int(pb_ver.split(".")[0])
    if major >= 5:
        print(f"  protobuf {pb_ver} — may cause 'GetPrototype' errors with older TensorFlow")
        print("  (These are warnings only and do NOT affect PyTorch VITS training)")
    else:
        print(f"  protobuf {pb_ver} [OK]")
except Exception:
    pass

# === SUMMARY ===
print("\n" + "=" * 60)
if errors:
    print(f"VALIDATION FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED ✓")
    print("Ready to start training.")
    sys.exit(0)
