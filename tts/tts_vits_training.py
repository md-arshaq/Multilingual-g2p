#!/usr/bin/env python3
"""
Hindi VITS Training script (Baseline vs Clustered)
Adapted from notebooks/tts_vits_training.ipynb to run locally inside WSL2.
"""

import sys
import os
import platform
import random
import time
import json
import importlib.metadata
import torch
import numpy as np

# Enable deterministic kernels for controlled comparison
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Add tts/ to sys.path so we can import local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from vits_tokenizer import (
    BASELINE_PHONEMES,
    CLUSTER_TOKENS,
    build_vocab,
    patch_tokenizer,
)

# Token vocabularies
baseline_vocab = build_vocab(BASELINE_PHONEMES)
clustered_vocab = build_vocab(CLUSTER_TOKENS)

print(f"Baseline vocab: {len(baseline_vocab)} tokens")
print(f"Clustered vocab: {len(clustered_vocab)} tokens")

# Custom Formatter
from TTS.tts.datasets import register_formatter

def tts_formatter(root_path, manifest_file, **kwargs):
    """Parse our 2-column pipe-delimited metadata."""
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("|", 1)
            if len(cols) < 2:
                continue
            wav_stem = cols[0].strip()
            text = cols[1].strip()
            wav_path = os.path.join(root_path, "processed", wav_stem + ".wav")
            if not os.path.exists(wav_path):
                continue
            items.append({
                "text": text,
                "audio_file": wav_path,
                "speaker_name": "tts_female",
                "root_path": root_path,
            })
    return items

try:
    register_formatter("custom_tts", tts_formatter)
    print("Custom formatter registered: custom_tts")
except ValueError:
    from TTS.tts.datasets import formatters
    formatters._FORMATTER_REGISTRY["custom_tts"] = tts_formatter
    print("Custom formatter updated: custom_tts")

def get_latest_checkpoint(output_dir):
    """Auto-resume training by finding the latest checkpoint recursively."""
    if not os.path.exists(output_dir):
        return None
    ckpts = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.startswith("checkpoint_") and f.endswith(".pth"):
                ckpts.append(os.path.join(root, f))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: os.path.getmtime(x))
    return ckpts[-1]

def get_best_checkpoint(output_dir):
    """Find best_model.pth recursively or fall back to latest checkpoint."""
    if not os.path.exists(output_dir):
        return None
    for root, _, files in os.walk(output_dir):
        if "best_model.pth" in files:
            return os.path.join(root, "best_model.pth")
    return get_latest_checkpoint(output_dir)

# Shared configuration
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import CharactersConfig, BaseDatasetConfig
from TTS.config.shared_configs import BaseAudioConfig

AUDIO_CFG = BaseAudioConfig(
    sample_rate=22050,
    fft_size=1024,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
    do_trim_silence=False,
)

SEED = 42
MAX_STEPS = 15000
# With batch size 32 and 834 train samples, 1 epoch is ~26 steps.
# 15,000 steps = 576 epochs.
EPOCHS = 576

CHAR_CFG_BASELINE = CharactersConfig(
    characters_class="TTS.tts.utils.text.characters.Graphemes",
    pad="<pad>", eos="<eos>", bos="<bos>", blank="<blnk>",
    characters=" ".join(list(BASELINE_PHONEMES) + ["<wb>"]),
    punctuations=""
)

CHAR_CFG_CLUSTERED = CharactersConfig(
    characters_class="TTS.tts.utils.text.characters.Graphemes",
    pad="<pad>", eos="<eos>", bos="<bos>", blank="<blnk>",
    characters=" ".join(list(CLUSTER_TOKENS) + ["<wb>"]),
    punctuations=""
)

print(f"Audio SR  : {AUDIO_CFG.sample_rate}")
print(f"Target Epochs : {EPOCHS} (~{MAX_STEPS} steps)")
print(f"Seed      : {SEED}")

def run_baseline_training(data_dir, models_dir, lang="hi"):
    from trainer import Trainer, TrainerArgs
    from TTS.tts.models.vits import Vits
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.tts.datasets import load_tts_samples

    lang_names = {"hi": "hindi", "mr": "marathi", "gu": "gujarati"}
    lang_name = lang_names.get(lang, lang)
    print("="*60)
    print(f"TRAINING: BASELINE VITS ({lang_name.upper()}, 57 phonemes)")
    print("="*60)

    baseline_output = f"{models_dir}/baseline"
    os.makedirs(baseline_output, exist_ok=True)

    ds_cfg_b = BaseDatasetConfig(
        formatter="custom_tts",
        dataset_name=f"{lang_name}_female_baseline",
        path=data_dir,
        meta_file_train="train/metadata_baseline.csv",
        meta_file_val="val/metadata_baseline.csv",
        language=lang
    )

    cfg_b = VitsConfig(
        run_name=f"vits_{lang_name}_female_baseline",
        output_path=baseline_output,
        audio=AUDIO_CFG,
        batch_size=32,
        eval_batch_size=32,
        mixed_precision=True,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        print_step=50,
        save_step=500,
        save_n_checkpoints=3,
        save_best_after=500,
        run_eval=False,
        test_delay_epochs=99999,
        lr_gen=0.0002,
        lr_disc=0.0002,
        optimizer="AdamW",
        optimizer_params={"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01},
        lr_scheduler="ExponentialLR",
        lr_scheduler_params={"gamma": 0.999875},
        cudnn_benchmark=False,
        epochs=EPOCHS,
        datasets=[ds_cfg_b],
        use_phonemes=False,
        phoneme_language=None,
        phonemizer=None,
        text_cleaner=None,
        characters=CHAR_CFG_BASELINE,
        test_sentences=[],
    )

    ap_b = AudioProcessor.init_from_config(cfg_b)
    tok_b, cfg_b = TTSTokenizer.init_from_config(cfg_b)
    tok_b = patch_tokenizer(tok_b, baseline_vocab)

    train_b, eval_b = load_tts_samples(
        ds_cfg_b, eval_split=True,
        eval_split_size=0.05
    )
    print(f"Train samples: {len(train_b)}, Eval samples: {len(eval_b)}")

    seed_everything(SEED)
    model_b = Vits(cfg_b, ap_b, tok_b, speaker_manager=None)

    ckpt_b = get_latest_checkpoint(baseline_output)
    if ckpt_b:
        print(f"Resuming Baseline training from checkpoint: {ckpt_b}")

    t0 = time.time()
    trainer_b = Trainer(
        TrainerArgs(restore_path=ckpt_b, skip_train_epoch=False),
        cfg_b,
        output_path=baseline_output,
        model=model_b,
        train_samples=train_b,
        eval_samples=eval_b,
    )
    trainer_b.fit()
    baseline_duration = time.time() - t0
    baseline_steps = getattr(trainer_b, "total_steps", None)
    baseline_best_checkpoint = get_best_checkpoint(baseline_output)
    return baseline_duration, baseline_steps, baseline_best_checkpoint


def run_clustered_training(data_dir, models_dir, lang="hi"):
    from trainer import Trainer, TrainerArgs
    from TTS.tts.models.vits import Vits
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.tts.datasets import load_tts_samples

    lang_names = {"hi": "hindi", "mr": "marathi", "gu": "gujarati"}
    lang_name = lang_names.get(lang, lang)
    print("="*60)
    print(f"TRAINING: CLUSTERED VITS ({lang_name.upper()}, 39 clusters)")
    print("="*60)

    clustered_output = f"{models_dir}/clustered"
    os.makedirs(clustered_output, exist_ok=True)

    ds_cfg_c = BaseDatasetConfig(
        formatter="custom_tts",
        dataset_name=f"{lang_name}_female_clustered",
        path=data_dir,
        meta_file_train="train/metadata_clustered.csv",
        meta_file_val="val/metadata_clustered.csv",
        language=lang
    )

    cfg_c = VitsConfig(
        run_name=f"vits_{lang_name}_female_clustered",
        output_path=clustered_output,
        audio=AUDIO_CFG,
        batch_size=32,
        eval_batch_size=32,
        mixed_precision=True,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        print_step=50,
        save_step=500,
        save_n_checkpoints=3,
        save_best_after=500,
        run_eval=False,
        test_delay_epochs=99999,
        lr_gen=0.0002,
        lr_disc=0.0002,
        optimizer="AdamW",
        optimizer_params={"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01},
        lr_scheduler="ExponentialLR",
        lr_scheduler_params={"gamma": 0.999875},
        cudnn_benchmark=False,
        epochs=EPOCHS,
        datasets=[ds_cfg_c],
        use_phonemes=False,
        phoneme_language=None,
        phonemizer=None,
        text_cleaner=None,
        characters=CHAR_CFG_CLUSTERED,
        test_sentences=[],
    )

    ap_c = AudioProcessor.init_from_config(cfg_c)
    tok_c, cfg_c = TTSTokenizer.init_from_config(cfg_c)
    tok_c = patch_tokenizer(tok_c, clustered_vocab)

    train_c, eval_c = load_tts_samples(
        ds_cfg_c, eval_split=True,
        eval_split_size=0.05
    )
    print(f"Train samples: {len(train_c)}, Eval samples: {len(eval_c)}")

    seed_everything(SEED)
    model_c = Vits(cfg_c, ap_c, tok_c, speaker_manager=None)

    ckpt_c = get_latest_checkpoint(clustered_output)
    if ckpt_c:
        print(f"Resuming Clustered training from checkpoint: {ckpt_c}")

    t0 = time.time()
    trainer_c = Trainer(
        TrainerArgs(restore_path=ckpt_c, skip_train_epoch=False),
        cfg_c,
        output_path=clustered_output,
        model=model_c,
        train_samples=train_c,
        eval_samples=eval_c,
    )
    trainer_c.fit()
    clustered_duration = time.time() - t0
    clustered_steps = getattr(trainer_c, "total_steps", None)
    clustered_best_checkpoint = get_best_checkpoint(clustered_output)
    return clustered_duration, clustered_steps, clustered_best_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VITS baseline and/or clustered models.")
    parser.add_argument("--lang", type=str, choices=["hi", "mr", "gu"], default="hi",
                        help="Language code ('hi', 'mr', or 'gu', default: 'hi')")
    parser.add_argument("--model", type=str, choices=["baseline", "clustered", "both"], default="both",
                        help="Which model to train (default: both)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to TTS data root (defaults to /mnt/d/tts_data/{lang})")
    parser.add_argument("--models_dir", type=str, default=None, help="Path to save models (defaults to /mnt/d/tts_models/{lang})")
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else f"/mnt/d/tts_data/{args.lang}"
    models_dir = args.models_dir if args.models_dir is not None else f"/mnt/d/tts_models/{args.lang}"
    os.makedirs(models_dir, exist_ok=True)

    baseline_duration = None
    baseline_steps = None
    baseline_best_checkpoint = None

    clustered_duration = None
    clustered_steps = None
    clustered_best_checkpoint = None

    if args.model in ["baseline", "both"]:
        baseline_duration, baseline_steps, baseline_best_checkpoint = run_baseline_training(data_dir, models_dir, args.lang)

    if args.model in ["clustered", "both"]:
        clustered_duration, clustered_steps, clustered_best_checkpoint = run_clustered_training(data_dir, models_dir, args.lang)

    log_path = f"{models_dir}/training_log.json"
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log = json.load(f)
        except Exception:
            log = {}
    else:
        log = {}

    log.update({
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "coqui_tts_version": importlib.metadata.version("coqui-tts"),
        "transformers_version": importlib.metadata.version("transformers"),
        "random_seed": SEED,
        "max_steps": MAX_STEPS,
        "batch_size": 32,
        "lr_gen": 0.0002,
        "lr_disc": 0.0002,
        "audio_config": {
            "sample_rate": 22050, "fft_size": 1024, "win_length": 1024, "hop_length": 256, "num_mels": 80,
        },
    })

    if "baseline" not in log:
        log["baseline"] = {"vocab_size": len(baseline_vocab)}
    if "clustered" not in log:
        log["clustered"] = {"vocab_size": len(clustered_vocab)}

    if baseline_duration is not None:
        log["baseline"].update({
            "vocab_size": len(baseline_vocab),
            "training_hours": baseline_duration / 3600,
            "completed_steps": baseline_steps,
            "best_checkpoint": baseline_best_checkpoint,
        })

    if clustered_duration is not None:
        log["clustered"].update({
            "vocab_size": len(clustered_vocab),
            "training_hours": clustered_duration / 3600,
            "completed_steps": clustered_steps,
            "best_checkpoint": clustered_best_checkpoint,
        })

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(json.dumps(log, indent=2, default=str))
    print(f"\nLog saved: {log_path}")
    print("\n*** ALL REQUESTED TRAINING RUNS COMPLETE ***")
