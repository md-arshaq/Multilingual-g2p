#!/usr/bin/env python3
"""
In-Memory GPU Synthesizer & G2P Engine for Live TTS Playground.

Loads and caches VITS checkpoints on GPU for low-latency (< 300 ms)
synthesis across Hindi, Marathi, and Gujarati.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 1. Import TTS & PyTorch BEFORE any local sys.path modifications
import torch
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.models.vits import Vits

# 2. Add repository root and tts/ directory to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
TTS_DIR = PROJECT_ROOT / "tts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TTS_DIR))

from tts_g2p_labeling import (
    load_hindi_g2p_dict,
    load_marathi_g2p_dict,
    load_gujarati_g2p_dict,
    load_phoneme_vocab,
    load_cluster_mapping,
    text_to_baseline_tokens,
    baseline_to_clustered_tokens,
    normalize_text,
)
from vits_tokenizer import (
    BASELINE_PHONEMES, CLUSTER_TOKENS,
    build_vocab, patch_tokenizer, assert_tokenizer_round_trip
)

# Output directory for dynamic synthesis
GENERATED_DIR = PROJECT_ROOT / "data" / "generated_audio"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Model Checkpoints on disk
CHECKPOINTS = {
    "hi": {
        "baseline": "/mnt/d/tts_models/hi/baseline/vits_hindi_female_baseline-August-31-2026_11+25PM-fb53263/best_model.pth",
        "clustered": "/mnt/d/tts_models/hi/clustered/vits_hindi_female_clustered-September-01-2026_08+04AM-fb53263/best_model.pth",
    },
    "mr": {
        "baseline": "/mnt/d/tts_models/mr/baseline/vits_marathi_female_baseline-September-01-2026_06+29PM-fb53263/best_model.pth",
        "clustered": "/mnt/d/tts_models/mr/clustered/vits_marathi_female_clustered-September-02-2026_01+32AM-fb53263/best_model.pth",
    },
    "gu": {
        "baseline": "/mnt/d/tts_models/gu/baseline/vits_gujarati_female_baseline-September-02-2026_10+33AM-fb53263/best_model.pth",
        "clustered": "/mnt/d/tts_models/gu/clustered/vits_gujarati_female_clustered-September-02-2026_03+42PM-fb53263/best_model.pth",
    },
}

# Curated Presets Library
PRESETS = {
    "hi": [
        {"category": "Conversation", "text": "नमस्ते, आप आज कैसे हैं? क्या सब कुछ ठीक है?"},
        {"category": "News", "text": "भारत ने अंतरिक्ष विज्ञान और तकनीकी क्षेत्र में महत्वपूर्ण प्रगति हासिल की है।"},
        {"category": "Proverb", "text": "जहाँ चाह, वहाँ राह होती है।"},
        {"category": "Tongue Twister", "text": "चंदू के चाचा ने, चंदू की चाची को, चाँदनी रात में, चाँदी की चम्मच से चटनी चटाई।"},
        {"category": "Literature", "text": "जीवन में आगे बढ़ने के लिए निरंतर परिश्रम और धैर्य बहुत आवश्यक है।"}
    ],
    "mr": [
        {"category": "Conversation", "text": "नमस्कार, तुम्ही कसे आहात? आजचा दिवस छान जावो."},
        {"category": "News", "text": "महाराष्ट्रात आधुनिक तंत्रज्ञानाचा वापर करून जलसंधारणाची नवी कामे सुरू झाली आहेत."},
        {"category": "Proverb", "text": "प्रयत्ने वाळूचे कण रगडीता तेलही गळे."},
        {"category": "Tongue Twister", "text": "कावळ्याने काकडी खाल्ली आणि झाडावर जाऊन बसला."},
        {"category": "Literature", "text": "मनुष्याचे विचारच त्याच्या जीवनाची दिशा आणि ध्येय ठरवतात."}
    ],
    "gu": [
        {"category": "Conversation", "text": "નમસ્તે, તમે કેમ છો? તમારો આજનો દિવસ સારો રહે."},
        {"category": "News", "text": "ગુજરાતમાં નવી સૌર ઊર્જા પરિયોજનાઓનું સફળતાપૂર્વક નિર્માણ થઈ રહ્યું છે."},
        {"category": "Proverb", "text": "સિદ્ધિ તેને જઈ વરે, જે પરસેવે ન્હાય."},
        {"category": "Tongue Twister", "text": "કાચા પાપડ પાકા પાપડ, પાકા પાપડ કાચા પાપડ."},
        {"category": "Literature", "text": "સત્ય અને અહિંસાના માર્ગ પર ચાલવાથી જીવનમાં શાંતિ મળે છે."}
    ]
}


# ── Optimized VITS inference parameters for natural, human-quality audio ──
# noise_scale:    Controls stochastic variance in the flow decoder.
#                 Lower = cleaner, less robotic buzz.  (default 0.667 → 0.333)
# noise_scale_dp: Controls stochastic duration predictor noise.
#                 Lower = smoother, more consistent phoneme timing. (default 0.8 → 0.333)
# length_scale:   Speaking rate multiplier.
#                 < 1.0 = slightly faster, more natural cadence. (default 1.0 → 0.92)
VITS_NOISE_SCALE = 0.333
VITS_NOISE_SCALE_DP = 0.333
VITS_LENGTH_SCALE = 0.92
VITS_PEAK_HEADROOM = 0.85   # Normalize peak amplitude to prevent digital clipping


class VitsWrapper:
    """Wrapper around VITS model with AudioProcessor for synthesis."""
    def __init__(self, vits_model, audio_processor, tokenizer):
        self.model = vits_model
        self.ap = audio_processor
        self.tokenizer = tokenizer

    def tts_to_file(self, text, file_path):
        import scipy.io.wavfile as wavfile
        with torch.no_grad():
            res = self.model.synthesize(
                text,
                self.model.config,
                noise_scale=VITS_NOISE_SCALE,
                noise_scale_dp=VITS_NOISE_SCALE_DP,
                length_scale=VITS_LENGTH_SCALE,
            )
            wav = np.asarray(res["wav"], dtype=np.float32)
            sr = self.ap.sample_rate if hasattr(self.ap, "sample_rate") else 22050
            # Gentle edge smoothing (5ms cosine ramp) to prevent onset/offset click
            fade_len = min(int(sr * 0.005), len(wav) // 4)
            if fade_len > 0:
                fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
                fade_out = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_len) / fade_len))
                wav[:fade_len] *= fade_in
                wav[-fade_len:] *= fade_out
            # Headroom normalization: peak = 0.85 (-1.4 dBFS) prevents DAC distortion & clipping
            peak = np.max(np.abs(wav))
            if peak > 0:
                wav = wav * (VITS_PEAK_HEADROOM / peak)
            wav_int16 = (wav * 32767.0).astype(np.int16)
            wavfile.write(file_path, sr, wav_int16)


def load_vits_model(model_path, config_path, vocab, smoke_sequence):
    """Load and patch VITS model on GPU."""
    if model_path and os.path.isdir(model_path):
        for root, _, files in os.walk(model_path):
            if "best_model.pth" in files:
                model_path = os.path.join(root, "best_model.pth")
                break

    if config_path is None and model_path:
        dir_path = os.path.dirname(os.path.abspath(model_path))
        candidate = os.path.join(dir_path, "config.json")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            parent_candidate = os.path.join(os.path.dirname(dir_path), "config.json")
            if os.path.exists(parent_candidate):
                config_path = parent_candidate

    cfg = load_config(config_path)
    ap = AudioProcessor.init_from_config(cfg)
    tok, cfg = TTSTokenizer.init_from_config(cfg)
    tok = patch_tokenizer(tok, vocab)

    model = Vits(cfg, ap, tok, speaker_manager=None)
    model.load_checkpoint(cfg, model_path, eval=True)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    ids = assert_tokenizer_round_trip(tok, smoke_sequence)
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"  [Model Loaded] {os.path.basename(model_path)} on {device_name} | Restored: {smoke_sequence!r} -> {ids}")
    return VitsWrapper(model, ap, tok)


class ModelManager:
    """Manages lazy-loading and in-memory GPU caching of VITS models."""
    def __init__(self):
        self.models = {}  # key: (lang, model_type) -> loaded_model
        self.vocab_baseline = build_vocab(BASELINE_PHONEMES)
        self.vocab_clustered = build_vocab(CLUSTER_TOKENS)
        
        # Load G2P dictionaries & cluster mappings
        print("  Initializing G2P Resources...")
        self.dict_hi = load_hindi_g2p_dict()
        self.dict_mr = load_marathi_g2p_dict()
        self.dict_gu = load_gujarati_g2p_dict()
        self.valid_phonemes = load_phoneme_vocab()
        self.cluster_map = load_cluster_mapping()
        print("  G2P Resources Loaded Successfully!")

    def text_to_tokens(self, text, lang):
        """Convert arbitrary input text to baseline phonemes and cluster tokens."""
        text = text.strip()
        if not text:
            return "", "", [], []
            
        if lang == "hi":
            d = self.dict_hi
        elif lang == "mr":
            d = self.dict_mr
        elif lang == "gu":
            d = self.dict_gu
        else:
            raise ValueError(f"Unsupported language: {lang}")

        # Convert to baseline tokens with word boundaries
        b_seq, oovs, ok = text_to_baseline_tokens(text, d, self.valid_phonemes, lang=lang)
        
        # Convert to clustered tokens
        c_seq, unmapped, cl_ok = baseline_to_clustered_tokens(b_seq, self.cluster_map)
        
        b_toks = b_seq.split() if b_seq else []
        c_toks = c_seq.split() if c_seq else []
        
        return b_seq, c_seq, b_toks, c_toks

    def get_model(self, lang, model_type):
        """Retrieve or lazily load model into GPU memory."""
        key = (lang, model_type)
        if key in self.models:
            return self.models[key]
            
        ckpt_path = CHECKPOINTS.get(lang, {}).get(model_type)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for {lang} ({model_type}): {ckpt_path}")
            
        vocab = self.vocab_baseline if model_type == "baseline" else self.vocab_clustered
        smoke_seq = "b aa r ax t" if model_type == "baseline" else "C0 C10 C38"
        
        print(f"  [GPU Cache] Loading {lang.upper()} ({model_type}) into GPU VRAM...")
        model = load_vits_model(ckpt_path, None, vocab, smoke_seq)
        self.models[key] = model
        print(f"  [GPU Cache] {lang.upper()} ({model_type}) ready for live synthesis!")
        return model

    def synthesize(self, text, lang, model_type="both"):
        """
        Synthesize audio for input text.
        
        Returns dict containing:
        - baseline: {success, filename, audio_url, latency_ms, duration_sec, rtf, error}
        - clustered: {success, filename, audio_url, latency_ms, duration_sec, rtf, error}
        - g2p: {baseline_seq, clustered_seq, baseline_tokens, clustered_tokens, reduction_pct}
        """
        b_seq, c_seq, b_toks, c_toks = self.text_to_tokens(text, lang)
        
        if not b_seq or len(b_toks) == 0:
            raise ValueError(f"Could not convert input text '{text}' to valid Indic phonemes. Please ensure input text is in Hindi, Marathi, or Gujarati script.")

        reduction_pct = 0.0
        if len(b_toks) > 0:
            reduction_pct = round(((len(b_toks) - len(c_toks)) / len(b_toks)) * 100, 1)

        result = {
            "text": text,
            "lang": lang,
            "g2p": {
                "baseline_seq": b_seq,
                "clustered_seq": c_seq,
                "baseline_tokens": b_toks,
                "clustered_tokens": c_toks,
                "baseline_count": len(b_toks),
                "clustered_count": len(c_toks),
                "reduction_pct": reduction_pct,
            }
        }

        # Unique hash for this text & lang to cache audio output
        text_hash = hashlib.md5(f"{lang}_{text}".encode("utf-8")).hexdigest()[:12]

        # 1. Synthesize Baseline
        if model_type in ["both", "baseline"]:
            try:
                m_base = self.get_model(lang, "baseline")
                fname_base = f"live_{lang}_baseline_{text_hash}.wav"
                out_path_base = GENERATED_DIR / fname_base
                
                t0 = time.perf_counter()
                m_base.tts_to_file(b_seq, str(out_path_base))
                latency_ms = (time.perf_counter() - t0) * 1000
                
                file_size = os.path.getsize(out_path_base)
                duration_sec = max(0.1, (file_size - 44) / (22050.0 * 2))
                rtf = (latency_ms / 1000.0) / duration_sec
                
                result["baseline"] = {
                    "success": True,
                    "filename": fname_base,
                    "audio_url": f"/audio/generated/{fname_base}",
                    "latency_ms": round(latency_ms, 1),
                    "duration_sec": round(duration_sec, 2),
                    "rtf": round(rtf, 3),
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                result["baseline"] = {"success": False, "error": str(e)}

        # 2. Synthesize Clustered
        if model_type in ["both", "clustered"]:
            try:
                m_clust = self.get_model(lang, "clustered")
                fname_clust = f"live_{lang}_clustered_{text_hash}.wav"
                out_path_clust = GENERATED_DIR / fname_clust
                
                t0 = time.perf_counter()
                m_clust.tts_to_file(c_seq, str(out_path_clust))
                latency_ms = (time.perf_counter() - t0) * 1000
                
                file_size = os.path.getsize(out_path_clust)
                duration_sec = max(0.1, (file_size - 44) / (22050.0 * 2))
                rtf = (latency_ms / 1000.0) / duration_sec
                
                result["clustered"] = {
                    "success": True,
                    "filename": fname_clust,
                    "audio_url": f"/audio/generated/{fname_clust}",
                    "latency_ms": round(latency_ms, 1),
                    "duration_sec": round(duration_sec, 2),
                    "rtf": round(rtf, 3),
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                result["clustered"] = {"success": False, "error": str(e)}

        return result


# Global synthesizer singleton
SYNTHESIZER = None

def get_synthesizer():
    global SYNTHESIZER
    if SYNTHESIZER is None:
        SYNTHESIZER = ModelManager()
    return SYNTHESIZER
