# -*- coding: utf-8 -*-
"""
Week 3 — Task 3: End-to-End Test
=================================
Full pipeline per language:
  [Sample sentence]
       ↓
  [Your trained G2P model]  (loads weights from week 2)
       ↓
  [Post-processing: schwa deletion + stress]  (Task 2 rules)
       ↓
  [Parler-TTS: generate audio]
       ↓
  [Save .wav file + print phoneme analysis]

Sample sentences:
  Hindi   : "नमस्ते, आप कैसे हैं?"
  Gujarati: "તમે કેમ છો?"
  Marathi : "तुम्ही कसे आहात?"

Run:
  python week3_task3_e2e_test.py

Outputs:
  hindi_output.wav
  gujarati_output.wav
  marathi_output.wav
  e2e_report.txt
"""

import os
import sys
import json
import time
import tensorflow as tf
import numpy as np

# ── Import post-processing from Task 2 ────────────────────────────────────────
# (Task 2 file must be in the same folder)
try:
    from week3_task2_schwa_deletion import postprocess, G2P_TO_IPA, g2p_to_ipa
    POSTPROCESS_AVAILABLE = True
except ImportError:
    print("⚠️  week3_task2_schwa_deletion.py not found in this folder.")
    print("   Schwa deletion / stress will be skipped.")
    POSTPROCESS_AVAILABLE = False

    def postprocess(phonemes, lang):
        return phonemes

    def g2p_to_ipa(phonemes):
        return " ".join(phonemes)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD YOUR WEEK 2 G2P MODEL
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_PATH  = "best_g2p_transformer.weights.h5"
SRC_TOK_PATH  = "src_tokenizer.json"
TGT_TOK_PATH  = "tgt_tokenizer.json"

# Transformer hyperparameters — must match Week 2 training
NUM_LAYERS   = 3
D_MODEL      = 128
NUM_HEADS    = 4
DFF          = 512
DROPOUT_RATE = 0.1
MAX_SRC_LEN  = 200
MAX_TGT_LEN  = 200


class Tokenizer:
    """Minimal tokenizer (same as week 2 — needed to load saved JSON)."""
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.s2i = {self.pad_token: 0, self.unk_token: 1,
                    self.sos_token: 2, self.eos_token: 3}
        self.i2s = {0: self.pad_token, 1: self.unk_token,
                    2: self.sos_token, 3: self.eos_token}
        self.vocab_size = 4

    def encode(self, sentence, add_special=True):
        encoded = [self.s2i.get(tok, self.s2i[self.unk_token])
                   for tok in sentence]
        if add_special:
            encoded = ([self.s2i[self.sos_token]]
                       + encoded
                       + [self.s2i[self.eos_token]])
        return encoded

    def decode(self, indices, remove_special=True):
        special = {self.pad_token, self.sos_token, self.eos_token}
        result = []
        for idx in indices:
            tok = self.i2s.get(idx, self.unk_token)
            if remove_special and tok in special:
                continue
            result.append(tok)
        return result

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.s2i = data["s2i"]
        tok.i2s = {int(k): v for k, v in data["i2s"].items()}
        tok.vocab_size = data["vocab_size"]
        return tok


def build_transformer_model(src_vocab, tgt_vocab):
    """Re-create the same architecture as week 2."""
    from tensorflow.keras import layers

    def get_positional_encoding(max_len, d_model):
        positions = np.arange(max_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

    def create_padding_mask(seq):
        return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(size):
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    def create_masks(src, tgt):
        enc_pm = create_padding_mask(src)
        dec_pm = create_padding_mask(src)
        lah    = create_look_ahead_mask(tf.shape(tgt)[1])
        dec_tgt_pm = create_padding_mask(tgt)
        lah    = tf.maximum(lah, dec_tgt_pm)
        return enc_pm, lah, dec_pm

    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, **kw):
            super().__init__(**kw)
            self.num_heads = num_heads
            self.d_model   = d_model
            self.depth     = d_model // num_heads
            self.wq = layers.Dense(d_model)
            self.wk = layers.Dense(d_model)
            self.wv = layers.Dense(d_model)
            self.dense = layers.Dense(d_model)

        def split_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, inputs, **kw):
            q, k, v, mask = inputs
            batch_size = tf.shape(q)[0]
            q = self.split_heads(self.wq(q), batch_size)
            k = self.split_heads(self.wk(k), batch_size)
            v = self.split_heads(self.wv(v), batch_size)
            scale    = tf.math.sqrt(tf.cast(self.depth, tf.float32))
            scores   = tf.matmul(q, k, transpose_b=True) / scale
            if mask is not None:
                scores += (mask * -1e9)
            weights  = tf.nn.softmax(scores, axis=-1)
            output   = tf.matmul(weights, v)
            output   = tf.transpose(output, perm=[0, 2, 1, 3])
            output   = tf.reshape(output, (batch_size, -1, self.d_model))
            return self.dense(output)

    class EncoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, dropout_rate, **kw):
            super().__init__(**kw)
            self.mha  = MultiHeadAttention(d_model, num_heads)
            self.ffn  = tf.keras.Sequential([
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model)
            ])
            self.ln1  = layers.LayerNormalization(epsilon=1e-6)
            self.ln2  = layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = layers.Dropout(dropout_rate)
            self.drop2 = layers.Dropout(dropout_rate)

        def call(self, inputs, training=False):
            x, mask = inputs
            attn = self.mha((x, x, x, mask), training=training)
            x    = self.ln1(x + self.drop1(attn, training=training))
            ffn  = self.ffn(x)
            return self.ln2(x + self.drop2(ffn, training=training))

    class DecoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, dropout_rate, **kw):
            super().__init__(**kw)
            self.mha1 = MultiHeadAttention(d_model, num_heads)
            self.mha2 = MultiHeadAttention(d_model, num_heads)
            self.ffn  = tf.keras.Sequential([
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model)
            ])
            self.ln1  = layers.LayerNormalization(epsilon=1e-6)
            self.ln2  = layers.LayerNormalization(epsilon=1e-6)
            self.ln3  = layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = layers.Dropout(dropout_rate)
            self.drop2 = layers.Dropout(dropout_rate)
            self.drop3 = layers.Dropout(dropout_rate)

        def call(self, inputs, training=False):
            x, enc_out, lah_mask, pad_mask = inputs
            a1 = self.mha1((x, x, x, lah_mask), training=training)
            x  = self.ln1(x + self.drop1(a1, training=training))
            a2 = self.mha2((x, enc_out, enc_out, pad_mask), training=training)
            x  = self.ln2(x + self.drop2(a2, training=training))
            f  = self.ffn(x)
            return self.ln3(x + self.drop3(f, training=training))

    class Encoder(tf.keras.layers.Layer):
        def __init__(self, n, d, h, dff, vocab, pe, dr, **kw):
            super().__init__(**kw)
            self.d_model   = d
            self.embedding = layers.Embedding(vocab, d)
            self.pos_enc   = get_positional_encoding(pe, d)
            self.enc_layers = [EncoderLayer(d, h, dff, dr) for _ in range(n)]
            self.dropout   = layers.Dropout(dr)

        def call(self, inputs, training=False):
            x, mask = inputs
            x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x = x + self.pos_enc[:, :tf.shape(x)[1], :]
            x = self.dropout(x, training=training)
            for layer in self.enc_layers:
                x = layer((x, mask), training=training)
            return x

    class Decoder(tf.keras.layers.Layer):
        def __init__(self, n, d, h, dff, vocab, pe, dr, **kw):
            super().__init__(**kw)
            self.d_model   = d
            self.embedding = layers.Embedding(vocab, d)
            self.pos_enc   = get_positional_encoding(pe, d)
            self.dec_layers = [DecoderLayer(d, h, dff, dr) for _ in range(n)]
            self.dropout   = layers.Dropout(dr)

        def call(self, inputs, training=False):
            x, enc_out, lah_mask, pad_mask = inputs
            x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x = x + self.pos_enc[:, :tf.shape(x)[1], :]
            x = self.dropout(x, training=training)
            for layer in self.dec_layers:
                x = layer((x, enc_out, lah_mask, pad_mask), training=training)
            return x

    class Transformer(tf.keras.Model):
        def __init__(self, **kw):
            super().__init__()
            self._create_masks = create_masks
            self.encoder = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
                                   src_vocab, MAX_SRC_LEN, DROPOUT_RATE)
            self.decoder = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
                                   tgt_vocab, MAX_TGT_LEN, DROPOUT_RATE)
            self.final   = layers.Dense(tgt_vocab)

        def call(self, inputs, training=False):
            src, tgt = inputs
            enc_pm, lah, dec_pm = create_masks(src, tgt)
            enc_out = self.encoder((src, enc_pm), training=training)
            dec_out = self.decoder((tgt, enc_out, lah, dec_pm), training=training)
            return self.final(dec_out)

    return Transformer()


def greedy_decode(model, src, tgt_tokenizer, max_len=50):
    sos_id = tgt_tokenizer.s2i[tgt_tokenizer.sos_token]
    eos_id = tgt_tokenizer.s2i[tgt_tokenizer.eos_token]
    dec    = tf.expand_dims([sos_id], 0)
    for _ in range(max_len):
        preds    = model((src, dec), training=False)
        next_tok = tf.argmax(preds[:, -1, :], axis=-1)
        next_tok = tf.expand_dims(next_tok, 0)
        dec      = tf.concat([dec, next_tok], axis=-1)
        if next_tok.numpy()[0][0] == eos_id:
            break
    return dec.numpy()[0].tolist()


def run_g2p(word, lang, model, src_tokenizer, tgt_tokenizer):
    """Run a single word through the G2P model."""
    src_seq  = [f"<{lang}>"] + list(word)
    src_enc  = src_tokenizer.encode(src_seq)
    src_enc  = tf.keras.preprocessing.sequence.pad_sequences(
                   [src_enc], maxlen=MAX_SRC_LEN, padding="post", value=0)
    src_tensor = tf.constant(src_enc)
    ids      = greedy_decode(model, src_tensor, tgt_tokenizer)
    return tgt_tokenizer.decode(ids, remove_special=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PARLER-TTS AUDIO GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_audio_parler(text, description, out_path):
    """
    Generate audio using Parler-TTS and save to out_path.
    Downloads ~1.5GB model on first run (cached after that).
    """
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        import soundfile as sf
        import torch
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("     Run: pip install git+https://github.com/huggingface/parler-tts.git")
        print("          pip install transformers soundfile torch")
        return False

    print(f"  🔊 Generating: {out_path}")
    print(f"     Text       : {text}")
    print(f"     Description: {description}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"     Device     : {device}")

    model_id = "parler-tts/parler-tts-mini-v1"

    print("     Loading Parler-TTS model (downloads ~1.5GB on first run)…")
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt  = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = tts_model.generate(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
        )

    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(out_path, audio_arr, tts_model.config.sampling_rate)
    print(f"  ✅ Saved: {out_path}  ({tts_model.config.sampling_rate} Hz)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TEST SENTENCES PER LANGUAGE
# ─────────────────────────────────────────────────────────────────────────────

TEST_SENTENCES = {
    "HI": [
        {
            "script":      "नमस्ते, आप कैसे हैं?",
            "romanized":   "Namaste, aap kaise hain?",
            "words":       ["नमस्ते", "आप", "कैसे", "हैं"],
            "description": "A female speaker delivers a clear, friendly Hindi sentence at a natural pace.",
            "wav":         "hindi_namaste.wav",
        },
        {
            "script":      "भारत एक सुंदर देश है।",
            "romanized":   "Bharat ek sundar desh hai.",
            "words":       ["भारत", "एक", "सुंदर", "देश", "है"],
            "description": "A male speaker reads a calm Hindi statement in a neutral tone.",
            "wav":         "hindi_bharat.wav",
        },
        {
            "script":      "पानी पिलाओ।",
            "romanized":   "Paani pilao.",
            "words":       ["पानी", "पिलाओ"],
            "description": "A clear female voice gives a short Hindi command.",
            "wav":         "hindi_paani.wav",
        },
    ],
    "GU": [
        {
            "script":      "તમે કેમ છો?",
            "romanized":   "Tame kem chho?",
            "words":       ["તમે", "કેમ", "છો"],
            "description": "A female speaker delivers a friendly Gujarati greeting clearly.",
            "wav":         "gujarati_tame.wav",
        },
        {
            "script":      "ગુજરાત સારી જગ્યા છે।",
            "romanized":   "Gujarat saari jagya chhe.",
            "words":       ["ગુજરાત", "સારી", "જગ્યા", "છે"],
            "description": "A calm male voice reads a simple Gujarati sentence.",
            "wav":         "gujarati_gujarat.wav",
        },
    ],
    "MR": [
        {
            "script":      "तुम्ही कसे आहात?",
            "romanized":   "Tumhi kase aahat?",
            "words":       ["तुम्ही", "कसे", "आहात"],
            "description": "A female speaker asks a polite question in Marathi.",
            "wav":         "marathi_tumhi.wav",
        },
        {
            "script":      "महाराष्ट्र माझी माय आहे।",
            "romanized":   "Maharashtra maajhi maay aahe.",
            "words":       ["महाराष्ट्र", "माझी", "माय", "आहे"],
            "description": "A proud male voice recites a Marathi phrase clearly.",
            "wav":         "marathi_maharashtra.wav",
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MAIN E2E PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("WEEK 3 — TASK 3: END-TO-END PIPELINE TEST")
    print("=" * 70)

    report_lines = ["WEEK 3 END-TO-END REPORT", "=" * 70, ""]

    # ── 4a. Load G2P model ────────────────────────────────────────────────────
    g2p_available = False

    if (os.path.exists(WEIGHTS_PATH)
            and os.path.exists(SRC_TOK_PATH)
            and os.path.exists(TGT_TOK_PATH)):

        print("\n📦 Loading G2P model from Week 2 weights…")
        src_tok = Tokenizer.load(SRC_TOK_PATH)
        tgt_tok = Tokenizer.load(TGT_TOK_PATH)
        g2p_model = build_transformer_model(src_tok.vocab_size, tgt_tok.vocab_size)

        # Warm up the model so weights can be loaded
        dummy_src = tf.zeros((1, 10), dtype=tf.int32)
        dummy_tgt = tf.zeros((1,  5), dtype=tf.int32)
        _ = g2p_model((dummy_src, dummy_tgt), training=False)

        g2p_model.load_weights(WEIGHTS_PATH)
        print("  ✅ G2P model loaded.")
        g2p_available = True
    else:
        missing = [p for p in [WEIGHTS_PATH, SRC_TOK_PATH, TGT_TOK_PATH]
                   if not os.path.exists(p)]
        print(f"\n⚠️  G2P model files not found: {missing}")
        print("   Make sure you've run phase2_baseline_g2p.py first.")
        print("   G2P step will be skipped — TTS will still be tested.\n")

    # ── 4b. Run per-language tests ────────────────────────────────────────────
    lang_names = {"HI": "Hindi", "GU": "Gujarati", "MR": "Marathi"}
    tts_success = 0
    tts_total   = 0

    for lang, sentences in TEST_SENTENCES.items():
        lang_name = lang_names[lang]
        print(f"\n{'─'*70}")
        print(f"🌐 LANGUAGE: {lang_name} ({lang})")
        print(f"{'─'*70}")
        report_lines += [f"\n{'─'*70}", f"LANGUAGE: {lang_name}", f"{'─'*70}"]

        for item in sentences:
            print(f"\n  Sentence : {item['script']}")
            print(f"  Roman    : {item['romanized']}")
            report_lines += [f"\nSentence : {item['script']}", f"Roman    : {item['romanized']}"]

            # ── Step 1: G2P per word ──────────────────────────────────────────
            all_phonemes = []
            print(f"\n  STEP 1 — G2P phoneme prediction:")

            for word in item["words"]:
                if g2p_available:
                    try:
                        raw_ph = run_g2p(word, lang, g2p_model, src_tok, tgt_tok)
                    except Exception as e:
                        raw_ph = ["?"]
                        print(f"    G2P error for '{word}': {e}")
                else:
                    raw_ph = ["[G2P not available]"]

                print(f"    {word:<15} → {' '.join(raw_ph)}")
                all_phonemes.extend(raw_ph)

            report_lines.append(f"  Raw G2P  : {' '.join(all_phonemes)}")

            # ── Step 2: Post-processing ───────────────────────────────────────
            print(f"\n  STEP 2 — Post-processing (schwa deletion + stress):")
            print(f"    Before : {' '.join(all_phonemes)}")

            processed = postprocess(all_phonemes, lang)
            print(f"    After  : {' '.join(processed)}")
            report_lines.append(f"  Processed: {' '.join(processed)}")

            # ── Step 3: IPA conversion ────────────────────────────────────────
            if POSTPROCESS_AVAILABLE:
                from week3_task2_schwa_deletion import g2p_to_ipa
                ipa = g2p_to_ipa([p for p in processed if p != "ˈ"])
                print(f"    IPA    : [{ipa}]")
                report_lines.append(f"  IPA      : [{ipa}]")

            # ── Step 4: TTS audio generation ─────────────────────────────────
            print(f"\n  STEP 3 — Parler-TTS audio generation:")
            tts_total += 1
            success = generate_audio_parler(
                text=item["script"],
                description=item["description"],
                out_path=item["wav"],
            )
            if success:
                tts_success += 1
                report_lines.append(f"  Audio    : {item['wav']} ✅")
            else:
                report_lines.append(f"  Audio    : FAILED ❌")

            print()

    # ── 4c. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 E2E TEST SUMMARY")
    print("=" * 70)
    print(f"  G2P model        : {'✅ Loaded' if g2p_available else '❌ Not found'}")
    print(f"  Post-processing  : {'✅ Applied' if POSTPROCESS_AVAILABLE else '❌ Not applied'}")
    print(f"  TTS audio files  : {tts_success}/{tts_total} generated")
    print(f"  Languages tested : {', '.join(lang_names[l] for l in TEST_SENTENCES.keys())}")

    wav_files = [item["wav"] for sentences in TEST_SENTENCES.values()
                 for item in sentences]
    existing_wavs = [f for f in wav_files if os.path.exists(f)]
    print(f"\n  Output .wav files:")
    for wf in wav_files:
        status = "✅" if os.path.exists(wf) else "❌"
        print(f"    {status}  {wf}")

    # ── 4d. Save report ───────────────────────────────────────────────────────
    report_lines += [
        "",
        "=" * 70,
        "SUMMARY",
        f"G2P model       : {'Loaded' if g2p_available else 'Not found'}",
        f"Post-processing : {'Applied' if POSTPROCESS_AVAILABLE else 'Skipped'}",
        f"TTS success     : {tts_success}/{tts_total}",
    ]

    with open("e2e_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n  Full report saved: e2e_report.txt")
    print("\n" + "=" * 70)
    print("Week 3 complete! 🎉")
    print("=" * 70)
