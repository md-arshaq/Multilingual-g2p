# TTS Experiment: Hindi Baseline (57-phoneme) vs Clustered (39-cluster) VITS

## Research Question

Does reducing the G2P output vocabulary from 57 phonemes to 39 phonetic clusters
change synthesized-speech quality when training a VITS TTS model from scratch?

## Dataset

- **Source:** [SPRINGLab/IndicTTS-Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi)
- **License:** Derived from the Indic TTS Database (IIT Madras). Users must accept
  the HuggingFace license and comply with the original IndicTTS terms.
- **Original format:** WAV audio at 48,000 Hz with Hindi text transcriptions.

## Selection Criteria

- **Gender:** Female recordings only. No gender mixing.
- **Duration target:** Exactly 1.75 hours of usable speech (acceptable: 1.70–1.80h).
- **Duration filter:** Utterances must be 2.5–20 seconds after validation.
- **Quality filter:** Silent, corrupted, NaN-containing, or empty-text samples are rejected.
- **G2P filter:** Utterances with out-of-vocabulary words (no dictionary match) are excluded.
- **Selection seed:** Random seed 42 for deterministic, reproducible selection.

### Sample Counts

| Metric | Count |
|--------|-------|
| Total in dataset | 11,825 |
| Female samples | 5,842 |
| Valid candidates | 1,645 |
| Excluded (too short/long) | 82 (76 short, 6 long) |
| Excluded (quality checks) | 0 |
| Excluded (OOV words) | 4,115 |
| Selected for experiment | 1,031 |
| Final processed duration | 1.7494 hours |

## Preprocessing

All audio undergoes identical preprocessing before training:

| Setting | Value |
|---------|-------|
| Sample rate | 22,050 Hz |
| Channels | Mono |
| Silence trimming | librosa `effects.trim`, top_db=25 |
| Loudness normalization | pyloudnorm LUFS (-23 LUFS) or RMS fallback |
| Clipping protection | Scaled to < 0.99 amplitude |

The same processed WAV files are used for both baseline and clustered experiments.

## G2P Labeling

- **Dictionary source:** `data/g2p_hi.txt` (15,624 Hindi word→phoneme entries)
  and `data/multilingual_g2p_dataset.txt` (Hindi entries with `<HI>` prefix).
- **Method:** Exact word-level dictionary lookup. No model inference used for OOV.
- **Cluster mapping:** `g2p/phoneme_cluster_mapping.json` maps each of the
  57 phonemes to one of 39 clusters (C0–C38).
- **Word boundaries:** `<wb>` tokens inserted between word-level phoneme sequences.
- **Exclusion rule:** Any utterance containing an out-of-vocabulary word or an
  unmappable phoneme is excluded entirely. No fallback or approximation.

### Baseline Vocabulary (57 phonemes)

```
a aa ae ax b bh c ch d dh dx dxh dxhq dxq ee ei f g gh gq h hq
i ii j jh k kh khq kq l lx m mq n ng nj nx o ou p ph q r rq
s sh sx t th tx txh u uu w y z
```

### Clustered Vocabulary (39 clusters)

```
C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18 C19
C20 C21 C22 C23 C24 C25 C26 C27 C28 C29 C30 C31 C32 C33 C34 C35 C36 C37 C38
```

## Fair-Training Controls

Both VITS models are trained with identical settings. The **only** intentional
difference is the input-token vocabulary and the corresponding metadata file.

| Parameter | Value |
|-----------|-------|
| Architecture | Coqui VITS, single-speaker, from scratch |
| Audio | 22,050 Hz, 80 mel channels, FFT=1024, win=1024, hop=256 |
| Optimizer | AdamW (betas=0.8/0.99, eps=1e-9, weight_decay=0.01) |
| LR schedule | ExponentialLR (gamma=0.999875) |
| Learning rate | 0.0002 |
| Batch size | 32 (reduced identically if T4 OOM) |
| Max updates | 15,000 |
| Validation interval | Every 500 updates |
| Checkpoint interval | Every 500 updates |
| Random seed | 42 |
| GPU | Google Colab T4 (16 GB VRAM) |
| Checkpoint selection | Best validation loss (not listening preference) |
| Data split | 90% train / 5% val / 5% test (seed=42) |
| Input bypass | `use_phonemes=false` — no internal phonemizer |

### Training Log

| Item | Baseline | Clustered |
|------|----------|-----------|
| GPU type | _TBD_ | _TBD_ |
| Software versions | _TBD_ | _TBD_ |
| Training duration | _TBD_ | _TBD_ |
| Best val loss | _TBD_ | _TBD_ |
| Best checkpoint step | _TBD_ | _TBD_ |
| Batch size (actual) | _TBD_ | _TBD_ |

## Evaluation Protocol

### Automated MOS (Supporting Evidence)

- **Method:** UTMOS/SpeechMOS (preferred) or librosa-based heuristic (fallback).
- **Labeling:** Clearly marked as "automated/predicted MOS" — NOT human MOS.
- **Metrics:** Per-condition mean, std, 95% CI, sample count, per-sentence paired difference.
- **Sets:** Both held-out test (5% of training data) and unseen generalization (20 sentences).

### Human Evaluation (Primary Evidence)

- **Listeners:** 5 native Hindi speakers.
- **Blinding:** Anonymous System-A/System-B labels. Randomized condition order per pair.
- **Scales:** Naturalness (1–5), Intelligibility (1–5), optional pairwise preference.
- **Analysis:** Wilcoxon signed-rank test on paired differences.
- **Significance threshold:** p < 0.05. No significance claim if p >= 0.05.
- **Listener IDs:** Stored anonymously (L1–L5).

## Limitations

> [!WARNING]
> This is a **preliminary pilot study** with significant scope limitations:

1. **One language only:** Hindi. Results may not generalize to Gujarati or Marathi.
2. **One voice:** Female speaker only. Male or mixed-gender synthesis is untested.
3. **Limited data:** 1.75 hours of training audio — well below the ~10h recommended for high-quality VITS.
4. **Small listener pool:** 5 human evaluators — insufficient for strong statistical claims.
5. **Vocabulary mismatch:** The 57→39 reduction is specific to this clustering configuration.
6. **Dictionary-only G2P:** No model inference for OOV words — utterance coverage depends
   on dictionary completeness.
7. **Single training run:** No repeated runs with different seeds for statistical robustness.

> [!CAUTION]
> **The gTTS-generated samples and MOS scores in `samples/` and `results/` directories
> from previous tasks (task5–task9) are NOT evidence for this VITS experiment.**
> They were generated using a fundamentally different synthesis method and must not
> be compared to or combined with VITS results.

## Reproducibility

All scripts, configs, manifests, and split lists are committed to Git.
The following are gitignored: raw/processed audio, model checkpoints,
generated WAV samples, HuggingFace tokens/caches.

`notebooks/tts_vits_training.ipynb` is the executable source of truth for
VITS training parameters. The JSON files in `configs/tts/` are reference
artifacts and must be kept in sync if they are used outside that notebook.

Before training, regenerate the deterministic split with
`python tts/tts_split_data.py`. The splitter groups duplicate normalized text,
so a sentence cannot occur in more than one of train, validation, or test.
Copy `tts/vits_tokenizer.py` to `SAMSUNG-TTS-EXPERIMENT/tts/` in Google Drive;
the Colab notebook loads this shared strict tokenizer so training and inference
use the same atomic phoneme/cluster tokens.

To reproduce:
```bash
# 1. Authenticate with HuggingFace
huggingface-cli login

# 2. Download and select data subset
python tts/tts_data_download.py

# 3. Generate G2P labels
python tts/tts_g2p_labeling.py

# 4. Preprocess audio
python tts/tts_audio_preprocess.py

# 5. Create splits and VITS metadata
python tts/tts_split_data.py

# 6. Verify tokenization
python tts/tts_tokenizer.py --test

# 7. Train on Colab T4 (see notebooks/tts_vits_training.ipynb)

# 8. Generate evaluation audio
python tts/tts_inference.py --baseline_model ... --clustered_model ...

# 9. Run automated evaluation
python tts/tts_automated_eval.py

# 10. Prepare human evaluation package
python tts/tts_human_eval_prep.py

# 11. Analyze human results (after collection)
python tts/tts_human_eval_analysis.py
```
