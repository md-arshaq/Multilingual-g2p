```text
You are implementing the TTS phase of an existing Samsung internship research project. Work only inside the repository already provided to you. Do not redesign the research question, retrain the G2P models, replace the selected dataset, or use gTTS as experimental evidence.

Goal
Build a scientifically fair Hindi proof-of-concept that compares speech quality from:
1. A TTS model trained on the existing 57-phoneme vocabulary.
2. A separate TTS model trained on the existing 39-cluster vocabulary.

The experiment must show whether reducing the G2P output vocabulary from 57 phonemes to 39 clusters changes synthesized-speech quality.

Fixed research decisions
- Language: Hindi only for this implementation.
- Dataset: Hugging Face `SPRINGLab/IndicTTS-Hindi`.
- Voice selection: female recordings only. Do not mix genders.
- Dataset size: select exactly 1.75 hours of usable female speech; acceptable range is 1.70–1.80 hours.
- Training platform: Google Colab T4 GPU with 16 GB VRAM.
- TTS architecture: Coqui TTS VITS, single-speaker, trained from scratch.
- Do not implement FastSpeech 2 or any other fallback model.
- Evaluation: automated MOS as supporting evidence plus blinded human evaluation by five native Hindi listeners.
- Never upload raw audio, processed audio, access tokens, caches, or full model checkpoints to GitHub.
- The repository must remain suitable for private collaboration.

Existing project artifacts: treat these as the source of truth
- `g2p/phoneme_vocab.json`: baseline 57-phoneme inventory.
- `g2p/phoneme_cluster_mapping.json`: phoneme-to-cluster mapping; current target is 39 clusters, C0 through C38.
- `models/best_g2p_transformer.weights.h5`: trained baseline G2P model.
- `models/g2p_clustered_model.weights.h5`: trained clustered G2P model.
- `data/multilingual_g2p_dataset.txt`: existing word-to-phoneme data.
- `g2p/task5_tts_data_prep.py`, `g2p/task6_g2p_tts_pipeline.py`, `g2p/task7_mos_evaluation.py`, `g2p/task8_mos_visualization.py`, `g2p/task9_correlation_analysis.py`: existing scripts that may be reused or replaced only when needed.
- Existing `samples/` and current MOS reports were generated using gTTS-style placeholder synthesis. Do not treat them as valid VITS results and do not overwrite them.

Important scientific constraint
The clustered TTS model must consume the 39 cluster tokens directly. It must not convert C0–C38 back into ordinary phonemes before training or inference.

The VITS text-processing path must bypass any internal grapheme-to-phoneme converter. It must tokenize supplied whitespace-separated tokens directly:
- Baseline example: `b aa r ax t`
- Clustered example: `C3 C0 C29 C0 C1`

Do not let `C10` be split into `C`, `1`, and `0`. It is one token.
Preserve word boundaries using one explicit special boundary token, for example `<wb>`.
Use only the inventory tokens plus necessary special tokens such as padding/start/end/boundary. Do not include raw Devanagari characters in either TTS vocabulary.

Dataset acquisition and subset creation
- The user will authenticate to Hugging Face and accept the original IndicTTS licence if required.
- Download/access the Hindi dataset through the Hugging Face `datasets` library, not through manual file downloads.
- Filter to `gender == female`.
- Retain only clean, single-utterance Hindi samples.
- Reject samples that are silent, corrupted, have empty text, have a duration below 2.5 seconds or above 20 seconds, or cannot be converted into a valid complete token sequence.
- Use a deterministic selection procedure with random seed 42 to select valid female samples until total duration is between 1.70 and 1.80 hours, targeting 1.75 hours.
- Save a manifest containing: original dataset row ID, text, gender, original duration, selected/not-selected status, exclusion reason if excluded, baseline token sequence, clustered token sequence, and split assignment.
- Keep raw downloaded data outside Git tracking.

Transcript and G2P-label preparation
For every selected Hindi transcript:
1. Normalize text conservatively. Preserve the spoken Hindi words; remove only punctuation and formatting that are not spoken.
2. Split into words.
3. Obtain baseline phoneme tokens from `data/multilingual_g2p_dataset.txt` when an exact Hindi word match exists.
4. For an out-of-vocabulary word, use the existing trained baseline G2P model only if its current inference implementation can be reproduced correctly from repository artifacts.
5. Derive the clustered sequence by applying `g2p/phoneme_cluster_mapping.json` to the baseline phoneme sequence. Do not use raw text as a fallback label.
6. Exclude an utterance if any word cannot produce a valid baseline phoneme sequence or any baseline phoneme cannot map to a cluster.
7. Record all fallbacks and exclusions in the manifest.

This ensures the baseline and clustered training labels are fully aligned and differ only by the phoneme-to-cluster representation.

Audio preprocessing
- Preserve the original source audio unchanged.
- Create a separate processed-data directory.
- Convert every selected audio file to mono WAV at 22,050 Hz.
- Trim leading and trailing silence consistently.
- Apply consistent loudness normalization.
- Reject processing failures and log them.
- Verify there are no empty audio files, NaNs, clipping, or transcript/audio mismatches introduced by preprocessing.
- Use exactly the same processed WAV files in both baseline and clustered experiments.

Data split
Use one immutable split for both conditions:
- Training: 90%
- Validation: 5%
- Test: 5%
- Random seed: 42

Split at utterance level after final filtering. Save the exact sample-ID lists. No sample may occur in more than one split.

TTS training
Train two completely separate VITS models:

A. Baseline VITS
- Input vocabulary: exactly the existing 57 phoneme tokens plus required special tokens.
- Training metadata: processed WAV path plus baseline token sequence.

B. Clustered VITS
- Input vocabulary: exactly `C0` through `C38` plus required special tokens.
- Training metadata: the same processed WAV path plus clustered token sequence.

Fairness requirements:
- Use identical audio, splits, preprocessing, VITS architecture, optimizer, random seed, batch size, learning-rate schedule, validation interval, checkpoint interval, and maximum training updates.
- The only intentional difference is the input-token vocabulary and metadata.
- Use 22,050 Hz audio, 80 mel channels, FFT size 1024, window size 1024, and hop length 256.
- Use a maximum of 15,000 optimizer updates.
- Run validation and save a checkpoint every 500 updates.
- Choose the final checkpoint for each condition strictly by the best validation loss, not by listening preference.
- Log GPU type, software versions, random seeds, hyperparameters, training duration, best validation loss, and checkpoint path.
- If the T4 runs out of memory, lower batch size only, apply the identical batch size to both conditions, and document the change.
- Do not silently change architecture or one model’s hyperparameters without applying the same change to the other.

Required validation before full training
Before starting both full runs, prove with a small smoke test that:
- A baseline token sequence maps to the intended baseline token IDs.
- A clustered sequence such as `C0 C10 C38` maps to exactly three cluster token IDs.
- No internal phonemizer alters either input sequence.
- The model can complete a forward pass and produce audio.
- The baseline and clustered data loaders contain the same WAV filenames for corresponding splits.

Inference and evaluation data
Create two evaluation sets:

1. Held-out test set
- Use the reserved 5% of selected dataset samples.
- Produce one baseline and one clustered synthesis per held-out text.

2. Unseen generalization set
- Create exactly 20 Hindi sentences not present in training, validation, or held-out test data.
- Use normal, neutral Hindi sentences of varied length.
- Each sentence must be fully tokenizable by the G2P route.
- Produce one baseline and one clustered synthesis per sentence.

For end-to-end unseen synthesis:
- Baseline route: Hindi text → existing baseline G2P → baseline VITS.
- Clustered route: Hindi text → existing clustered G2P → clustered VITS.
- Log every generated G2P output and reject/log invalid output rather than silently substituting text.

Evaluation
Automated evaluation:
- Compute an automated MOS-quality score for every generated audio file.
- Clearly label it as automated/predicted MOS, not human MOS.
- Report per-condition mean, standard deviation, 95% confidence interval, sample count, and per-sentence paired difference.

Human evaluation:
- Prepare a blinded evaluation package for five native Hindi-speaking listeners.
- Randomize audio order and replace model names with anonymous IDs.
- For each audio sample, collect:
  - Naturalness score: 1–5
  - Intelligibility score: 1–5
  - Optional pairwise preference when both conditions are compared
- Do not expose “baseline” or “clustered” to evaluators.
- Randomize which condition appears first in each pair.
- Store listener IDs anonymously.
- Analyze paired results using a Wilcoxon signed-rank test; report p-value and effect direction.
- Do not claim statistical significance if p >= 0.05.

Required output structure
Create a clearly separated TTS experiment area, preferably:

- `data/tts_hindi_female/`
  - manifests and split lists only in Git
  - raw and processed WAV data ignored by Git
- `configs/tts/`
  - baseline and clustered VITS configs
- `models/tts_hindi_female/`
  - ignored checkpoints
- `samples/tts_hindi_female/`
  - ignored generated WAV files
- `results/tts_hindi_female/`
  - small CSV/Markdown reports and plots suitable for Git
- `docs/tts_hindi_female_experiment.md`
  - reproducibility and methodology document

Git and security requirements
- Update `.gitignore` so it excludes:
  - raw/processed audio
  - Hugging Face caches
  - model checkpoints
  - generated WAV/MP3 samples
  - tokens, environment files, and Colab secrets
- Keep code, configs, manifests without copyrighted transcript bulk, result summaries, and documentation in Git.
- Do not commit any Hugging Face token.
- Do not publish the repository; assume it is private Samsung/team collaboration only.

Documentation requirements
Write one concise reproducibility document that states:
- Research question.
- Dataset source and licence notice.
- Female-only, 1.75-hour selection rule.
- Exact retained/excluded sample counts and duration.
- Preprocessing settings.
- G2P fallback/exclusion statistics.
- Baseline and clustered vocabularies.
- Fair-training controls.
- Training hyperparameters and selected checkpoints.
- Evaluation protocol.
- Limitations: one language, one voice, 1.75 hours, five human listeners, and preliminary pilot scope.
- Explicit warning that older gTTS samples/MOS results are not evidence for this VITS experiment.

Acceptance criteria
The implementation is complete only when all of the following are true:
1. The manifest proves a female-only Hindi subset of 1.70–1.80 hours.
2. Every retained WAV has valid baseline and clustered labels.
3. Both models use the same WAV files and the same split lists.
4. Cluster tokens are handled as atomic direct input tokens.
5. Both VITS runs complete and save best-validation checkpoints.
6. Both held-out and unseen sets have paired baseline/clustered audio outputs.
7. Automated and human-evaluation templates/results are produced.
8. Results clearly distinguish preliminary automated results from human results.
9. No audio, checkpoints, secrets, or copied dataset contents are committed to Git.
10. Existing G2P artifacts, old samples, and reports are preserved.

If a required dependency, access approval, licence condition, or source dataset field is unavailable, stop at that point. Do not invent a workaround. Report the exact blocker, evidence, and the smallest action needed from the user.
```