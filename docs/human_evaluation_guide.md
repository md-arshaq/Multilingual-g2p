# Multilingual TTS Human Evaluation Web App (`tts_app`)

## Overview

The `tts_app` package provides a standalone, web application for conducting **subjective listening tests (Mean Opinion Score - MOS)** comparing the **Baseline (57 phonemes, 62 tokens)** vs **Clustered (39 clusters, 44 tokens)** VITS models across **Hindi**, **Marathi**, and **Gujarati**.

---

## 🚀 How to Run the App

1. Launch the server from WSL / command line:
   ```bash
   wsl -d Ubuntu-22.04 /mnt/d/g2p_env/bin/python tts_app/app.py
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

---

## 🏗️ Architecture & Folder Structure

```
tts_app/
├── app.py                     # Flask web server & REST API
├── analysis.py                # Offline statistical report & inter-rater analysis
├── templates/
│   ├── base.html              # Shared dark-mode base layout
│   ├── index.html             # Landing page with language selector & sample count picker
│   ├── evaluate.html          # Blind AB evaluation interface with dual audio players
│   └── results.html           # Real-time unblinded results dashboard with Chart.js
└── static/
    ├── css/
    │   └── style.css          # Design system (glassmorphism dark mode)
    └── js/
        ├── evaluate.js        # Audio player, rating logic & keyboard shortcuts
        └── results.js         # Analytics data loader & chart rendering
```

---

## 🎯 Key Features

1. **Blind A/B Randomization**:
   - For every trial and evaluator, the assignment of `Sample A` and `Sample B` (baseline vs clustered) is deterministically randomized with a 50/50 probability.
   - Evaluators never see model names during testing.
2. **Language & Sample Count Filtering**:
   - Filter by **Hindi (70 pairs)**, **Marathi (60 pairs)**, **Gujarati (46 pairs)**, or **All Languages (176 pairs)**.
   - Choose any number of trials from **10** up to the selected language's maximum.
   - Preset chips: `Quick (10)`, `Standard (25)`, `Thorough (50)`, `All Available`.
3. **Tactile 1–5 MOS Rating & Pairwise Preference**:
   - Instant 1.0 to 5.0 score buttons in 0.5 increments.
   - Pairwise preference selector ("Sample A", "Sample B", "No Preference / Equal").
4. **Keyboard Shortcuts**:
   - `1`: Play Sample A
   - `2`: Play Sample B
   - `Enter`: Submit & Advance to Next Trial
5. **Real-time Results Dashboard**:
   - Live unblinded MOS comparison across languages.
   - Preference distribution doughnut chart.
   - One-click CSV export (`/api/export/csv`).
6. **Offline Analysis Script**:
   - Run `python tts_app/analysis.py` to generate statistical reports with paired t-test, Wilcoxon, and Cohen's d effect sizes in `results/human_eval/`.
