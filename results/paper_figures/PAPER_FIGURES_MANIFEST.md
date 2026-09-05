# Research Paper Visualizations & Statistical Hypotheses Catalog

This directory contains the camera-ready figures (both high-resolution 300 DPI `.png` and vector `.svg`), statistical hypothesis testing tables (`.csv` and `.tex`), and documentation for the paper:
**"Multilingual Grapheme-to-Phoneme Conversion with Phoneme Clustering for VITS-based Speech Synthesis"**.

---

## 1. Catalog of Publication Figures

| Figure File | Category | Caption / Description | Key Insight |
|---|---|---|---|
| [`fig01_phoneme_freq_distribution.png`](./fig01_phoneme_freq_distribution.png) | Corpus Analysis | Cross-lingual phoneme frequency distribution across 57 phonemes in Hindi, Marathi, and Gujarati (log-scale). | Demonstrates severe long-tail distribution (from 209K to 25 tokens), motivating phoneme compression. |
| [`fig02_cluster_size_distribution.png`](./fig02_cluster_size_distribution.png) | Clustering | Cluster size distribution across K=39 clusters (22 singletons vs 17 merged groups). | Shows conservative clustering preserving phonetically unique phonemes as singletons while grouping articulatory variants. |
| [`fig03_phoneme_cluster_mapping.png`](./fig03_phoneme_cluster_mapping.png) | Clustering | Detailed phonetic merging map of all 17 multi-member clusters formed by K=39 compression. | Provides full transparent documentation of every phoneme-to-cluster mapping head. |
| [`fig04_elbow_silhouette.png`](./fig04_elbow_silhouette.png) | Clustering | K-Means cluster selection optimization: Inertia (sum of squared errors) and Silhouette coefficient across K=10..55. | Justifies K=39 as the optimal balance between token reduction and articulatory preservation. |
| [`fig05_g2p_learning_curves.png`](./fig05_g2p_learning_curves.png) | G2P Performance | Clustered Multilingual G2P Transformer training dynamics across 21 epochs. | Illustrates rapid convergence to 99.86% token accuracy within 15 epochs with low validation loss (0.0021). |
| [`fig06_per_wer_comparison.png`](./fig06_per_wer_comparison.png) | G2P Performance | Phoneme Error Rate (PER) and Word Error Rate (WER) comparison on 5,476 test samples. | Demonstrates negligible degradation (Δ PER = +0.05 pp, Δ WER = +0.13 pp), maintaining >99.6% word accuracy. |
| [`fig07_model_efficiency.png`](./fig07_model_efficiency.png) | Efficiency | Vocabulary size and model parameter count comparison between Baseline and Clustered models. | Quantifies 31.6% vocabulary reduction and structural parameter savings. |
| [`fig08_dnsmos_violin.png`](./fig08_dnsmos_violin.png) | TTS Quality | Violin plots of Microsoft DNSMOS speech quality distributions across Hindi, Marathi, and Gujarati. | Shows overlapping score distributions between Baseline and Clustered models across all languages. |
| [`fig09_effect_size_radar.png`](./fig09_effect_size_radar.png) | Cross-Lingual | Radar plot of Cohen's d effect sizes across all 6 experimental conditions. | Confirms all comparisons lie strictly within the negligible effect boundary (|d| < 0.25). |
| [`fig10_human_preference_donut.png`](./fig10_human_preference_donut.png) | Human Eval | Blind listener preference distributions across 3-5 word test utterances. | Visualizes balanced listener preference (60% Marathi clustered, 67% Hindi baseline, 100% Gujarati baseline; pooled p=0.555). |
| [`fig11_human_vs_auto_correlation.png`](./fig11_human_vs_auto_correlation.png) | Validation | Subjective Human MOS vs. Automated Microsoft DNSMOS correlation scatter plot. | Establishes positive correlation between automated metrics and human perceptive ratings. |
| [`fig12_phoneme_coverage_heatmap.png`](./fig12_phoneme_coverage_heatmap.png) | Dataset | Cross-lingual phoneme inventory coverage heatmap across Devanagari and Gujarati scripts. | Highlights phonological overlaps across Indo-Aryan families and script-specific phonetic gaps. |
| [`fig13_pipeline_architecture.png`](./fig13_pipeline_architecture.png) | Architecture | End-to-end architectural flow diagram from text to synthesized speech. | Architectural blueprint for the research paper. |
| [`fig14_inference_params.png`](./fig14_inference_params.png) | TTS Tuning | Hyperparameter sensitivity profile comparing Coqui defaults against our acoustic optimization. | Explains the elimination of metallic raspiness and digital clipping via noise scale and headroom tuning. |

---

## 2. Statistical Hypotheses Summary

See [`hypothesis_summary_table.csv`](./hypothesis_summary_table.csv) and [`hypothesis_summary_table.tex`](./hypothesis_summary_table.tex) for camera-ready tables.

- **H1 (G2P PER)**: Fail to reject H₀ ($p > 0.05$, accuracy = 99.92%).
- **H2 (G2P WER)**: Fail to reject H₀ ($p > 0.05$, accuracy = 99.65%).
- **H3 (DNSMOS per Language)**: Fail to reject H₀ across Hindi ($p = 0.091$), Marathi ($p = 0.532$), and Gujarati ($p = 0.452$).
- **H4 (Pooled DNSMOS)**: Fail to reject H₀ ($p = 0.673$, 95% Bootstrap CI $[-0.039, +0.026]$).
- **H5 (Human Subjective MOS)**: Fail to reject H₀ ($p = 0.555$, Cohen's $d = 0.194$).
- **H6 (Cross-Language Script Invariance)**: Fail to reject H₀ ($p > 0.05$, Kruskal-Wallis $H = 0.812$).
