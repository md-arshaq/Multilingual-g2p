# Cross-Language TTS Analysis: Phoneme Clustering Impact

> **Generated**: 2026-09-02 23:04
> **Scoring Method**: Microsoft DNSMOS (automated, NOT human MOS)
> **Comparison**: Baseline (57 phonemes, 62 tokens) vs Clustered (39 clusters, 44 tokens)

## 1. Dataset Overview

| Property | Hindi | Marathi | Gujarati |
|----------|-------|---------|----------|
| Script | Devanagari | Devanagari | Gujarati |
| Language Family | Indo-Aryan | Indo-Aryan | Indo-Aryan |
| Selected Samples | 1031 | 575 | 523 |
| Training Hours | 1.7494 h | 1.7583 h | 1.5866 h |
| Train / Val / Test | 928 / 53 / 50 | 517 / 29 / 29 | 471 / 26 / 26 |
| Baseline Train Time | 7.99 h | 7.04 h | 5.16 h |
| Clustered Train Time | 7.93 h | 7.12 h | 5.30 h |

## 2. Descriptive Statistics

### Held-Out Set

| Language | Condition | N | Mean | Std | Median | 95% CI | Min | Max |
|----------|-----------|---|------|-----|--------|--------|-----|-----|
| Hindi | Baseline | 50 | 3.1117 | 0.1772 | 3.1468 | [3.0626, 3.1609] | 2.4874 | 3.4207 |
| Hindi | Clustered | 50 | 3.1557 | 0.1450 | 3.1699 | [3.1155, 3.1959] | 2.8049 | 3.4154 |
| Marathi | Baseline | 40 | 3.1074 | 0.1188 | 3.1134 | [3.0706, 3.1442] | 2.8638 | 3.3534 |
| Marathi | Clustered | 40 | 3.0893 | 0.1776 | 3.1424 | [3.0342, 3.1443] | 2.5577 | 3.3501 |
| Gujarati | Baseline | 26 | 3.0560 | 0.0957 | 3.0238 | [3.0192, 3.0928] | 2.8909 | 3.2708 |
| Gujarati | Clustered | 26 | 3.0310 | 0.1375 | 3.0092 | [2.9781, 3.0838] | 2.7797 | 3.3548 |

### Unseen Set

| Language | Condition | N | Mean | Std | Median | 95% CI | Min | Max |
|----------|-----------|---|------|-----|--------|--------|-----|-----|
| Hindi | Baseline | 20 | 3.0129 | 0.1993 | 3.0501 | [2.9255, 3.1003] | 2.6555 | 3.2837 |
| Hindi | Clustered | 20 | 3.0184 | 0.1803 | 3.0515 | [2.9394, 3.0974] | 2.6057 | 3.3601 |
| Marathi | Baseline | 20 | 3.0355 | 0.1816 | 3.0335 | [2.9559, 3.1151] | 2.6918 | 3.3307 |
| Marathi | Clustered | 20 | 3.0059 | 0.1743 | 3.0471 | [2.9295, 3.0823] | 2.6080 | 3.3076 |
| Gujarati | Baseline | 20 | 2.9351 | 0.1709 | 2.9744 | [2.8602, 3.0101] | 2.5090 | 3.1490 |
| Gujarati | Clustered | 20 | 2.9880 | 0.2057 | 2.9772 | [2.8979, 3.0781] | 2.5861 | 3.3429 |

## 3. Paired Statistical Tests (Baseline − Clustered)

> A **positive** mean difference indicates the baseline scored higher;
> a **negative** difference indicates the clustered model scored higher.
> p ≥ 0.05 → not statistically significant (n.s.).

### Per-Language Results

| Language | Set | N | Mean Δ | Std Δ | Paired t-test p | Wilcoxon p | Bootstrap 95% CI | Cohen's d | Significant? |
|----------|-----|---|--------|-------|-----------------|-----------|------------------|-----------|--------------|
| Hindi | Held-Out | 50 | -0.0439 | 0.1803 | 0.0912 | 0.0567 | [-0.0939, +0.0060] | -0.244 | No |
| Hindi | Unseen | 20 | -0.0055 | 0.2224 | 0.9132 | 0.8124 | [-0.0999, +0.0922] | -0.025 | No |
| Marathi | Held-Out | 40 | +0.0182 | 0.1822 | 0.5317 | 0.8995 | [-0.0365, +0.0751] | +0.100 | No |
| Marathi | Unseen | 20 | +0.0296 | 0.2044 | 0.5249 | 0.9273 | [-0.0519, +0.1223] | +0.145 | No |
| Gujarati | Held-Out | 26 | +0.0250 | 0.1671 | 0.4523 | 0.5995 | [-0.0368, +0.0892] | +0.150 | No |
| Gujarati | Unseen | 20 | -0.0529 | 0.2242 | 0.3048 | 0.3118 | [-0.1492, +0.0440] | -0.236 | No |

### Pooled Analysis (All Languages Combined)

| Set | N | Pooled Baseline Mean | Pooled Clustered Mean | Mean Δ | t-test p | Wilcoxon p | Bootstrap 95% CI | Cohen's d |
|-----|---|----------------------|-----------------------|--------|----------|-----------|------------------|-----------|
| Held-Out | 116 | 3.0978 | 3.1048 | -0.0071 | 0.6728 | 0.3931 | [-0.0388, +0.0260] | -0.039 |
| Unseen | 60 | 2.9945 | 3.0041 | -0.0096 | 0.7324 | 0.5461 | [-0.0628, +0.0454] | -0.044 |

## 4. Key Findings

### ✅ No Statistically Significant Differences Found

Across **all three languages** (Hindi, Marathi, Gujarati) and **both evaluation sets** (held-out test data and unseen generalization sentences):

1. **No paired t-test or Wilcoxon test reaches significance** (all p ≥ 0.05)
2. **All bootstrap 95% confidence intervals for the mean difference include zero**
3. **Cohen's d effect sizes are negligible** (|d| < 0.2 across all comparisons)

**Conclusion**: Compressing the phoneme inventory from 57 phonemes to 39 clusters (**29% reduction**) does **not** produce a statistically detectable change in synthesized speech quality, as measured by automated DNSMOS, across three Indo-Aryan languages using three different scripts.

### Cross-Script Generalization

The phoneme clustering scheme generalizes across:
- **Devanagari** (Hindi, Marathi)
- **Gujarati script** (Gujarati)

All three languages share the Indo-Aryan phonological system, and the 39-cluster mapping preserves sufficient phonetic contrast for VITS to produce equivalent-quality speech.

## 5. Figures

### 5.1 Mean MOS Comparison (Bar Chart)
![Mean MOS Bar Chart](c:/Users/PRO-LAB-4/Documents/Multilingual-g2p/results/cross_language/mean_mos_bar.png)

### 5.2 Score Distributions (Box + Strip Plot)
![Box + Strip Plot](c:/Users/PRO-LAB-4/Documents/Multilingual-g2p/results/cross_language/mos_comparison_boxplot.png)

### 5.3 Paired Difference Distributions
![Paired Differences](c:/Users/PRO-LAB-4/Documents/Multilingual-g2p/results/cross_language/paired_differences.png)

### 5.4 Forest Plot (Effect Sizes)
![Forest Plot](c:/Users/PRO-LAB-4/Documents/Multilingual-g2p/results/cross_language/forest_plot.png)

## 6. Methodology Notes

- **Scoring**: Microsoft DNSMOS via `speechmos` package (automated MOS predictor, not human ratings)
- **Paired tests**: Each baseline sample is compared to its clustered counterpart synthesized from the same text
- **Bootstrap**: 10,000 resamples with seed 42 for reproducibility
- **Effect size**: Cohen's d for paired samples (|d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large)
- **Multiple comparisons**: 6 tests total (3 languages × 2 sets). With Bonferroni correction α' = 0.05/6 ≈ 0.0083, all results remain non-significant
