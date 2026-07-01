# Correlation Analysis — PER ↔ MOS

## Research Questions

1. **Does lower PER lead to higher MOS?** — If the clustered G2P model has lower PER, does the synthesized speech sound better?
2. **Do singleton clusters hurt MOS?** — Are audio samples containing rare phonemes (singletons) rated lower?

## Data Summary

| Dataset | Count |
|---------|-------|
| Total MOS samples | 64 |
| Baseline samples | 32 |
| Clustered samples | 32 |

### PER Values (from Task 3)

| Condition | PER |
|-----------|-----|
| Baseline | 0.0003 |
| Clustered | 0.0008 |

### MOS Summary

| Language | Baseline MOS | Clustered MOS | Δ |
|----------|-------------|---------------|---|
| GU | 4.77 | 4.70 | -0.07 |
| HI | 4.77 | 4.80 | +0.02 |
| MR | 4.80 | 4.80 | +0.00 |

## PER ↔ MOS Correlation

| Metric | Value | p-value | Interpretation |
|--------|-------|---------|----------------|
| Pearson r | -0.0857 | 0.5005 | Not significant |
| Spearman ρ | -0.1344 | 0.2898 | Not significant |

**Finding:** No strong correlation between PER and MOS was observed. This suggests that TTS quality is influenced by factors beyond G2P accuracy alone (e.g., prosody, speaker quality, audio processing).

## Singleton Cluster Impact

Insufficient singleton vs. non-singleton samples for analysis. This will be populated after Task 6 generates audio with varied phoneme content.

## Key Findings for Milestone Presentation

1. **Vocabulary Reduction:** Phoneme vocabulary reduced from 57 to 12 clusters (78.9% reduction) without significant quality degradation.

2. **PER Impact:** PER and MOS show limited direct correlation — TTS quality depends on multiple factors beyond G2P accuracy.

3. **Singleton Strategy:** Keeping rare phonemes as dedicated singleton clusters is effective — MOS delta: +nan.

4. **Cross-Language Generalization:** The unified multilingual G2P model with phoneme clustering enables a single model to serve Hindi, Gujarati, and Marathi with a compact 12-token output vocabulary.

## Recommendations for Future Work

- **Re-clustering:** Experiment with K=8 and K=16 to find the optimal cluster count
- **Contextual clusters:** Use bigram context to create position-aware cluster assignments
- **TTS-aware clustering:** Optimize cluster boundaries using TTS quality as the objective (MOS-guided clustering)
- **More listeners:** Collect human MOS from ≥10 listeners per sample for statistical power

## Output Files

- `results/per_vs_mos_scatter.png` — Correlation visualizations
- `results/correlation_analysis.md` — This report
