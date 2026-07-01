# Evaluation Report — Baseline vs. Clustered G2P Model

## Summary

This report compares the baseline G2P model (trained on raw phonemes) with the
clustered G2P model (trained on cluster labels derived from K-Means phoneme clustering).

Both models use the same Transformer architecture:
- Layers: 3
- d_model: 128
- Attention heads: 4
- Feed-forward dim: 512
- Dropout: 0.1

Both use the same train/val/test split (80/10/10, random seed=42).

## Comparison Table

| Metric | Baseline | Clustered | Δ |
|--------|----------|-----------|---|
| PER | 0.0003 (0.03%) | 0.0008 (0.08%) | +0.0006 |
| WER | 0.0022 (0.22%) | 0.0035 (0.35%) | +0.0013 |
| Output Vocab Size | 57 phonemes | 39 clusters | −31.6% |
| Parameter Count | 1,422,141 | 1,417,515 | -4,626 |
| Checkpoint Size (MB) | 5.68 | 5.66 | -0.02 |
| Test Samples | 5476 | 5476 | — |

## Key Findings

1. **Vocab Reduction:** The phoneme vocabulary was reduced from 57 to 39 tokens (31.6% reduction).

2. **PER Change:** PER increased by 0.06 percentage points.

3. **WER Change:** WER increased by 0.13 percentage points.

4. **Model Size:** The clustered model has fewer parameters (4,626 fewer) due to the smaller output vocabulary.

## Sample Predictions

### Baseline Model

| # | Predicted | Expected | PER |
|---|-----------|----------|-----|
| 1 | `n a aa` | `n a aa` | 0.0000 |
| 2 | `r a q g a u n a w a aa l a aa` | `r a q g a u n a w a aa l a aa` | 0.0000 |
| 3 | `s a u h a aa n a aa` | `s a u h a aa n a aa` | 0.0000 |
| 4 | `ph a i r a w a aa y a c a ee` | `ph a i r a w a aa y a c a ee` | 0.0000 |
| 5 | `b a j a ee tx a n a ii` | `b a j a ee tx a n a ii` | 0.0000 |
| 6 | `o r a b a ii tx a m a aa q` | `o r a b a ii tx a m a aa q` | 0.0000 |
| 7 | `sh a aa q t a i c a aa h a n a aa` | `sh a aa q t a i c a aa h a n a aa` | 0.0000 |
| 8 | `bh a aa w a ee` | `bh a aa w a ee` | 0.0000 |
| 9 | `s a r a w a aa q c a y a aa c a` | `s a r a w a aa q c a y a aa c a` | 0.0000 |
| 10 | `n a i hq s a w a aa r a th a` | `n a i hq s a w a aa r a th a` | 0.0000 |

### Clustered Model

| # | Predicted | Expected | PER |
|---|-----------|----------|-----|
| 1 | `C24 C0 C1` | `C24 C0 C1` | 0.0000 |
| 2 | `C29 C0 C23 C13 C0 C36 C24 C0 C37 C0 C1 C21 C0 C1` | `C29 C0 C23 C13 C0 C36 C24 C0 C37 C0 C1 C21 C0 C1` | 0.0000 |
| 3 | `C30 C0 C36 C15 C0 C1 C24 C0 C1` | `C30 C0 C36 C15 C0 C1 C24 C0 C1` | 0.0000 |
| 4 | `C12 C0 C16 C29 C0 C37 C0 C1 C38 C0 C5 C0 C11` | `C12 C0 C16 C29 C0 C37 C0 C1 C38 C0 C5 C0 C11` | 0.0000 |
| 5 | `C3 C0 C17 C0 C11 C34 C0 C24 C0 C16` | `C3 C0 C17 C0 C11 C34 C0 C24 C0 C16` | 0.0000 |
| 6 | `C26 C29 C0 C3 C0 C16 C34 C0 C22 C0 C1 C23` | `C26 C29 C0 C3 C0 C16 C34 C0 C22 C0 C1 C23` | 0.0000 |
| 7 | `C31 C0 C1 C23 C32 C0 C16 C5 C0 C1 C15 C0 C24 C0 C1` | `C31 C0 C1 C23 C32 C0 C16 C5 C0 C1 C15 C0 C24 C0 C1` | 0.0000 |
| 8 | `C4 C0 C1 C37 C0 C11` | `C4 C0 C1 C37 C0 C11` | 0.0000 |
| 9 | `C30 C0 C29 C0 C37 C0 C1 C23 C5 C0 C38 C0 C1 C5 C0` | `C30 C0 C29 C0 C37 C0 C1 C23 C5 C0 C38 C0 C1 C5 C0` | 0.0000 |
| 10 | `C24 C0 C16 C15 C30 C0 C37 C0 C1 C29 C0 C33 C0` | `C24 C0 C16 C15 C30 C0 C37 C0 C1 C29 C0 C33 C0` | 0.0000 |

## Notes

- PER is computed as `editdistance(predicted, reference) / len(reference)` at the token level.
- WER is binary: 1.0 if the entire predicted sequence differs from reference, 0.0 if exact match.
- Both models were evaluated on the same held-out test set using greedy decoding.
- The clustered model's PER operates at the cluster level — each "error" is a wrong cluster assignment.
