# Vocab Reduction Report

## Summary

| Metric | Value |
|--------|-------|
| Original phoneme count | 57 |
| Cluster count | 39 |
| **Vocab reduction** | **31.6%** |
| Total phoneme tokens | 541,757 |
| Dataset lines | 54,753 |
| Languages | GU, HI, MR |

## Language Distribution

| Language | Entries |
|----------|---------|
| GU | 16,676 |
| HI | 15,623 |
| MR | 22,454 |

## Cluster Detail

| Cluster ID | Label | Members | Size | Total Frequency | Note |
|------------|-------|---------|------|-----------------|------|
| C0 | Phonetic_Group_a | a, ax | 2 | 0 |  |
| C1 | Phonetic_Group_aa | aa | 1 | 0 | ⚠️ Singleton |
| C2 | Phonetic_Group_ei | ae, ei | 2 | 0 |  |
| C3 | Phonetic_Group_b | b | 1 | 0 | ⚠️ Singleton |
| C4 | Phonetic_Group_bh | bh | 1 | 0 | ⚠️ Singleton |
| C5 | Phonetic_Group_c | c | 1 | 0 | ⚠️ Singleton |
| C6 | Phonetic_Group_ch | ch | 1 | 0 | ⚠️ Singleton |
| C7 | Phonetic_Group_d | d | 1 | 0 | ⚠️ Singleton |
| C8 | Phonetic_Group_dh | dh | 1 | 0 | ⚠️ Singleton |
| C9 | Phonetic_Group_dx | dx, dxq | 2 | 0 |  |
| C10 | Phonetic_Group_dxh | dxh, dxhq | 2 | 0 |  |
| C11 | Phonetic_Group_e | ee | 1 | 0 | ⚠️ Singleton |
| C12 | Phonetic_Group_ph | f, ph | 2 | 0 |  |
| C13 | Phonetic_Group_g | g, gq | 2 | 0 |  |
| C14 | Phonetic_Group_gh | gh | 1 | 0 | ⚠️ Singleton |
| C15 | Phonetic_Group_h | h, hq | 2 | 0 |  |
| C16 | Phonetic_Group_i | i, ii | 2 | 0 |  |
| C17 | Phonetic_Group_j | j, z | 2 | 0 |  |
| C18 | Phonetic_Group_jh | jh | 1 | 0 | ⚠️ Singleton |
| C19 | Phonetic_Group_k | k, kq | 2 | 0 |  |
| C20 | Phonetic_Group_kh | kh, khq | 2 | 0 |  |
| C21 | Phonetic_Group_l | l, lx | 2 | 0 |  |
| C22 | Phonetic_Group_m | m | 1 | 0 | ⚠️ Singleton |
| C23 | Phonetic_Group_q | mq, q | 2 | 0 |  |
| C24 | Phonetic_Group_n | n, ng, nj | 3 | 0 |  |
| C25 | Phonetic_Group_nx | nx | 1 | 0 | ⚠️ Singleton |
| C26 | Phonetic_Group_o | o | 1 | 0 | ⚠️ Singleton |
| C27 | Phonetic_Group_ou | ou | 1 | 0 | ⚠️ Singleton |
| C28 | Phonetic_Group_p | p | 1 | 0 | ⚠️ Singleton |
| C29 | Phonetic_Group_r | r, rq | 2 | 0 |  |
| C30 | Phonetic_Group_s | s | 1 | 0 | ⚠️ Singleton |
| C31 | Phonetic_Group_sh | sh, sx | 2 | 0 |  |
| C32 | Phonetic_Group_t | t | 1 | 0 | ⚠️ Singleton |
| C33 | Phonetic_Group_th | th | 1 | 0 | ⚠️ Singleton |
| C34 | Phonetic_Group_tx | tx | 1 | 0 | ⚠️ Singleton |
| C35 | Phonetic_Group_txh | txh | 1 | 0 | ⚠️ Singleton |
| C36 | Phonetic_Group_u | u, uu | 2 | 0 |  |
| C37 | Phonetic_Group_w | w | 1 | 0 | ⚠️ Singleton |
| C38 | Phonetic_Group_y | y | 1 | 0 | ⚠️ Singleton |

## Singleton Cluster Analysis

Singleton clusters are clusters containing only one phoneme. These are typically rare phonemes
that are phonetically distinct enough to not merge with any other cluster at K=39.

**Strategy chosen:** Keep singletons as-is with their dedicated cluster IDs.

**Rationale:** Since these phonemes already have unique cluster labels (C1, C4, C6, C8, C10, C11),
the model can still learn them — just with fewer training examples. Merging them with the nearest
centroid neighbor could hurt accuracy for words containing these specific phonemes.

| Singleton Cluster | Phoneme | Frequency | % of Total Tokens |
|-------------------|---------|-----------|-------------------|
| C1 | `aa` | 0 | 0.000% |
| C3 | `b` | 0 | 0.000% |
| C4 | `bh` | 0 | 0.000% |
| C5 | `c` | 0 | 0.000% |
| C6 | `ch` | 0 | 0.000% |
| C7 | `d` | 0 | 0.000% |
| C8 | `dh` | 0 | 0.000% |
| C11 | `ee` | 0 | 0.000% |
| C14 | `gh` | 0 | 0.000% |
| C18 | `jh` | 0 | 0.000% |
| C22 | `m` | 0 | 0.000% |
| C25 | `nx` | 0 | 0.000% |
| C26 | `o` | 0 | 0.000% |
| C27 | `ou` | 0 | 0.000% |
| C28 | `p` | 0 | 0.000% |
| C30 | `s` | 0 | 0.000% |
| C32 | `t` | 0 | 0.000% |
| C33 | `th` | 0 | 0.000% |
| C34 | `tx` | 0 | 0.000% |
| C35 | `txh` | 0 | 0.000% |
| C37 | `w` | 0 | 0.000% |
| C38 | `y` | 0 | 0.000% |

## Transformation Example

**Original:**
```
<HI> अँग	a mq g a
<HI> अंक	a q k a
<GU> સહૃદય	s a h a d a y a
```

**Clustered:**
```
<HI> अँग	C0 C9 C3 C0
<HI> अंक	C0 C2 C3 C0
<GU> સહૃદય	C7 C0 C2 C0 C3 C0 C2 C0
```

## Output Files

- `data/multilingual_g2p_clustered.txt` — Full clustered dataset (54,753 lines)
- `results/vocab_reduction_report.md` — This report
