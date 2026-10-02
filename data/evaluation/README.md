# Evaluation & Threshold Tuning (Part 2)

## Protocol & Architecture

Evaluation adheres to an independent partition protocol to prevent threshold overfitting and data leakage:
```text
Clean Evaluation Dataset
         │
         ├── Validation Partition (23 images: 20 genuine probes, 2 independent impostor identities, 1 invalid no-face)
         │        ↓
         │    Threshold Sweep (0.00 to 1.00)
         │        ↓
         │    Select Threshold (Highest Recall subject to FPR <= 0.01)
         │        ↓
         │    FROZEN THRESHOLD = 0.89
         │
         └── Test Partition (23 images: 20 genuine probes under challenging conditions, 2 independent impostors, 1 invalid multi-face)
                  ↓
              Frozen Test Evaluation (Operating Threshold = 0.89)
                  ↓
              Wilson 95% Confidence Intervals & Score Distributions
```

---

## Historical Comparison (Old Contaminated Dataset vs. Clean Dataset)

| Metric / Property | Old Contaminated Dataset | Clean Independent Dataset |
| :--- | :--- | :--- |
| **Impostor Composition** | Mirrored transforms of enrolled identities (`impostor_01`–`05`) + SHA-256 duplicates (`07`–`10`) | Genuinely independent human subjects (`WIN_...` video + `impostor_06` captures) |
| **Duplicate Files** | 8 duplicate pairs | **0 duplicate files** |
| **Max Impostor Similarity** | `0.9668` (contaminated same-subject reflection) | **`0.2441`** (clean biometric separation) |
| **Selected Threshold** | `0.97` (artificially inflated due to pseudo-negatives) | **`0.89`** (derived from clean validation sweep) |
| **Validation AUROC** | `0.9808` | **`1.0000`** |
| **Test AUROC** | `0.9485` | **`1.0000`** |
| **Status** | **OBSOLETE / INVALID** | **CLEAN EVALUATION BASELINE** |

---

## Clean Evaluation Results Summary

- **Selected Operating Threshold**: `0.89` (**FROZEN**)
- **Validation (105 pairs: 20 genuine, 85 impostor)**:
  - Precision: `100.0%`
  - Recall: `100.0%`
  - FPR: `0.0%`
  - F1-Score: `100.0%`
- **Frozen Test Evaluation (110 pairs: 20 genuine, 90 impostor)**:
  - Precision: `100.0%` (95% CI: `70.1% – 100.0%`, $n=9$)
  - Recall: `45.0%` (95% CI: `25.8% – 65.8%`, $n=20$)
  - FPR: `0.0%` (95% CI: `0.0% – 4.1%`, $n=90$)
  - F1-Score: `62.1%`
- **Test Score Distributions**:
  - Genuine: Min=`0.6497`, Max=`0.9859`, Mean=`0.8547`, Median=`0.8698`
  - Impostor: Min=`-0.1314`, Max=`0.2441`, Mean=`0.0333`, Median=`0.0200`
