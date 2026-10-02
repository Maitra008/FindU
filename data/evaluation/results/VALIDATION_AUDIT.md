# Validation Dataset Audit Report

**Status**: COMPLETE  
**Conclusion**: `DATASET_HAS_ISSUES`  
**Operating Threshold**: `NOT CHANGED` (Retained pending dataset correction)

---

## 1. Executive Summary

This audit was conducted prior to finalizing the operating threshold for the Recognition Core. The objective was to investigate an anomalous similarity score distribution on the validation dataset—specifically, several negative ("impostor") comparisons producing unexpectedly high cosine similarities between **0.9226** and **0.9668**.

### Key Findings
1. **Root Cause of High Impostor Scores**: The high-scoring impostor images (`impostor_01` through `impostor_05`) were discovered to be horizontally flipped transformations of the registered identities (`person_01` through `person_05`). Because ArcFace aligns faces via 5 facial landmarks and is invariant to bilateral reflection, the model accurately recognized the underlying facial geometry ($\text{similarity} > 0.92$). However, because the dataset directory structure assigned them novel identity labels (`impostor_XX`), the evaluation system marked them as negative pairs (`label = 0`).
2. **Exact Duplicate Contamination**: Identities `impostor_07`, `impostor_08`, `impostor_09`, and `impostor_10` were byte-for-byte SHA-256 duplicate copies of `impostor_01` through `impostor_04`. This explains the exact repetition of similarity values (e.g. `0.9668`, `0.9428`, `0.9414`, `0.9226`).
3. **Pipeline Integrity**: The face detection (SCRFD), ArcFace embedding extraction, L2 normalization ($\|\mathbf{e}\|_2 = 1.0$), and pair-generation logic were audited and verified to be working correctly with zero caching bugs or state leakage.
4. **Low-Scoring Genuine Pairs**: Genuine comparisons with similarity $< 0.97$ ($0.8916 - 0.9637$) were verified as legitimate difficult cases (`VALID_HARD_CASE`) caused by intentional downscaling (low resolution) and directional motion blur.

---

## 2. Dataset Summary

| Partition | Total Images | Valid (1-Face) | No-Face | Multi-Face | Total Pairs | Genuine Pairs | Impostor Pairs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation** | 31 | 30 | 1 | 0 | 150 | 20 | 130 |
| **Test** | 31 | 30 | 0 | 1 | 150 | 20 | 130 |

- **Enrolled Identities**: 5 (`person_01`, `person_02`, `person_03`, `person_04`, `person_05`)
- **Evaluation Identities in Validation**: 15 (5 genuine + 10 impostor labels)

---

## 3. High-Scoring Impostors ($\text{Similarity} \ge 0.90$)

All 9 negative pairs with cosine similarity $\ge 0.90$ are detailed below, sorted descending by similarity:

| Enrolled Person | Evaluation ID | Image Path | Similarity | Label | True Identity / Root Cause | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `person_03` | `impostor_03` | `data/evaluation/validation/impostor_03/val_impostor.jpg` | **0.9668** | 0 | Mirrored image of `person_03` | `WRONG_LABEL` |
| `person_03` | `impostor_09` | `data/evaluation/validation/impostor_09/val_impostor.jpg` | **0.9668** | 0 | SHA-256 duplicate of `impostor_03` | `DATA_DUPLICATE` |
| `person_02` | `impostor_02` | `data/evaluation/validation/impostor_02/val_impostor.jpg` | **0.9428** | 0 | Mirrored image of `person_02` | `WRONG_LABEL` |
| `person_02` | `impostor_08` | `data/evaluation/validation/impostor_08/val_impostor.jpg` | **0.9428** | 0 | SHA-256 duplicate of `impostor_02` | `DATA_DUPLICATE` |
| `person_01` | `impostor_01` | `data/evaluation/validation/impostor_01/val_impostor.jpg` | **0.9414** | 0 | Mirrored image of `person_01` | `WRONG_LABEL` |
| `person_01` | `impostor_07` | `data/evaluation/validation/impostor_07/val_impostor.jpg` | **0.9414** | 0 | SHA-256 duplicate of `impostor_01` | `DATA_DUPLICATE` |
| `person_05` | `impostor_05` | `data/evaluation/validation/impostor_05/val_impostor.jpg` | **0.9237** | 0 | Mirrored image of `person_05` | `WRONG_LABEL` |
| `person_04` | `impostor_04` | `data/evaluation/validation/impostor_04/val_impostor.jpg` | **0.9226** | 0 | Mirrored image of `person_04` | `WRONG_LABEL` |
| `person_04` | `impostor_10` | `data/evaluation/validation/impostor_10/val_impostor.jpg` | **0.9226** | 0 | SHA-256 duplicate of `impostor_04` | `DATA_DUPLICATE` |

### High-Priority Cases ($\text{Similarity} \ge 0.95$)
- **`person_03` vs `impostor_03` (0.9668)**: Cosine similarity $0.9668$ against `person_03` template, but $< 0.05$ against all other identities. Visual inspection and pixel differencing confirm this is `person_03` with horizontal reflection.
- **`person_03` vs `impostor_09` (0.9668)**: Bitwise clone of `impostor_03/val_impostor.jpg`.

---

## 4. Low-Scoring Genuine Cases ($\text{Similarity} < 0.97$)

All genuine comparisons below the candidate threshold $0.97$ represent legitimate real-world surveillance challenges:

| Enrolled Person | Probe Image | Similarity | Condition Tested | Pose / Illumination / Artifact | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `person_02` | `val_downscaled.jpg` | **0.8916** | Low Resolution | $48 \times 48$ downsample + upsample interpolation | `VALID_HARD_CASE` |
| `person_01` | `val_downscaled.jpg` | **0.9022** | Low Resolution | High-frequency detail loss | `VALID_HARD_CASE` |
| `person_05` | `val_downscaled.jpg` | **0.9191** | Low Resolution | Severe blur on facial landmarks | `VALID_HARD_CASE` |
| `person_04` | `val_downscaled.jpg` | **0.9232** | Low Resolution | Quantization artifacts | `VALID_HARD_CASE` |
| `person_02` | `val_motionblur.jpg` | **0.9349** | Motion Blur | Directional smear kernel across eyes | `VALID_HARD_CASE` |
| `person_01` | `val_motionblur.jpg` | **0.9589** | Motion Blur | Smearing across facial edges | `VALID_HARD_CASE` |
| `person_03` | `val_motionblur.jpg` | **0.9598** | Motion Blur | Linear blur convolution | `VALID_HARD_CASE` |
| `person_05` | `val_motionblur.jpg` | **0.9610** | Motion Blur | Smear along horizontal axis | `VALID_HARD_CASE` |
| `person_03` | `val_downscaled.jpg` | **0.9633** | Low Resolution | Moderate pixelation | `VALID_HARD_CASE` |
| `person_04` | `val_motionblur.jpg` | **0.9637** | Motion Blur | Edge degradation | `VALID_HARD_CASE` |

**Verification**: Face detection SCRFD retained $> 0.85$ confidence on all degraded genuine images. The score reduction is consistent with the expected sensitivity of ArcFace to feature degradation in low-resolution video probes.

---

## 5. Duplicate & Near-Duplicate Analysis

### Exact File Hash Matches (SHA-256)
- `data/evaluation/validation/impostor_01/val_impostor.jpg` $\equiv$ `impostor_07/val_impostor.jpg` (`c15f4056ee0519e5...`)
- `data/evaluation/validation/impostor_02/val_impostor.jpg` $\equiv$ `impostor_08/val_impostor.jpg` (`025193cd3a527d44...`)
- `data/evaluation/validation/impostor_03/val_impostor.jpg` $\equiv$ `impostor_09/val_impostor.jpg` (`9a57b29785053065...`)
- `data/evaluation/validation/impostor_04/val_impostor.jpg` $\equiv$ `impostor_10/val_impostor.jpg` (`c720f0ff036e48a9...`)
- Identical duplicates also replicated in `data/evaluation/test/impostor_01..04` vs `test/impostor_07..10`.

### Near-Duplicates (Transforms of Registered Subjects)
- `impostor_01` $\longleftrightarrow$ `person_01` (horizontal flip)
- `impostor_02` $\longleftrightarrow$ `person_02` (horizontal flip)
- `impostor_03` $\longleftrightarrow$ `person_03` (horizontal flip)
- `impostor_04` $\longleftrightarrow$ `person_04` (horizontal flip)
- `impostor_05` $\longleftrightarrow$ `person_05` (horizontal flip)
- Truly distinct impostor: `impostor_06` (unrelated subject, similarity $\le 0.1469$ against all 5 enrolled identities).

---

## 6. Leakage Analysis

1. **Registration $\longleftrightarrow$ Validation / Test**:
   - `data/registration/` images (`reg_01`, `reg_02`, `reg_03`) are independent crops from the same capture sessions as the baseline probe images (`val_normal.jpg`).
   - The validation "impostor" pool contains transformed versions of the enrolled templates, creating pseudo-negative leakage.
2. **Validation $\longleftrightarrow$ Test**:
   - Validation and test sets use independent degradation parameters for genuine identities (`lowlight`, `motionblur`, `downscaled` in val vs `compressed`, `lowres_lowlight`, `occlusion_glasses`, `uneven_light` in test).
   - However, the impostor pools in both sets suffer from the same mirroring and duplication pattern.

---

## 7. Embedding Pipeline & Face Detection Verification

- **L2 Normalization**: PASS. $\|\mathbf{e}\|_2 = 1.0 \pm 10^{-6}$ for all extracted feature vectors.
- **Embedding Isolation**: PASS. Every image is loaded, detected, and embedded independently with no cross-image state caching.
- **Similarity Computation**: PASS. Cosine similarity calculated via dot product $\mathbf{u} \cdot \mathbf{v}$.
- **Face Detection & Selection**: PASS. SCRFD detected exactly 1 face for all valid images. Single-face filtering properly discarded `invalid_no_face.jpg` (0 faces) and `invalid_multi_face.jpg` (2 faces).

---

## 8. Findings Classification

```text
- impostor_01 (vs person_01): WRONG_LABEL (mirrored person_01)
- impostor_02 (vs person_02): WRONG_LABEL (mirrored person_02)
- impostor_03 (vs person_03): WRONG_LABEL (mirrored person_03)
- impostor_04 (vs person_04): WRONG_LABEL (mirrored person_04)
- impostor_05 (vs person_05): WRONG_LABEL (mirrored person_05)
- impostor_06 (vs all):       VALID_IMPOSTOR (independent subject, sim <= 0.1469)
- impostor_07 (vs person_01): DATA_DUPLICATE (exact clone of impostor_01)
- impostor_08 (vs person_02): DATA_DUPLICATE (exact clone of impostor_02)
- impostor_09 (vs person_03): DATA_DUPLICATE (exact clone of impostor_03)
- impostor_10 (vs person_04): DATA_DUPLICATE (exact clone of impostor_04)
- genuine < 0.97 (10 cases):  VALID_HARD_CASE (adverse surveillance conditions)
```

---

## 9. Conclusion

### **`DATASET_HAS_ISSUES`**

**Rationale**:
The high impostor similarities (0.9226 – 0.9668) are not caused by model confusion or natural biometric overlap between different individuals. They are the direct result of dataset labeling and creation flaws where mirrored and duplicated crops of enrolled individuals were placed in the impostor identity folders.

Because these pseudo-impostors were evaluated as negative pairs, the validation False Positive Rate was severely and falsely inflated at reasonable operating thresholds (0.40–0.70), artificially forcing the threshold selection algorithm to select a high operating point (`0.97`) to achieve $\text{FPR} \le 0.01$.

---

## 10. Recommended Next Steps

1. **Do NOT finalize or adopt 0.97 as the operating threshold.**
2. **Replace the pseudo-impostor images** in `data/evaluation/validation` and `data/evaluation/test` with genuinely distinct, unrelated human subjects from independent capture sources.
3. **Remove duplicate identity directories** (`impostor_07`–`impostor_10`).
4. **Re-run the evaluation protocol** to establish a clean, scientifically defensible threshold curve and operating point.
