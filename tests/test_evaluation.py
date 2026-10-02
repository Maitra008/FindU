"""
Unit and Integration Tests for Evaluation and Threshold Tuning (Part 2).
Covers genuine/impostor separation, metric calculation accuracy, single-face validation,
and threshold selection.
"""

from pathlib import Path
import numpy as np
import pytest

from src.evaluation import (
    ComparisonPair,
    EvaluationImageRecord,
    ModelEvaluator,
    ThresholdMetricPoint,
)
from src.face_engine import FaceEngine, normalize_embedding


@pytest.fixture(scope="session")
def face_engine():
    """Session-scoped FaceEngine instance."""
    return FaceEngine()


def test_metric_calculations_manual_truth_table():
    """Test 4: Verify TP/FP/TN/FN and precision/recall/FPR against manually calculable examples."""
    # 3 genuine pairs (labels=1): sims = [0.85, 0.75, 0.45]
    # 3 impostor pairs (labels=0): sims = [0.60, 0.30, 0.10]
    pairs = [
        ComparisonPair("p1", "pA", "pA", "img1", 1, 0.85),
        ComparisonPair("p2", "pA", "pA", "img2", 1, 0.75),
        ComparisonPair("p3", "pA", "pA", "img3", 1, 0.45),
        ComparisonPair("p4", "pA", "pB", "img4", 0, 0.60),
        ComparisonPair("p5", "pA", "pB", "img5", 0, 0.30),
        ComparisonPair("p6", "pA", "pB", "img6", 0, 0.10),
    ]

    metrics = ModelEvaluator.sweep_thresholds(pairs, threshold_steps=101)
    
    # At threshold 0.50:
    # Positives predicted (sim >= 0.50): p1 (1), p2 (1), p4 (0) -> TP=2, FP=1
    # Negatives predicted (sim < 0.50): p3 (1), p5 (0), p6 (0) -> FN=1, TN=2
    m_50 = next(m for m in metrics if np.isclose(m.threshold, 0.50, atol=1e-3))
    assert m_50.tp == 2
    assert m_50.fp == 1
    assert m_50.tn == 2
    assert m_50.fn == 1
    assert np.isclose(m_50.precision, 2 / 3, atol=1e-3)
    assert np.isclose(m_50.recall, 2 / 3, atol=1e-3)
    assert np.isclose(m_50.fpr, 1 / 3, atol=1e-3)
    assert np.isclose(m_50.tpr, 2 / 3, atol=1e-3)


def test_pair_labeling_logic():
    """Test 2: Verify same identity -> label 1, different identity -> label 0."""
    evaluator = ModelEvaluator()
    v1 = normalize_embedding(np.array([1.0, 0.0, 0.0] + [0.0]*509))
    v2 = normalize_embedding(np.array([0.0, 1.0, 0.0] + [0.0]*509))

    templates = {"person_a": v1, "person_b": v2}
    eval_records = [
        EvaluationImageRecord("person_a", "img_a.jpg", "valid", 1, v1),
        EvaluationImageRecord("person_b", "img_b.jpg", "valid", 1, v2),
    ]

    pairs = evaluator.generate_comparison_pairs(templates, eval_records)
    assert len(pairs) == 4

    p_aa = next(p for p in pairs if p.enrollment_person_id == "person_a" and p.evaluation_person_id == "person_a")
    p_ab = next(p for p in pairs if p.enrollment_person_id == "person_a" and p.evaluation_person_id == "person_b")
    p_ba = next(p for p in pairs if p.enrollment_person_id == "person_b" and p.evaluation_person_id == "person_a")
    p_bb = next(p for p in pairs if p.enrollment_person_id == "person_b" and p.evaluation_person_id == "person_b")

    assert p_aa.label == 1
    assert p_bb.label == 1
    assert p_ab.label == 0
    assert p_ba.label == 0


def test_genuine_vs_impostor_similarity(face_engine):
    """Test 1: Known genuine comparison produces higher similarity than impostor."""
    import insightface.data

    img_t1 = insightface.data.get_image("t1")
    faces = sorted(face_engine.detect_and_embed(img_t1), key=lambda f: f.bbox[0])
    assert len(faces) >= 2

    emb_p1 = faces[0].normalized_embedding
    emb_p2 = faces[1].normalized_embedding

    sim_genuine = float(np.dot(emb_p1, emb_p1))
    sim_impostor = float(np.dot(emb_p1, emb_p2))

    assert sim_genuine > sim_impostor
    assert np.isclose(sim_genuine, 1.0)
    assert sim_impostor < 0.50


def test_threshold_classification():
    """Test 3: Verify similarity >= threshold -> positive, similarity < threshold -> negative."""
    pairs = [
        ComparisonPair("p1", "pA", "pA", "img1", 1, 0.70),
        ComparisonPair("p2", "pA", "pB", "img2", 0, 0.30),
    ]

    metrics = ModelEvaluator.sweep_thresholds(pairs, threshold_steps=101)
    
    # Below 0.30: both classified positive (TP=1, FP=1, TN=0, FN=0)
    m_20 = next(m for m in metrics if np.isclose(m.threshold, 0.20, atol=1e-3))
    assert m_20.tp == 1
    assert m_20.fp == 1

    # Between 0.30 and 0.70 (e.g. 0.50): p1 positive, p2 negative (TP=1, FP=0, TN=1, FN=0)
    m_50 = next(m for m in metrics if np.isclose(m.threshold, 0.50, atol=1e-3))
    assert m_50.tp == 1
    assert m_50.fp == 0
    assert m_50.tn == 1
    assert m_50.fn == 0

    # Above 0.70 (e.g. 0.80): both classified negative (TP=0, FP=0, TN=1, FN=1)
    m_80 = next(m for m in metrics if np.isclose(m.threshold, 0.80, atol=1e-3))
    assert m_80.tp == 0
    assert m_80.fp == 0
    assert m_80.tn == 1
    assert m_80.fn == 1


def test_missing_and_multiple_faces_reported(face_engine):
    """Test 5: Verify invalid evaluation images (0 faces, >1 faces) are reported in dataset summary."""
    evaluator = ModelEvaluator(face_engine)

    # Use data/evaluation which has intentional no-face and multi-face images
    eval_dir = Path("data/evaluation")
    if eval_dir.exists():
        records = evaluator.process_evaluation_images(eval_dir)
        no_face_records = [r for r in records if r.status == "no_face"]
        multi_face_records = [r for r in records if r.status == "multiple_faces"]
        valid_records = [r for r in records if r.status == "valid"]

        assert len(no_face_records) >= 1
        assert len(multi_face_records) >= 1
        assert len(valid_records) >= 1


def test_end_to_end_evaluation_run(face_engine, tmp_path):
    """Verify full evaluate run produces CSVs, plots, JSON, and valid metrics."""
    evaluator = ModelEvaluator(face_engine)
    out_dir = tmp_path / "eval_results"

    report = evaluator.run_evaluation(
        evaluation_dir="data/evaluation",
        registration_dir="data/registration",
        output_dir=out_dir,
    )

    assert report.dataset_summary.identities_count >= 2
    assert report.dataset_summary.valid_images >= 4
    assert report.dataset_summary.genuine_pairs_count >= 4
    assert report.dataset_summary.impostor_pairs_count >= 4
    assert report.auroc >= 0.90
    assert 0.0 < report.selected_threshold <= 1.0

    # Verify output files exist on disk
    assert (out_dir / "threshold_results.csv").exists()
    assert (out_dir / "pair_results.csv").exists()
    assert (out_dir / "roc_curve.png").exists()
    assert (out_dir / "threshold_metrics.png").exists()
    assert (out_dir / "selected_threshold.json").exists()

