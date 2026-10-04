"""
Evaluation and Threshold Tuning Module for Recognition Core (Part 2).
Implements a scientifically defensible protocol:
  Dataset -> Validation/Test Split -> Validation Threshold Sweep ->
  Validation Threshold Selection (under FPR constraint) ->
  Frozen Test Evaluation -> 95% Wilson Confidence Intervals ->
  Comprehensive CSV/JSON/Plot Generation.
"""

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive plotting backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

from src.face_engine import FaceEngine, get_face_engine, normalize_embedding
from src.registration import FaceRegistrar, RegistrationResult

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes / positive trials.
        n: Total number of trials.
        confidence: Confidence level (default: 0.95 -> z ~ 1.96).

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound) in range [0.0, 1.0].
    """
    if n <= 0:
        return 0.0, 0.0, 0.0

    p = k / n
    z = 1.959963984540054  # 95% normal quantile

    denominator = 1.0 + (z**2) / n
    center_adjusted = (p + (z**2) / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))

    lower = max(0.0, center_adjusted - margin)
    upper = min(1.0, center_adjusted + margin)

    return round(p, 4), round(lower, 4), round(upper, 4)


@dataclass
class EvaluationImageRecord:
    """Record of an evaluated image."""
    person_id: str
    image_path: str
    status: str  # 'valid', 'no_face', 'multiple_faces', 'read_error'
    num_faces: int
    embedding: Optional[np.ndarray] = None


@dataclass
class ComparisonPair:
    """Record of a genuine or impostor comparison pair."""
    pair_id: str
    enrollment_person_id: str
    evaluation_person_id: str
    evaluation_image: str
    label: int  # 1 = genuine, 0 = impostor
    similarity: float


@dataclass
class ThresholdMetricPoint:
    """Metrics evaluated at a specific threshold."""
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    fpr: float
    tpr: float
    f1: float


@dataclass
class SimilarityDistributionStats:
    """Statistical summary of similarity scores."""
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float


@dataclass
class TestEvaluationMetrics:
    """Comprehensive metrics on the independent test set at the frozen threshold."""
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    precision_ci_lower: float
    precision_ci_upper: float
    precision_n: int
    recall: float
    recall_ci_lower: float
    recall_ci_upper: float
    recall_n: int
    fpr: float
    fpr_ci_lower: float
    fpr_ci_upper: float
    fpr_n: int
    tpr: float
    f1: float
    auroc: float
    genuine_stats: SimilarityDistributionStats
    impostor_stats: SimilarityDistributionStats


@dataclass
class DatasetPartitionSummary:
    """Dataset partition summary counts."""
    partition_name: str
    total_images: int
    valid_images: int
    no_face_images: int
    multiple_faces_images: int
    read_error_images: int
    identities_count: int
    genuine_pairs_count: int
    impostor_pairs_count: int


@dataclass
class FullEvaluationReport:
    """Complete evaluation report with validation tuning and frozen test evaluation."""
    validation_summary: DatasetPartitionSummary
    test_summary: DatasetPartitionSummary
    selected_threshold: float
    selection_rule: str
    validation_metric_at_selected_threshold: ThresholdMetricPoint
    validation_auroc: float
    test_metrics: TestEvaluationMetrics
    output_files: Dict[str, str]

    @property
    def dataset_summary(self) -> DatasetPartitionSummary:
        return self.validation_summary

    @property
    def auroc(self) -> float:
        return self.validation_auroc


class ModelEvaluator:
    """
    Evaluates FaceEngine recognition performance on separate validation and test sets.
    """

    def __init__(self, face_engine: Optional[FaceEngine] = None):
        """
        Initialize ModelEvaluator.

        Args:
            face_engine: Existing FaceEngine instance or creates a new one.
        """
        self.engine = face_engine or get_face_engine()

    def load_enrollment_templates(
        self,
        registration_dir: Union[str, Path],
        embeddings_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Enroll identities from registration images or pre-computed .npz archives.

        Args:
            registration_dir: Directory containing person subfolders with registration images.
            embeddings_dir: Optional directory with stored .npz embeddings.

        Returns:
            Dict mapping person_id -> normalized identity template embedding (512,).
        """
        reg_dir = Path(registration_dir)
        templates: Dict[str, np.ndarray] = {}

        if embeddings_dir:
            emb_p = Path(embeddings_dir)
            if emb_p.exists():
                for npz_file in emb_p.glob("*.npz"):
                    try:
                        data = np.load(str(npz_file), allow_pickle=True)
                        pid = str(data["person_id"])
                        emb = normalize_embedding(data["embedding"])
                        templates[pid] = emb
                        logger.info("Loaded pre-computed enrollment embedding for '%s'", pid)
                    except Exception as e:
                        logger.warning("Could not read npz %s: %s", npz_file, e)

        if reg_dir.exists():
            registrar = FaceRegistrar(self.engine)
            person_folders = [p for p in reg_dir.iterdir() if p.is_dir()]
            for p_folder in sorted(person_folders):
                person_id = p_folder.name
                if person_id not in templates:
                    logger.info("Enrolling person '%s' from %s", person_id, p_folder)
                    res = registrar.register_person(
                        person_id=person_id,
                        name=person_id.replace("_", " ").title(),
                        image_dir_or_paths=p_folder,
                        output_dir=embeddings_dir or "models/embeddings",
                    )
                    if res is not None:
                        templates[person_id] = res.identity_embedding

        logger.info("Enrolled %d identity template(s) for evaluation.", len(templates))
        return templates

    def process_evaluation_images(
        self,
        dataset_dir: Union[str, Path],
        enrolled_images_set: Optional[set] = None,
    ) -> List[EvaluationImageRecord]:
        """
        Scan evaluation images, detect faces with single-face constraint, and extract embeddings.

        Single-face policy:
        - Exactly 1 face: valid
        - 0 faces: marked as 'no_face'
        - >1 faces: marked as 'multiple_faces'

        Args:
            dataset_dir: Directory containing person subfolders.
            enrolled_images_set: Set of absolute paths used in enrollment to prevent data leakage.

        Returns:
            List of EvaluationImageRecord instances.
        """
        path_dir = Path(dataset_dir)
        if not path_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {path_dir}")

        enrolled_set = enrolled_images_set or set()
        records: List[EvaluationImageRecord] = []
        person_dirs: List[Path] = []
        for d in sorted(path_dir.iterdir()):
            if d.is_dir():
                sub_dirs = [sd for sd in d.iterdir() if sd.is_dir()]
                if sub_dirs:
                    person_dirs.extend(sorted(sub_dirs))
                else:
                    person_dirs.append(d)

        logger.info("Scanning dataset %s: found %d identity folder(s)", path_dir.name, len(person_dirs))

        for p_dir in person_dirs:
            person_id = p_dir.name
            img_files = []
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                img_files.extend(p_dir.glob(f"*{ext}"))
                img_files.extend(p_dir.glob(f"*{ext.upper()}"))

            img_files = sorted(list(set(img_files)))

            for img_file in img_files:
                if str(img_file.resolve()) in enrolled_set:
                    logger.warning("Skipping image '%s': already used in enrollment.", img_file.name)
                    continue

                img = self.engine.load_image(img_file)
                if img is None:
                    records.append(
                        EvaluationImageRecord(
                            person_id=person_id,
                            image_path=str(img_file),
                            status="read_error",
                            num_faces=0,
                        )
                    )
                    continue

                faces = self.engine.detect_and_embed(img)
                num_faces = len(faces)

                if num_faces == 0:
                    records.append(
                        EvaluationImageRecord(
                            person_id=person_id,
                            image_path=str(img_file),
                            status="no_face",
                            num_faces=0,
                        )
                    )
                elif num_faces == 1:
                    norm_emb = normalize_embedding(faces[0].normalized_embedding)
                    records.append(
                        EvaluationImageRecord(
                            person_id=person_id,
                            image_path=str(img_file),
                            status="valid",
                            num_faces=1,
                            embedding=norm_emb,
                        )
                    )
                else:
                    records.append(
                        EvaluationImageRecord(
                            person_id=person_id,
                            image_path=str(img_file),
                            status="multiple_faces",
                            num_faces=num_faces,
                        )
                    )

        return records

    def generate_comparison_pairs(
        self,
        enrollment_templates: Dict[str, np.ndarray],
        evaluation_records: List[EvaluationImageRecord],
    ) -> List[ComparisonPair]:
        """
        Generate all genuine and impostor comparison pairs with cosine similarities.

        Args:
            enrollment_templates: Dict mapping person_id -> enrollment embedding.
            evaluation_records: List of evaluation image records.

        Returns:
            List of ComparisonPair objects.
        """
        valid_records = [r for r in evaluation_records if r.status == "valid" and r.embedding is not None]
        pairs: List[ComparisonPair] = []
        pair_idx = 1

        for enr_id, enr_emb in sorted(enrollment_templates.items()):
            for eval_rec in valid_records:
                eval_id = eval_rec.person_id
                eval_emb = eval_rec.embedding
                if eval_emb is None:
                    continue

                sim = float(np.dot(enr_emb, eval_emb))
                sim = max(-1.0, min(1.0, sim))

                is_genuine = 1 if enr_id == eval_id else 0
                pair_id = f"pair_{pair_idx:06d}"

                pairs.append(
                    ComparisonPair(
                        pair_id=pair_id,
                        enrollment_person_id=enr_id,
                        evaluation_person_id=eval_id,
                        evaluation_image=eval_rec.image_path,
                        label=is_genuine,
                        similarity=sim,
                    )
                )
                pair_idx += 1

        return pairs

    @staticmethod
    def sweep_thresholds(
        pairs: List[ComparisonPair],
        threshold_steps: int = 101,
    ) -> List[ThresholdMetricPoint]:
        """
        Sweep threshold from 0.00 to 1.00 and compute classification metrics at each point.

        Args:
            pairs: List of comparison pairs with ground truth label and similarity.
            threshold_steps: Number of threshold intervals (default: 101 for 0.00..1.00).

        Returns:
            List of ThresholdMetricPoint instances.
        """
        if not pairs:
            return []

        labels = np.array([p.label for p in pairs], dtype=int)
        similarities = np.array([p.similarity for p in pairs], dtype=float)

        thresholds = np.linspace(0.0, 1.0, threshold_steps)
        metrics: List[ThresholdMetricPoint] = []

        total_positives = int(np.sum(labels == 1))
        total_negatives = int(np.sum(labels == 0))

        for t in thresholds:
            thresh_val = round(float(t), 4)
            preds = (similarities >= thresh_val).astype(int)

            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            tn = int(np.sum((preds == 0) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            tpr = recall
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            metrics.append(
                ThresholdMetricPoint(
                    threshold=thresh_val,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    fpr=round(fpr, 4),
                    tpr=round(tpr, 4),
                    f1=round(f1, 4),
                )
            )

        return metrics

    @staticmethod
    def select_threshold_on_validation(
        val_metrics: List[ThresholdMetricPoint],
        target_max_fpr: float = 0.01,
    ) -> Tuple[float, str, ThresholdMetricPoint]:
        """
        Select operating threshold exclusively on the validation set using a predefined rule:
        'Choose the threshold that achieves highest recall on the validation set while keeping FPR <= target_max_fpr'.

        Tie-breaking rule:
        If multiple thresholds achieve identical recall under the FPR constraint:
        1. Select the one with the lowest FPR.
        2. Break any remaining tie by selecting the higher (more conservative) threshold.

        Args:
            val_metrics: List of swept threshold metrics on validation set.
            target_max_fpr: Maximum acceptable false positive rate target (default: 0.01).

        Returns:
            Tuple of (selected_threshold, selection_rule_description, metric_point_at_threshold).
        """
        if not val_metrics:
            return 0.40, "default_fallback", ThresholdMetricPoint(0.40, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # 1. Filter candidates satisfying FPR <= target_max_fpr with at least 1 true positive
        candidates = [m for m in val_metrics if m.fpr <= target_max_fpr and m.tp > 0]

        if candidates:
            # Rank by: highest recall -> lowest fpr -> highest threshold
            best_point = max(
                candidates,
                key=lambda m: (m.recall, -m.fpr, m.threshold),
            )
            rule_desc = f"highest recall on validation set subject to FPR <= {target_max_fpr:.2f}"
        else:
            # Fallback if no thresholds have FPR <= target_max_fpr: minimize FPR, then maximize F1
            best_point = min(
                val_metrics,
                key=lambda m: (m.fpr, -m.f1, -m.threshold),
            )
            rule_desc = f"minimum FPR fallback (target FPR <= {target_max_fpr:.2f} could not be satisfied on validation set)"

        return best_point.threshold, rule_desc, best_point

    @staticmethod
    def evaluate_frozen_threshold(
        test_pairs: List[ComparisonPair],
        frozen_threshold: float,
    ) -> TestEvaluationMetrics:
        """
        Evaluate performance of the frozen threshold on the independent test set,
        computing Wilson score 95% confidence intervals and score distribution statistics.

        Args:
            test_pairs: List of comparison pairs from the test set.
            frozen_threshold: Operating threshold selected during validation.

        Returns:
            TestEvaluationMetrics instance.
        """
        if not test_pairs:
            raise ValueError("Test pairs list is empty.")

        labels = np.array([p.label for p in test_pairs], dtype=int)
        similarities = np.array([p.similarity for p in test_pairs], dtype=float)

        preds = (similarities >= frozen_threshold).astype(int)

        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        n_prec = tp + fp
        n_rec = tp + fn
        n_fpr = fp + tn

        prec, prec_l, prec_u = compute_wilson_ci(tp, n_prec)
        rec, rec_l, rec_u = compute_wilson_ci(tp, n_rec)
        fpr, fpr_l, fpr_u = compute_wilson_ci(fp, n_fpr)

        tpr = rec
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Similarity distributions
        genuine_sims = similarities[labels == 1]
        impostor_sims = similarities[labels == 0]

        def get_dist_stats(arr: np.ndarray) -> SimilarityDistributionStats:
            if len(arr) == 0:
                return SimilarityDistributionStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return SimilarityDistributionStats(
                count=int(len(arr)),
                min=round(float(np.min(arr)), 4),
                max=round(float(np.max(arr)), 4),
                mean=round(float(np.mean(arr)), 4),
                median=round(float(np.median(arr)), 4),
                std=round(float(np.std(arr)), 4),
            )

        gen_stats = get_dist_stats(genuine_sims)
        imp_stats = get_dist_stats(impostor_sims)

        if len(set(labels)) > 1:
            auroc_val = float(roc_auc_score(labels, similarities))
        else:
            auroc_val = 0.0

        return TestEvaluationMetrics(
            threshold=frozen_threshold,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=prec,
            precision_ci_lower=prec_l,
            precision_ci_upper=prec_u,
            precision_n=n_prec,
            recall=rec,
            recall_ci_lower=rec_l,
            recall_ci_upper=rec_u,
            recall_n=n_rec,
            fpr=fpr,
            fpr_ci_lower=fpr_l,
            fpr_ci_upper=fpr_u,
            fpr_n=n_fpr,
            tpr=tpr,
            f1=round(f1, 4),
            auroc=round(auroc_val, 4),
            genuine_stats=gen_stats,
            impostor_stats=imp_stats,
        )

    def run_full_evaluation(
        self,
        dataset_dir: Union[str, Path] = Path("data/evaluation"),
        validation_dir: Optional[Union[str, Path]] = None,
        test_dir: Optional[Union[str, Path]] = None,
        registration_dir: Union[str, Path] = Path("data/registration"),
        output_dir: Union[str, Path] = Path("data/evaluation/results"),
        target_max_fpr: float = 0.01,
        evaluation_dir: Optional[Union[str, Path]] = None,
    ) -> FullEvaluationReport:
        """
        Execute the complete scientific evaluation pipeline:
        1. Enroll identities from registration set.
        2. Evaluate validation set -> sweep thresholds -> select operating threshold.
        3. Freeze threshold.
        4. Evaluate independent test set -> compute 95% Wilson CIs and distributions.
        5. Generate ROC plots, metric plots, and JSON/CSV artifacts.

        Args:
            dataset_dir: Base evaluation dataset path.
            validation_dir: Optional override for validation directory.
            test_dir: Optional override for test directory.
            registration_dir: Path to enrollment registration images.
            output_dir: Path to write result artifacts.
            target_max_fpr: Upper bound FPR for threshold selection (default: 0.01).
            evaluation_dir: Optional alias for dataset_dir.

        Returns:
            FullEvaluationReport instance.
        """
        if evaluation_dir is not None:
            dataset_dir = evaluation_dir

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        base_eval_p = Path(dataset_dir)
        val_p = Path(validation_dir) if validation_dir else (base_eval_p / "validation" if (base_eval_p / "validation").exists() else base_eval_p)
        test_p = Path(test_dir) if test_dir else (base_eval_p / "test" if (base_eval_p / "test").exists() else base_eval_p)

        # 1. Enroll templates
        templates = self.load_enrollment_templates(registration_dir)
        if not templates:
            raise ValueError(f"No enrollment templates found or generated from {registration_dir}")

        enrolled_images_set = set()
        reg_p = Path(registration_dir)
        if reg_p.exists():
            for p_file in reg_p.rglob("*"):
                if p_file.is_file():
                    enrolled_images_set.add(str(p_file.resolve()))

        # 2. Process Validation Dataset
        val_records = self.process_evaluation_images(val_p, enrolled_images_set)
        val_pairs = self.generate_comparison_pairs(templates, val_records)

        val_summary = DatasetPartitionSummary(
            partition_name="validation",
            total_images=len(val_records),
            valid_images=sum(1 for r in val_records if r.status == "valid"),
            no_face_images=sum(1 for r in val_records if r.status == "no_face"),
            multiple_faces_images=sum(1 for r in val_records if r.status == "multiple_faces"),
            read_error_images=sum(1 for r in val_records if r.status == "read_error"),
            identities_count=len(templates),
            genuine_pairs_count=sum(1 for p in val_pairs if p.label == 1),
            impostor_pairs_count=sum(1 for p in val_pairs if p.label == 0),
        )

        # 3. Sweep Thresholds on Validation Set
        val_metric_points = self.sweep_thresholds(val_pairs, threshold_steps=101)

        val_labels = [p.label for p in val_pairs]
        val_sims = [p.similarity for p in val_pairs]
        val_auroc = float(roc_auc_score(val_labels, val_sims)) if len(set(val_labels)) > 1 else 0.0

        # 4. Select Threshold on Validation Set
        selected_thresh, selection_rule, val_best_metric = self.select_threshold_on_validation(
            val_metric_points, target_max_fpr=target_max_fpr
        )
        logger.info("Selected validation threshold: %.2f via rule: %s", selected_thresh, selection_rule)

        # 5. Process Independent Test Dataset
        test_records = self.process_evaluation_images(test_p, enrolled_images_set)
        test_pairs = self.generate_comparison_pairs(templates, test_records)

        test_summary = DatasetPartitionSummary(
            partition_name="test",
            total_images=len(test_records),
            valid_images=sum(1 for r in test_records if r.status == "valid"),
            no_face_images=sum(1 for r in test_records if r.status == "no_face"),
            multiple_faces_images=sum(1 for r in test_records if r.status == "multiple_faces"),
            read_error_images=sum(1 for r in test_records if r.status == "read_error"),
            identities_count=len(templates),
            genuine_pairs_count=sum(1 for p in test_pairs if p.label == 1),
            impostor_pairs_count=sum(1 for p in test_pairs if p.label == 0),
        )

        # 6. Evaluate FROZEN Threshold on Test Set
        test_metrics = self.evaluate_frozen_threshold(test_pairs, selected_thresh)

        # 7. Save Output Files
        val_thresh_csv = out_p / "validation_threshold_results.csv"
        self._save_threshold_results_csv(val_metric_points, val_thresh_csv)
        self._save_threshold_results_csv(val_metric_points, out_p / "threshold_results.csv")

        val_pair_csv = out_p / "validation_pair_results.csv"
        self._save_pair_results_csv(val_pairs, val_pair_csv)
        self._save_pair_results_csv(val_pairs, out_p / "pair_results.csv")

        test_pair_csv = out_p / "test_pair_results.csv"
        self._save_pair_results_csv(test_pairs, test_pair_csv)

        roc_png = out_p / "roc_curve.png"
        test_labels = [p.label for p in test_pairs]
        test_sims = [p.similarity for p in test_pairs]
        self._plot_multi_roc_curve(val_labels, val_sims, val_auroc, test_labels, test_sims, test_metrics.auroc, roc_png)

        metrics_png = out_p / "threshold_metrics.png"
        self._plot_threshold_metrics(val_metric_points, selected_thresh, metrics_png)

        sel_json = out_p / "selected_threshold.json"
        self._save_selected_threshold_json(
            selected_threshold=selected_thresh,
            selection_rule=selection_rule,
            val_metric=val_best_metric,
            val_summary=val_summary,
            output_path=sel_json,
        )

        test_json = out_p / "test_metrics.json"
        self._save_test_metrics_json(test_metrics, test_summary, test_json)

        summary_json = out_p / "evaluation_summary.json"
        self._save_evaluation_summary_json(
            val_summary=val_summary,
            test_summary=test_summary,
            selected_threshold=selected_thresh,
            selection_rule=selection_rule,
            val_best_metric=val_best_metric,
            val_auroc=val_auroc,
            test_metrics=test_metrics,
            output_path=summary_json,
        )

        output_files = {
            "validation_threshold_results_csv": str(val_thresh_csv),
            "validation_pair_results_csv": str(val_pair_csv),
            "test_pair_results_csv": str(test_pair_csv),
            "selected_threshold_json": str(sel_json),
            "test_metrics_json": str(test_json),
            "evaluation_summary_json": str(summary_json),
            "roc_curve_png": str(roc_png),
            "threshold_metrics_png": str(metrics_png),
        }

        return FullEvaluationReport(
            validation_summary=val_summary,
            test_summary=test_summary,
            selected_threshold=selected_thresh,
            selection_rule=selection_rule,
            validation_metric_at_selected_threshold=val_best_metric,
            validation_auroc=val_auroc,
            test_metrics=test_metrics,
            output_files=output_files,
        )

    # Backwards compatibility alias
    run_evaluation = run_full_evaluation

    @staticmethod
    def _save_pair_results_csv(pairs: List[ComparisonPair], path: Path) -> None:
        """Save individual comparison pair similarities to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "enrollment_person_id", "evaluation_person_id", "label", "similarity"])
            for p in pairs:
                writer.writerow([p.pair_id, p.enrollment_person_id, p.evaluation_person_id, p.label, f"{p.similarity:.4f}"])

    @staticmethod
    def _save_threshold_results_csv(metrics: List[ThresholdMetricPoint], path: Path) -> None:
        """Save threshold sweep results to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "tp", "fp", "tn", "fn", "precision", "recall", "fpr", "tpr", "f1"])
            for m in metrics:
                writer.writerow([
                    f"{m.threshold:.2f}",
                    m.tp,
                    m.fp,
                    m.tn,
                    m.fn,
                    f"{m.precision:.4f}",
                    f"{m.recall:.4f}",
                    f"{m.fpr:.4f}",
                    f"{m.tpr:.4f}",
                    f"{m.f1:.4f}",
                ])

    @staticmethod
    def _save_selected_threshold_json(
        selected_threshold: float,
        selection_rule: str,
        val_metric: ThresholdMetricPoint,
        val_summary: DatasetPartitionSummary,
        output_path: Path,
    ) -> None:
        """Save selected threshold and validation metadata to JSON."""
        data = {
            "threshold": selected_threshold,
            "selection_dataset": "validation",
            "selection_rule": selection_rule,
            "validation_metrics": {
                "precision": val_metric.precision,
                "recall": val_metric.recall,
                "fpr": val_metric.fpr,
                "f1_score": val_metric.f1,
                "tp": val_metric.tp,
                "fp": val_metric.fp,
                "tn": val_metric.tn,
                "fn": val_metric.fn,
            },
            "validation_dataset_summary": asdict(val_summary),
            "disclaimer": "Threshold selected on validation set and frozen before test evaluation.",
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _save_test_metrics_json(test_metrics: TestEvaluationMetrics, test_summary: DatasetPartitionSummary, output_path: Path) -> None:
        """Save frozen test metrics with Wilson confidence intervals to JSON."""
        data = {
            "evaluation_dataset": "independent_test_set",
            "frozen_threshold": test_metrics.threshold,
            "test_dataset_summary": asdict(test_summary),
            "classification_counts": {
                "tp": test_metrics.tp,
                "fp": test_metrics.fp,
                "tn": test_metrics.tn,
                "fn": test_metrics.fn,
            },
            "metrics_with_95_percent_confidence_intervals": {
                "precision": {
                    "value": test_metrics.precision,
                    "ci_lower_95": test_metrics.precision_ci_lower,
                    "ci_upper_95": test_metrics.precision_ci_upper,
                    "n_trials": test_metrics.precision_n,
                },
                "recall": {
                    "value": test_metrics.recall,
                    "ci_lower_95": test_metrics.recall_ci_lower,
                    "ci_upper_95": test_metrics.recall_ci_upper,
                    "n_trials": test_metrics.recall_n,
                },
                "false_positive_rate": {
                    "value": test_metrics.fpr,
                    "ci_lower_95": test_metrics.fpr_ci_lower,
                    "ci_upper_95": test_metrics.fpr_ci_upper,
                    "n_trials": test_metrics.fpr_n,
                },
                "f1_score": test_metrics.f1,
                "auroc": test_metrics.auroc,
            },
            "similarity_distributions": {
                "genuine": asdict(test_metrics.genuine_stats),
                "impostor": asdict(test_metrics.impostor_stats),
            },
            "disclaimer": "Evaluated on independent test set using the frozen threshold selected from validation data.",
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _save_evaluation_summary_json(
        val_summary: DatasetPartitionSummary,
        test_summary: DatasetPartitionSummary,
        selected_threshold: float,
        selection_rule: str,
        val_best_metric: ThresholdMetricPoint,
        val_auroc: float,
        test_metrics: TestEvaluationMetrics,
        output_path: Path,
    ) -> None:
        """Save end-to-end evaluation summary to JSON."""
        data = {
            "evaluation_protocol": "Validation Tuning -> Threshold Freeze -> Independent Test Evaluation",
            "dataset": {
                "enrolled_identities": val_summary.identities_count,
                "validation_images_total": val_summary.total_images,
                "validation_images_valid": val_summary.valid_images,
                "validation_genuine_pairs": val_summary.genuine_pairs_count,
                "validation_impostor_pairs": val_summary.impostor_pairs_count,
                "test_images_total": test_summary.total_images,
                "test_images_valid": test_summary.valid_images,
                "test_genuine_pairs": test_summary.genuine_pairs_count,
                "test_impostor_pairs": test_summary.impostor_pairs_count,
            },
            "threshold_selection": {
                "rule": selection_rule,
                "selected_threshold": selected_threshold,
                "validation_auroc": round(val_auroc, 4),
                "validation_recall": val_best_metric.recall,
                "validation_fpr": val_best_metric.fpr,
                "validation_precision": val_best_metric.precision,
                "validation_f1": val_best_metric.f1,
            },
            "frozen_test_evaluation": {
                "frozen_threshold": test_metrics.threshold,
                "tp": test_metrics.tp,
                "fp": test_metrics.fp,
                "tn": test_metrics.tn,
                "fn": test_metrics.fn,
                "precision": f"{test_metrics.precision*100:.1f}% (95% CI: {test_metrics.precision_ci_lower*100:.1f}%-{test_metrics.precision_ci_upper*100:.1f}%), n={test_metrics.precision_n}",
                "recall": f"{test_metrics.recall*100:.1f}% (95% CI: {test_metrics.recall_ci_lower*100:.1f}%-{test_metrics.recall_ci_upper*100:.1f}%), n={test_metrics.recall_n}",
                "false_positive_rate": f"{test_metrics.fpr*100:.1f}% (95% CI: {test_metrics.fpr_ci_lower*100:.1f}%-{test_metrics.fpr_ci_upper*100:.1f}%), n={test_metrics.fpr_n}",
                "f1_score": test_metrics.f1,
                "auroc": test_metrics.auroc,
                "similarity_stats": {
                    "genuine_mean": test_metrics.genuine_stats.mean,
                    "genuine_min": test_metrics.genuine_stats.min,
                    "genuine_max": test_metrics.genuine_stats.max,
                    "impostor_mean": test_metrics.impostor_stats.mean,
                    "impostor_min": test_metrics.impostor_stats.min,
                    "impostor_max": test_metrics.impostor_stats.max,
                },
            },
            "limitations": [
                "Controlled evaluation dataset under synthetic condition variations.",
                "Real CCTV surveillance footage will experience wider pose, distance, and motion degradation.",
                "FPR uncertainty is quantified via 95% Wilson confidence intervals.",
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _plot_multi_roc_curve(
        val_labels: List[int],
        val_sims: List[float],
        val_auroc: float,
        test_labels: List[int],
        test_sims: List[float],
        test_auroc: float,
        output_path: Path,
    ) -> None:
        """Plot Validation and Test ROC curves on a single figure."""
        plt.figure(figsize=(7, 6))

        if len(set(val_labels)) > 1:
            fpr_v, tpr_v, _ = roc_curve(val_labels, val_sims)
            plt.plot(fpr_v, tpr_v, color="#1f77b4", lw=2.2, label=f"Validation ROC (AUROC = {val_auroc:.4f})")

        if len(set(test_labels)) > 1:
            fpr_t, tpr_t, _ = roc_curve(test_labels, test_sims)
            plt.plot(fpr_t, tpr_t, color="#2ca02c", lw=2.2, linestyle="--", label=f"Test ROC (AUROC = {test_auroc:.4f})")

        plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle=":", lw=1.5, label="No-Skill Reference (AUC = 0.50)")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate (FPR)", fontsize=11)
        plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_threshold_metrics(metrics: List[ThresholdMetricPoint], selected_threshold: float, output_path: Path) -> None:
        """Plot Validation threshold metrics curves with selected threshold marker."""
        thresholds = [m.threshold for m in metrics]
        precisions = [m.precision for m in metrics]
        recalls = [m.recall for m in metrics]
        fprs = [m.fpr for m in metrics]
        f1s = [m.f1 for m in metrics]

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precisions, label="Precision (Val)", color="#2ca02c", lw=2)
        plt.plot(thresholds, recalls, label="Recall (TPR, Val)", color="#1f77b4", lw=2)
        plt.plot(thresholds, fprs, label="False Positive Rate (Val)", color="#d62728", lw=2)
        plt.plot(thresholds, f1s, label="F1 Score (Val)", color="#9467bd", lw=1.8, linestyle="--")

        plt.axvline(
            x=selected_threshold,
            color="#333333",
            linestyle=":",
            lw=2.5,
            label=f"Frozen Selected Threshold ({selected_threshold:.2f})",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([-0.02, 1.05])
        plt.xlabel("Similarity Threshold", fontsize=11)
        plt.ylabel("Metric Score", fontsize=11)
        plt.title("Validation Metric Sweep & Frozen Threshold Selection", fontsize=13, fontweight="bold")
        plt.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()


def print_full_evaluation_report(report: FullEvaluationReport) -> None:
    """Print clean terminal report as specified in requirements."""
    vs = report.validation_summary
    ts = report.test_summary
    tm = report.test_metrics

    print()
    print("========================================")
    print("RECOGNITION EVALUATION REPORT")
    print("Protocol: Validation Selection -> Frozen Test Evaluation")
    print("========================================")
    print()
    print("--- 1. DATASET PARTITIONS ---")
    print(f"Enrolled Identities: {vs.identities_count}")
    print(f"Validation Partition: {vs.total_images} images (Valid: {vs.valid_images}, No-face: {vs.no_face_images}, Multi-face: {vs.multiple_faces_images})")
    print(f"  -> Validation Pairs: {vs.genuine_pairs_count} genuine, {vs.impostor_pairs_count} impostor")
    print(f"Test Partition:       {ts.total_images} images (Valid: {ts.valid_images}, No-face: {ts.no_face_images}, Multi-face: {ts.multiple_faces_images})")
    print(f"  -> Test Pairs:       {ts.genuine_pairs_count} genuine, {ts.impostor_pairs_count} impostor")
    print()
    print("--- 2. THRESHOLD SELECTION (ON VALIDATION SET) ---")
    print(f"Selection Rule:      {report.selection_rule}")
    print(f"Validation AUROC:    {report.validation_auroc:.4f}")
    print(f"Selected Threshold:  {report.selected_threshold:.2f} (FROZEN)")
    print(f"  -> Val Recall:     {report.validation_metric_at_selected_threshold.recall*100:.1f}%")
    print(f"  -> Val FPR:        {report.validation_metric_at_selected_threshold.fpr*100:.1f}%")
    print(f"  -> Val Precision:  {report.validation_metric_at_selected_threshold.precision*100:.1f}%")
    print()
    print("--- 3. FROZEN TEST EVALUATION (INDEPENDENT TEST SET) ---")
    print(f"Operating Threshold: {tm.threshold:.2f}")
    print(f"Confusion Matrix:    TP={tm.tp}, FP={tm.fp}, TN={tm.tn}, FN={tm.fn}")
    print(f"Recall (TPR):        {tm.recall*100:.1f}% (95% CI: {tm.recall_ci_lower*100:.1f}%-{tm.recall_ci_upper*100:.1f}%), n={tm.recall_n} genuine pairs")
    print(f"False Positive Rate: {tm.fpr*100:.1f}% (95% CI: {tm.fpr_ci_lower*100:.1f}%-{tm.fpr_ci_upper*100:.1f}%), n={tm.fpr_n} impostor pairs")
    print(f"Precision:           {tm.precision*100:.1f}% (95% CI: {tm.precision_ci_lower*100:.1f}%-{tm.precision_ci_upper*100:.1f}%), n={tm.precision_n} positive alerts")
    print(f"F1-Score:            {tm.f1*100:.1f}%")
    print(f"Test Set AUROC:      {tm.auroc:.4f}")
    print()
    print("--- 4. SIMILARITY DISTRIBUTIONS (TEST SET) ---")
    print(f"Genuine Pairs:   min={tm.genuine_stats.min:.4f}, max={tm.genuine_stats.max:.4f}, mean={tm.genuine_stats.mean:.4f}, median={tm.genuine_stats.median:.4f}")
    print(f"Impostor Pairs:  min={tm.impostor_stats.min:.4f}, max={tm.impostor_stats.max:.4f}, mean={tm.impostor_stats.mean:.4f}, median={tm.impostor_stats.median:.4f}")
    print()
    print("--- 5. GENERATED ARTIFACTS ---")
    for name, fpath in report.output_files.items():
        print(f"  {fpath}")
    print("========================================")
    print()


# Backwards compatibility alias
print_evaluation_report = print_full_evaluation_report
