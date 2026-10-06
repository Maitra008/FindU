"""
FindU Stage 2 Evaluation Subject Collection Tool.

Allows collecting and organizing evaluation subjects strictly within:
    data/evaluation/stage2/calibration/<subject_id>/
    data/evaluation/stage2/validation/<subject_id>/

Without modifying or touching the production database or production FAISS index.

Each subject structure:
    <subject_id>/
        manifest.json
        reference/
            ref_01.jpg .. ref_06.jpg (High-quality reference portrait photos)
        probes/
            probe_normal.jpg
            probe_distance_lowres.jpg
            probe_yaw_left.jpg
            probe_yaw_right.jpg
            probe_lowlight.jpg
            probe_motion.jpg
            probe_glasses.jpg (optional)
            probe_webcam_*.jpg (additional webcam frames)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2_DIR = PROJECT_ROOT / "data" / "evaluation" / "stage2"


def create_subject_directory(subject_id: str, split: str = "calibration") -> Path:
    """Create directory structure for an evaluation subject."""
    base = STAGE2_DIR / split / subject_id
    (base / "reference").mkdir(parents=True, exist_ok=True)
    (base / "probes").mkdir(parents=True, exist_ok=True)
    return base


def populate_from_existing_assets():
    """
    Populate Stage 2 Calibration and Validation folders using disjoint real-human subjects
    from existing repository assets.
    """
    print("=" * 70)
    print("FINDU — POPULATING STAGE 2 EVALUATION DATASET FROM REAL HUMAN ASSETS")
    print("=" * 70)

    # Clean / ensure directories exist
    calib_dir = STAGE2_DIR / "calibration"
    val_dir = STAGE2_DIR / "validation"
    calib_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. CALIBRATION SUBJECTS (Disjoint Subject Set A)
    # -------------------------------------------------------------
    # Subject 1: Rahul (Developer live webcam + high-res reference)
    rahul_base = create_subject_directory("subject_rahul", "calibration")
    rahul_refs = sorted(list((PROJECT_ROOT / "data" / "reference_photos" / "rahul_05702c").glob("*.jpg")))
    for idx, r in enumerate(rahul_refs, 1):
        shutil.copy2(r, rahul_base / "reference" / f"ref_{idx:02d}.jpg")

    rahul_probes = sorted(list((PROJECT_ROOT / "data" / "diagnostics" / "webcam_embedding").glob("frame_*.jpg")))
    for idx, p in enumerate(rahul_probes, 1):
        shutil.copy2(p, rahul_base / "probes" / f"probe_webcam_{idx:02d}.jpg")

    with open(rahul_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "subject_rahul",
            "split": "calibration",
            "source": "live_webcam_and_portrait",
            "reference_count": len(rahul_refs),
            "probe_count": len(rahul_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Calibration] Added subject_rahul: {len(rahul_refs)} refs, {len(rahul_probes)} probes")

    # Subject 2: Person 01
    p1_base = create_subject_directory("subject_person_01", "calibration")
    p1_refs = sorted(list((PROJECT_ROOT / "data" / "registration" / "person_01").glob("*.jpg")))
    for idx, r in enumerate(p1_refs, 1):
        shutil.copy2(r, p1_base / "reference" / f"ref_{idx:02d}.jpg")
    
    p1_test_probes = list((PROJECT_ROOT / "data" / "evaluation" / "test" / "person_01").glob("*.jpg"))
    p1_val_probes = list((PROJECT_ROOT / "data" / "evaluation" / "validation" / "person_01").glob("*.jpg"))
    for p in (p1_test_probes + p1_val_probes):
        shutil.copy2(p, p1_base / "probes" / p.name)
    
    with open(p1_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "subject_person_01",
            "split": "calibration",
            "source": "multi_condition_benchmark",
            "reference_count": len(p1_refs),
            "probe_count": len(p1_test_probes) + len(p1_val_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Calibration] Added subject_person_01: {len(p1_refs)} refs, {len(p1_test_probes) + len(p1_val_probes)} probes")

    # Subject 3: Person 02
    p2_base = create_subject_directory("subject_person_02", "calibration")
    p2_refs = sorted(list((PROJECT_ROOT / "data" / "registration" / "person_02").glob("*.jpg")))
    for idx, r in enumerate(p2_refs, 1):
        shutil.copy2(r, p2_base / "reference" / f"ref_{idx:02d}.jpg")
    
    p2_test_probes = list((PROJECT_ROOT / "data" / "evaluation" / "test" / "person_02").glob("*.jpg"))
    p2_val_probes = list((PROJECT_ROOT / "data" / "evaluation" / "validation" / "person_02").glob("*.jpg"))
    for p in (p2_test_probes + p2_val_probes):
        shutil.copy2(p, p2_base / "probes" / p.name)
    
    with open(p2_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "subject_person_02",
            "split": "calibration",
            "source": "multi_condition_benchmark",
            "reference_count": len(p2_refs),
            "probe_count": len(p2_test_probes) + len(p2_val_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Calibration] Added subject_person_02: {len(p2_refs)} refs, {len(p2_test_probes) + len(p2_val_probes)} probes")

    # Subject 4: Person 03
    p3_base = create_subject_directory("subject_person_03", "calibration")
    p3_refs = sorted(list((PROJECT_ROOT / "data" / "registration" / "person_03").glob("*.jpg")))
    for idx, r in enumerate(p3_refs, 1):
        shutil.copy2(r, p3_base / "reference" / f"ref_{idx:02d}.jpg")
    
    p3_test_probes = list((PROJECT_ROOT / "data" / "evaluation" / "test" / "person_03").glob("*.jpg"))
    p3_val_probes = list((PROJECT_ROOT / "data" / "evaluation" / "validation" / "person_03").glob("*.jpg"))
    for p in (p3_test_probes + p3_val_probes):
        shutil.copy2(p, p3_base / "probes" / p.name)
    
    with open(p3_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "subject_person_03",
            "split": "calibration",
            "source": "multi_condition_benchmark",
            "reference_count": len(p3_refs),
            "probe_count": len(p3_test_probes) + len(p3_val_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Calibration] Added subject_person_03: {len(p3_refs)} refs, {len(p3_test_probes) + len(p3_val_probes)} probes")

    # -------------------------------------------------------------
    # 2. VALIDATION SUBJECTS (Completely Unseen Disjoint Set B)
    # -------------------------------------------------------------
    # Subject 5: Person 04
    p4_base = create_subject_directory("subject_person_04", "validation")
    p4_refs = sorted(list((PROJECT_ROOT / "data" / "registration" / "person_04").glob("*.jpg")))
    for idx, r in enumerate(p4_refs, 1):
        shutil.copy2(r, p4_base / "reference" / f"ref_{idx:02d}.jpg")
    
    p4_test_probes = list((PROJECT_ROOT / "data" / "evaluation" / "test" / "person_04").glob("*.jpg"))
    p4_val_probes = list((PROJECT_ROOT / "data" / "evaluation" / "validation" / "person_04").glob("*.jpg"))
    for p in (p4_test_probes + p4_val_probes):
        shutil.copy2(p, p4_base / "probes" / p.name)
    
    with open(p4_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "subject_person_04",
            "split": "validation",
            "source": "multi_condition_benchmark",
            "reference_count": len(p4_refs),
            "probe_count": len(p4_test_probes) + len(p4_val_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Validation] Added subject_person_04: {len(p4_refs)} refs, {len(p4_test_probes) + len(p4_val_probes)} probes")

    # Subject 6: Person 05
    p5_base = create_subject_directory("subject_person_05", "validation")
    p5_refs = sorted(list((PROJECT_ROOT / "data" / "registration" / "person_05").glob("*.jpg")))
    for idx, r in enumerate(p5_refs, 1):
        shutil.copy2(r, p5_base / "reference" / f"ref_{idx:02d}.jpg")
    
    p5_test_probes = list((PROJECT_ROOT / "data" / "evaluation" / "test" / "person_05").glob("*.jpg"))
    p5_val_probes = list((PROJECT_ROOT / "data" / "evaluation" / "validation" / "person_05").glob("*.jpg"))
    for p in (p5_test_probes + p5_val_probes):
        shutil.copy2(p, p5_base / "probes" / p.name)
    
    with open(p5_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "subject_person_05",
            "split": "validation",
            "source": "multi_condition_benchmark",
            "reference_count": len(p5_refs),
            "probe_count": len(p5_test_probes) + len(p5_val_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Validation] Added subject_person_05: {len(p5_refs)} refs, {len(p5_test_probes) + len(p5_val_probes)} probes")

    # Subject 7: Unseen Dedicated Impostor 01 (Probe-only non-gallery subject)
    imp1_base = create_subject_directory("impostor_unseen_01", "validation")
    imp1_probes = list((PROJECT_ROOT / "data" / "evaluation" / "test" / "impostor_01").glob("*.jpg")) + \
                  list((PROJECT_ROOT / "data" / "evaluation" / "validation" / "impostor_01").glob("*.jpg"))
    for p in imp1_probes:
        shutil.copy2(p, imp1_base / "probes" / f"imp_{p.stem}_{p.name}")
    with open(imp1_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "impostor_unseen_01",
            "split": "validation",
            "source": "dedicated_negative_probe",
            "reference_count": 0,
            "probe_count": len(imp1_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Validation] Added impostor_unseen_01: 0 refs (unregistered), {len(imp1_probes)} probes")

    # Subject 8: Unseen Dedicated Impostor 02 (Probe-only non-gallery subject)
    imp2_base = create_subject_directory("impostor_unseen_02", "validation")
    imp2_probes = list((PROJECT_ROOT / "data" / "evaluation" / "test" / "impostor_02").glob("*.jpg")) + \
                  list((PROJECT_ROOT / "data" / "evaluation" / "validation" / "impostor_02").glob("*.jpg"))
    for p in imp2_probes:
        shutil.copy2(p, imp2_base / "probes" / f"imp_{p.stem}_{p.name}")
    with open(imp2_base / "manifest.json", "w") as f:
        json.dump({
            "subject_id": "impostor_unseen_02",
            "split": "validation",
            "source": "dedicated_negative_probe",
            "reference_count": 0,
            "probe_count": len(imp2_probes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[Validation] Added impostor_unseen_02: 0 refs (unregistered), {len(imp2_probes)} probes")

    print("\n[SUCCESS] Stage 2 evaluation dataset populated with disjoint subject partitioning.")


def capture_subject_interactive(subject_id: str, split: str = "validation"):
    """
    Interactive live webcam collector to capture a new subject in real-time.
    """
    print(f"\n--- COLLECTING NEW SUBJECT: {subject_id} ({split.upper()}) ---")
    subject_dir = create_subject_directory(subject_id, split)
    ref_dir = subject_dir / "reference"
    probes_dir = subject_dir / "probes"

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # Warmup
    for _ in range(10):
        cap.read()

    print("\n1. Capturing 4 High-Quality Reference Portraits...")
    for i in range(1, 5):
        input(f"  Press ENTER to capture Reference Portrait #{i} (Face camera directly in good lighting)...")
        ret, frame = cap.read()
        if ret and frame is not None:
            save_path = ref_dir / f"ref_{i:02d}.jpg"
            cv2.imwrite(str(save_path), frame)
            print(f"  Saved {save_path.name}")

    print("\n2. Capturing Multi-Condition Evaluation Probes...")
    probe_prompts = [
        ("probe_normal.jpg", "Normal frontal webcam view"),
        ("probe_distance_lowres.jpg", "Step back 2-3 meters (distance / low-res)"),
        ("probe_yaw_left.jpg", "Turn head 20-30 degrees LEFT"),
        ("probe_yaw_right.jpg", "Turn head 20-30 degrees RIGHT"),
        ("probe_lowlight.jpg", "Dim/lower the lighting if possible"),
        ("probe_motion.jpg", "Move head slowly while capturing (motion blur)"),
        ("probe_glasses.jpg", "Optional: Put on glasses / eyewear (or press enter for normal)")
    ]

    for fname, desc in probe_prompts:
        input(f"  Condition: {desc} -> Press ENTER to capture...")
        ret, frame = cap.read()
        if ret and frame is not None:
            save_path = probes_dir / fname
            cv2.imwrite(str(save_path), frame)
            print(f"  Saved {save_path.name}")

    cap.release()

    manifest = {
        "subject_id": subject_id,
        "split": split,
        "source": "live_interactive_collector",
        "reference_count": len(list(ref_dir.glob("*.jpg"))),
        "probe_count": len(list(probes_dir.glob("*.jpg"))),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(subject_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSubject '{subject_id}' successfully saved to {subject_dir}")


def main():
    parser = argparse.ArgumentParser(description="FindU Stage 2 Evaluation Subject Collector")
    parser.add_argument("--populate-existing", action="store_true", help="Populate Stage 2 dataset from existing real assets")
    parser.add_argument("--interactive", action="store_true", help="Capture a new real subject interactively via webcam")
    parser.add_argument("--subject-id", type=str, default=None, help="Subject ID for new collection")
    parser.add_argument("--split", type=str, choices=["calibration", "validation"], default="validation", help="Dataset split")

    args = parser.parse_args()

    if args.populate_existing or (not args.interactive and args.subject_id is None):
        populate_from_existing_assets()
    elif args.interactive:
        s_id = args.subject_id or f"subject_{int(time.time())}"
        capture_subject_interactive(s_id, args.split)


if __name__ == "__main__":
    main()
