"""
Forensic Diagnostic Script: Controlled Webcam Embedding Diagnostic
Target Identity: dynamically discovered from data/reference_photos or passed via CLI.

Strict Read-Only diagnostic script.
Does NOT modify any production code, database records, FAISS indices, or models.
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SIMILARITY_THRESHOLD, EMBEDDINGS_DIR, MODELS_DIR
from src.face_engine import DetectedFace, get_face_engine, normalize_embedding
from src.index import get_face_index


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).flatten()
    b = np.asarray(vec_b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    print("=" * 70)
    print("FINDU — CONTROLLED WEBCAM EMBEDDING FORENSIC DIAGNOSTIC")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "data" / "diagnostics" / "webcam_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target person_id
    if len(sys.argv) > 1:
        person_id = sys.argv[1]
    else:
        # Auto-detect from data/reference_photos/
        ref_base = PROJECT_ROOT / "data" / "reference_photos"
        subdirs = [d for d in ref_base.iterdir() if d.is_dir()]
        if subdirs:
            person_id = subdirs[0].name
        else:
            person_id = "rahul_05702c"

    print(f"Target Person ID: {person_id}")

    ref_dir = PROJECT_ROOT / "data" / "reference_photos" / person_id
    template_path = EMBEDDINGS_DIR / f"{person_id}_embedding.npz"
    if not template_path.exists():
        template_path = EMBEDDINGS_DIR / f"{person_id}.npz"

    # 1. Load canonical FaceEngine & FaceIndex
    face_engine = get_face_engine()
    face_index = get_face_index()

    print(f"FaceEngine Object ID: {id(face_engine)}")
    print(f"FaceIndex Object ID : {id(face_index)}")
    print(f"FAISS Total Identities (ntotal): {face_index.total_identities}")
    
    is_in_meta = any(m.get("person_id") == person_id for m in face_index.metadata)
    print(f"Target '{person_id}' in FAISS metadata: {is_in_meta}")

    # 2. Load Registered Template
    if not template_path.exists():
        print(f"ERROR: Template file not found at {template_path}")
        return

    npz_data = np.load(str(template_path), allow_pickle=True)
    registered_centroid = npz_data["embedding"]
    registered_centroid_norm = normalize_embedding(registered_centroid).flatten()

    print(f"\n[1] Registered Centroid Template:")
    print(f"  Path    : {template_path}")
    print(f"  Shape   : {registered_centroid.shape}")
    print(f"  Dtype   : {registered_centroid.dtype}")
    print(f"  L2 Norm : {np.linalg.norm(registered_centroid_norm):.6f}")

    # 3. Process Reference Photos
    ref_files = sorted([f for f in ref_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"\n[2] Processing {len(ref_files)} Reference Images from {ref_dir}:")

    ref_data = []
    for f_path in ref_files:
        img = cv2.imread(str(f_path))
        if img is None:
            print(f"  Failed to read {f_path.name}")
            continue

        dets: list[DetectedFace] = face_engine.detect_and_embed(img)
        if not dets:
            print(f"  No face detected in {f_path.name}")
            continue

        det = max(dets, key=lambda d: d.confidence)
        emb = det.normalized_embedding.flatten()
        bbox = det.bbox  # [x1, y1, x2, y2]
        kps = det.landmarks
        score = det.confidence

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect_ratio = (w / h) if h > 0 else 0

        ref_data.append({
            "filename": f_path.name,
            "path": str(f_path),
            "image_shape": img.shape,
            "bbox": [int(x) for x in bbox],
            "bbox_wh": [float(w), float(h)],
            "aspect_ratio": float(aspect_ratio),
            "kps": [[float(p[0]), float(p[1])] for p in kps] if kps is not None else [],
            "score": float(score),
            "embedding": emb,
            "norm": float(np.linalg.norm(emb))
        })
        print(f"  {f_path.name}: bbox=[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] ({w}x{h}, AR={aspect_ratio:.2f}), score={score:.4f}, norm={np.linalg.norm(emb):.6f}")

    # Pairwise Reference-to-Reference Matrix
    num_refs = len(ref_data)
    ref_matrix = np.zeros((num_refs, num_refs))
    print("\n--- 4x4 Reference-to-Reference Cosine Similarity Matrix ---")
    header = "          " + "".join([f"{r['filename']:>12}" for r in ref_data])
    print(header)
    for i in range(num_refs):
        row_str = f"{ref_data[i]['filename']:<10}"
        for j in range(num_refs):
            sim = compute_cosine_similarity(ref_data[i]["embedding"], ref_data[j]["embedding"])
            ref_matrix[i, j] = sim
            row_str += f"{sim:12.4f}"
        print(row_str)

    # Reference to Centroid Similarity
    print("\n--- Reference-to-Centroid Similarities ---")
    for r in ref_data:
        sim_c = compute_cosine_similarity(r["embedding"], registered_centroid_norm)
        print(f"  {r['filename']} vs Registered Centroid: {sim_c:.4f} ({'PASS >=0.89' if sim_c >= DEFAULT_SIMILARITY_THRESHOLD else 'FAIL <0.89'})")

    # 4. Same-Image Consistency Test
    print("\n[3] Same-Image Registration vs Recognition Consistency Test:")
    ref01_path = ref_data[0]["path"]
    ref01_img = cv2.imread(ref01_path)
    ref01_dets_repeat = face_engine.detect_and_embed(ref01_img)
    ref01_emb_repeat = ref01_dets_repeat[0].normalized_embedding.flatten()

    abs_diff = np.abs(ref_data[0]["embedding"] - ref01_emb_repeat)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    repeat_sim = compute_cosine_similarity(ref_data[0]["embedding"], ref01_emb_repeat)
    print(f"  Repeat extraction on {ref_data[0]['filename']}:")
    print(f"  Max Absolute Difference: {max_abs_diff:.8e}")
    print(f"  Mean Absolute Difference: {mean_abs_diff:.8e}")
    print(f"  Cosine Similarity (Self-Consistency): {repeat_sim:.8f}")

    # 5. Capture Webcam Frames (20 frames with face detection)
    print("\n[4] Capturing 20 Controlled Webcam Frames (camera_index=0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  CAP_DSHOW failed, falling back to default backend...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("  ERROR: Could not open cv2.VideoCapture(0)!")
        return

    # Warm up camera
    for _ in range(15):
        cap.read()
        time.sleep(0.04)

    webcam_data = []
    frames_captured = 0
    attempts = 0
    max_attempts = 150

    while frames_captured < 20 and attempts < max_attempts:
        attempts += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        dets = face_engine.detect_and_embed(frame)
        if not dets:
            time.sleep(0.05)
            continue

        # Face detected
        frames_captured += 1
        frame_filename = f"frame_{frames_captured:02d}.jpg"
        frame_save_path = output_dir / frame_filename
        cv2.imwrite(str(frame_save_path), frame)

        det = max(dets, key=lambda d: d.confidence)
        emb = det.normalized_embedding.flatten()
        bbox = det.bbox
        kps = det.landmarks
        score = det.confidence

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        ar = (w / h) if h > 0 else 0

        # Calculate similarities against references and centroid
        ref_sims = [compute_cosine_similarity(emb, r["embedding"]) for r in ref_data]
        centroid_sim = compute_cosine_similarity(emb, registered_centroid_norm)

        # Query FAISS index as well
        faiss_matches = face_index.search(emb, k=3, threshold=0.0)

        webcam_data.append({
            "frame_id": frames_captured,
            "filename": frame_filename,
            "path": str(frame_save_path),
            "image_shape": frame.shape,
            "bbox": [int(x) for x in bbox],
            "bbox_wh": [float(w), float(h)],
            "aspect_ratio": float(ar),
            "kps": [[float(p[0]), float(p[1])] for p in kps] if kps is not None else [],
            "score": float(score),
            "embedding": emb,
            "norm": float(np.linalg.norm(emb)),
            "ref_similarities": ref_sims,
            "centroid_similarity": centroid_sim,
            "top_faiss_match": {
                "person_id": faiss_matches[0].person_id,
                "name": faiss_matches[0].name,
                "similarity": faiss_matches[0].similarity,
                "is_match": faiss_matches[0].is_match
            } if faiss_matches else None
        })

        top_match_id = faiss_matches[0].name if faiss_matches else "none"
        top_match_sim = faiss_matches[0].similarity if faiss_matches else 0.0
        print(f"  Frame {frames_captured:02d}: bbox={w}x{h} (AR={ar:.2f}), score={score:.3f}, centroid_sim={centroid_sim:.4f}, top_faiss={top_match_id}:{top_match_sim:.4f}")
        time.sleep(0.08)

    cap.release()
    print(f"Captured {len(webcam_data)} frames with valid face detections.")

    # 6. Webcam Statistics & Analysis
    centroid_sims = [w["centroid_similarity"] for w in webcam_data]
    scores = [w["score"] for w in webcam_data]
    bbox_widths = [w["bbox_wh"][0] for w in webcam_data]
    bbox_heights = [w["bbox_wh"][1] for w in webcam_data]

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Webcam vs Centroid Similarity Statistics (N={len(centroid_sims)}):")
    print(f"  Min:    {np.min(centroid_sims):.4f}")
    print(f"  Max:    {np.max(centroid_sims):.4f}")
    print(f"  Mean:   {np.mean(centroid_sims):.4f}")
    print(f"  Median: {np.median(centroid_sims):.4f}")
    print(f"  Std:    {np.std(centroid_sims):.4f}")
    print(f"  Frames >= {DEFAULT_SIMILARITY_THRESHOLD}: {sum(1 for s in centroid_sims if s >= DEFAULT_SIMILARITY_THRESHOLD)} / {len(centroid_sims)}")

    print(f"\nDetection Bounding Box Summary:")
    print(f"  Webcam Mean BBox Size   : {np.mean(bbox_widths):.1f} x {np.mean(bbox_heights):.1f} px (Aspect Ratio: {np.mean([w['aspect_ratio'] for w in webcam_data]):.2f})")
    print(f"  Reference Mean BBox Size: {np.mean([r['bbox_wh'][0] for r in ref_data]):.1f} x {np.mean([r['bbox_wh'][1] for r in ref_data]):.1f} px (Aspect Ratio: {np.mean([r['aspect_ratio'] for r in ref_data]):.2f})")
    print(f"  Webcam Mean Detection Score   : {np.mean(scores):.4f}")
    print(f"  Reference Mean Detection Score: {np.mean([r['score'] for r in ref_data]):.4f}")

    # Per-reference average similarity
    print("\nWebcam-to-Reference Breakdown:")
    for idx, r in enumerate(ref_data):
        sims_to_this_ref = [w["ref_similarities"][idx] for w in webcam_data]
        print(f"  Webcam -> {r['filename']}: Min={np.min(sims_to_this_ref):.4f}, Max={np.max(sims_to_this_ref):.4f}, Mean={np.mean(sims_to_this_ref):.4f}")

    # 7. Save JSON Analysis
    json_output = {
        "person_id": person_id,
        "template_path": str(template_path),
        "template_norm": float(np.linalg.norm(registered_centroid_norm)),
        "reference_images": [
            {
                "filename": r["filename"],
                "image_shape": list(r["image_shape"]),
                "bbox": r["bbox"],
                "bbox_wh": r["bbox_wh"],
                "aspect_ratio": r["aspect_ratio"],
                "score": r["score"],
                "norm": r["norm"],
                "sim_to_centroid": compute_cosine_similarity(r["embedding"], registered_centroid_norm)
            }
            for r in ref_data
        ],
        "ref_similarity_matrix": ref_matrix.tolist(),
        "same_image_consistency": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "repeat_sim": repeat_sim
        },
        "webcam_frames": [
            {
                "frame_id": w["frame_id"],
                "filename": w["filename"],
                "image_shape": list(w["image_shape"]),
                "bbox": w["bbox"],
                "bbox_wh": w["bbox_wh"],
                "aspect_ratio": w["aspect_ratio"],
                "score": w["score"],
                "norm": w["norm"],
                "ref_similarities": w["ref_similarities"],
                "centroid_similarity": w["centroid_similarity"],
                "top_faiss_match": w["top_faiss_match"]
            }
            for w in webcam_data
        ],
        "statistics": {
            "centroid_similarity": {
                "min": float(np.min(centroid_sims)),
                "max": float(np.max(centroid_sims)),
                "mean": float(np.mean(centroid_sims)),
                "median": float(np.median(centroid_sims)),
                "std": float(np.std(centroid_sims)),
                "frames_passing_threshold": sum(1 for s in centroid_sims if s >= DEFAULT_SIMILARITY_THRESHOLD),
                "threshold": DEFAULT_SIMILARITY_THRESHOLD,
                "total_frames": len(centroid_sims)
            },
            "webcam_bbox": {
                "mean_width": float(np.mean(bbox_widths)),
                "mean_height": float(np.mean(bbox_heights)),
                "mean_aspect_ratio": float(np.mean([w["aspect_ratio"] for w in webcam_data])),
                "mean_score": float(np.mean(scores))
            },
            "ref_bbox": {
                "mean_width": float(np.mean([r["bbox_wh"][0] for r in ref_data])),
                "mean_height": float(np.mean([r["bbox_wh"][1] for r in ref_data])),
                "mean_aspect_ratio": float(np.mean([r["aspect_ratio"] for r in ref_data])),
                "mean_score": float(np.mean([r["score"] for r in ref_data]))
            }
        }
    }

    json_path = output_dir / "embedding_analysis.json"
    with open(str(json_path), "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nSaved complete diagnostic analysis to {json_path}")


if __name__ == "__main__":
    main()
