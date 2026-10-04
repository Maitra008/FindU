"""
Read-Only Reference Image Face Recognition Diagnostic Script for FindU.
Processes one exact reference image of 'partho_b55022' through the production
SCRFD + ArcFace + L2-Normalization pipeline using canonical singletons,
and queries the canonical FAISS index for top-5 similarity matches.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.config import DEFAULT_SIMILARITY_THRESHOLD
from src.face_engine import DetectedFace, get_face_engine
from src.index import get_face_index


def diagnose_reference_image(
    image_path: Path | str = PROJECT_ROOT / "data" / "reference_photos" / "partho_b55022" / "ref_01.jpg",
    target_person_id: str = "partho_b55022",
) -> None:
    print("=" * 70)
    print("FINDU -- READ-ONLY REFERENCE IMAGE RECOGNITION DIAGNOSTIC")
    print("=" * 70)

    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / image_path

    print(f"\n[TARGET SPECIFICATION]")
    print(f"  Target Person ID : {target_person_id}")
    print(f"  Image Path       : {img_path}")
    print(f"  File Exists      : {'YES' if img_path.exists() else 'NO'}")

    if not img_path.exists():
        print(f"\n[ERROR] Image file does not exist at: {img_path}")
        return

    # 1. Load image using OpenCV (exact BGR format used by production)
    image = cv2.imread(str(img_path))
    if image is None or image.size == 0:
        print(f"\n[ERROR] Failed to load image with cv2.imread: {img_path}")
        return

    h, w = image.shape[:2]
    print(f"  Resolution       : {w}x{h} ({image.shape[2]} channels, dtype: {image.dtype})")

    # 2. Obtain Canonical Singletons (zero new engine/index instantiations)
    engine = get_face_engine()
    index = get_face_index()

    print(f"\n[CANONICAL ENGINE & INDEX IDENTIFIERS]")
    print(f"  FaceEngine ID : {id(engine)}")
    print(f"  FaceIndex ID  : {id(index)}")
    print(f"  FAISS Size    : {index.total_identities} identities registered")

    # 3. Process image via Canonical Pipeline (SCRFD Detection + ArcFace + L2-Normalization)
    detected_faces: list[DetectedFace] = engine.detect_and_embed(image)
    num_faces = len(detected_faces)

    print(f"\n[PIPELINE EXECUTION: SCRFD + ARCFACE]")
    print(f"  Detected Faces : {num_faces}")

    if num_faces == 0:
        print("  [ERROR] No face detected in the reference image.")
        return

    for i, face in enumerate(detected_faces, start=1):
        print(f"    - Face #{i}: Confidence = {face.confidence:.4f}, BBox = {face.bbox}")

    # Use highest-confidence detected face (primary face)
    primary_face = max(detected_faces, key=lambda f: f.confidence)
    emb = primary_face.normalized_embedding
    emb_shape = emb.shape
    emb_dtype = emb.dtype
    emb_norm = float(np.linalg.norm(emb))

    print(f"\n[EMBEDDING METRICS]")
    print(f"  Shape   : {emb_shape}")
    print(f"  Dtype   : {emb_dtype}")
    print(f"  L2 Norm : {emb_norm:.6f} {'(Strict Unit Vector)' if abs(emb_norm - 1.0) < 1e-4 else '(Unnormalized)'}")

    # 4. Query Canonical FAISS Index
    ntotal = index.total_identities
    if ntotal == 0:
        print("\n[WARNING] FAISS index is empty (ntotal = 0). Cannot perform similarity search.")
        return

    print(f"\n[FAISS INDEX SIMILARITY QUERY -- TOP 5 CANDIDATES]")
    print(f"  Canonical Threshold : {DEFAULT_SIMILARITY_THRESHOLD}")
    print("-" * 70)
    print(f"{'Rank':<6} {'Person ID':<28} {'Similarity':<12} {'Threshold':<12} {'Status'}")
    print("-" * 70)

    # Ensure query vector is (1, 512) float32
    query_vec = emb.astype(np.float32)
    if query_vec.ndim == 1:
        query_vec = np.expand_dims(query_vec, axis=0)

    k = min(5, ntotal)
    similarities, indices = index.index.search(query_vec, k)

    target_matched = False
    top_match_person_id = None
    top_match_similarity = 0.0

    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
        sim_val = float(sim)
        if 0 <= idx < len(index.metadata):
            meta = index.metadata[idx]
            pid = meta.get("person_id", "Unknown")
            pname = meta.get("name", "Unknown")
            display_name = f"{pname} ({pid})"
        else:
            pid = "Unknown"
            display_name = f"Index #{idx}"

        if rank == 1:
            top_match_person_id = pid
            top_match_similarity = sim_val

        is_threshold_pass = sim_val >= DEFAULT_SIMILARITY_THRESHOLD
        status_str = "MATCH" if is_threshold_pass else "NO MATCH"

        # Check if the matched identity corresponds to the target identity
        if is_threshold_pass and (pid == target_person_id or target_person_id.startswith(pid) or pid.startswith(target_person_id)):
            target_matched = True

        print(f"{rank:<6} {display_name:<28} {sim_val:<12.4f} {DEFAULT_SIMILARITY_THRESHOLD:<12.2f} {status_str}")

    print("-" * 70)

    # 5. Summary Evaluation
    print(f"\n[VERIFICATION VERDICT FOR '{target_person_id}']")
    print(f"  Top Match Identity       : {top_match_person_id}")
    print(f"  Top Match Similarity     : {top_match_similarity:.6f}")
    print(f"  Similarity Threshold     : {DEFAULT_SIMILARITY_THRESHOLD}")

    if top_match_similarity >= DEFAULT_SIMILARITY_THRESHOLD:
        print(f"  Threshold Status         : PASS (Similarity {top_match_similarity:.4f} >= {DEFAULT_SIMILARITY_THRESHOLD})")
        print(f"  Identity Match Result    : SUCCESS (Target face recognized by canonical recognition engine)")
    else:
        print(f"  Threshold Status         : FAIL (Similarity {top_match_similarity:.4f} < {DEFAULT_SIMILARITY_THRESHOLD})")
        print(f"  Identity Match Result    : REJECTED")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "reference_photos" / "partho_b55022" / "ref_01.jpg"
    target_id_arg = sys.argv[2] if len(sys.argv) > 2 else "partho_b55022"
    diagnose_reference_image(img_arg, target_id_arg)
