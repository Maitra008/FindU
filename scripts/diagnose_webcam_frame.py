"""
FINDU -- Forensic Webcam Diagnostic
Tests cv2.VideoCapture(0) to acquire a live frame and trace recognition pipeline against partho_b55022.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_engine import get_face_engine
from src.index import get_face_index, normalize_embedding
from src.config import DEFAULT_SIMILARITY_THRESHOLD


def test_webcam(camera_index: int = 0, target_person_id: str = "partho_b55022"):
    print("=" * 70)
    print(f"WEBCAM FORENSIC TRACE: Index {camera_index}")
    print(f"Target Person ID: {target_person_id}")
    print("=" * 70)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Webcam index {camera_index} could not be opened (hardware unavailable or busy).")
        return

    engine = get_face_engine()
    index = get_face_index()

    print(f"[CANONICAL STATE]")
    print(f"  FaceEngine Object ID: {id(engine)}")
    print(f"  FaceIndex Object ID : {id(index)}")
    print(f"  FAISS Index ntotal  : {index.total_identities}")
    print(f"  Target '{target_person_id}' in FAISS: {any(m.get('person_id') == target_person_id for m in index.metadata)}")

    # Try reading a few frames
    frames_read = 0
    faces_detected_total = 0
    max_sim_target = -1.0
    max_sim_overall = -1.0

    for i in range(5):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frames_read += 1
        faces = engine.detect_and_embed(frame)
        faces_detected_total += len(faces)

        for face_idx, face in enumerate(faces):
            query_norm = normalize_embedding(face.normalized_embedding)
            if query_norm.ndim == 1:
                query_norm = np.expand_dims(query_norm, axis=0)
            sims, idxs = index.index.search(query_norm.astype(np.float32), min(10, index.index.ntotal))

            top_name = "Unknown"
            top_pid = "None"
            sim_val = float(sims[0][0]) if len(sims) > 0 else 0.0
            target_sim = -1.0

            if len(idxs) > 0 and idxs[0][0] >= 0 and idxs[0][0] < len(index.metadata):
                m = index.metadata[idxs[0][0]]
                top_pid = m.get("person_id", "Unknown")
                top_name = m.get("name", "Unknown")

            for s, idx in zip(sims[0], idxs[0]):
                if idx >= 0 and idx < len(index.metadata):
                    if index.metadata[idx].get("person_id") == target_person_id:
                        target_sim = float(s)
                        break

            if sim_val > max_sim_overall:
                max_sim_overall = sim_val
            if target_sim > max_sim_target:
                max_sim_target = target_sim

            decision = "MATCH" if sim_val >= DEFAULT_SIMILARITY_THRESHOLD else "NO MATCH"
            print(f"  Frame {frames_read} Face {face_idx+1}: Conf={face.confidence:.2f} | Top: {top_name} ({top_pid}) | Sim={sim_val:.4f} | Target Sim={target_sim:.4f} | Decision={decision}")

    cap.release()
    print(f"\nWebcam summary: Read {frames_read} frames, {faces_detected_total} faces.")
    print(f"Max Similarity Overall: {max_sim_overall:.4f}")
    print(f"Max Similarity Target ({target_person_id}): {max_sim_target:.4f}")


if __name__ == "__main__":
    test_webcam(0, "partho_b55022")
