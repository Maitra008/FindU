"""
FINDU -- FRESH IDENTITY RECOGNITION FORENSIC AUDIT (partho_d3f3e5)
Comprehensive 12-Phase Forensic Inspection Script.
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDINGS_DIR,
    INDEX_PATH,
    METADATA_PATH,
    PROJECT_ROOT,
    MODEL_NAME,
    EXECUTION_PROVIDERS,
    DETECTION_SIZE,
    DET_THRESH,
    EMBEDDING_DIM,
)
from src.db.database import get_session_factory
from src.db.models import Person, AlertRecord, TrackRecord, HistoricalJob
from src.face_engine import get_face_engine, FaceEngine, DetectedFace
from src.index import get_face_index, FaceIndex, normalize_embedding
from src.registration import FaceRegistrar
from src.tracker import FaceTracker
from src.services.alert_service import get_alert_service
from src.pipeline import recognize_frame

TARGET_ID = "partho_d3f3e5"

def run_phase_1():
    print("\n" + "=" * 70)
    print("PHASE 1 -- REGISTRATION FORENSICS")
    print("=" * 70)
    sf = get_session_factory()
    person_row = None
    with sf() as session:
        person_row = session.query(Person).filter(Person.person_id == TARGET_ID).first()
        if person_row:
            print(f"1. Person DB Record ID     : {person_row.id}")
            print(f"2. person_id               : {person_row.person_id}")
            print(f"   name                    : {person_row.name}")
            print(f"   case_id                 : {person_row.case_id}")
            print(f"   status                  : {person_row.status}")
            print(f"3. embedding_path (DB)     : {person_row.embedding_path}")
            print(f"11. registration timestamp : {person_row.created_at}")
        else:
            print(f"DB Record for '{TARGET_ID}': NOT FOUND in 'persons' table!")

    ref_dir = PROJECT_ROOT / "data" / "reference_photos" / TARGET_ID
    print(f"4. Reference photo directory: {ref_dir}")
    print(f"   Directory exists        : {ref_dir.exists()}")
    ref_photos = sorted(list(ref_dir.glob("*.*"))) if ref_dir.exists() else []
    print(f"5. Number of ref photos    : {len(ref_photos)}")
    for idx, p in enumerate(ref_photos, 1):
        print(f"   - Photo {idx}: {p.name} ({p.stat().st_size} bytes)")

    emb_file = EMBEDDINGS_DIR / f"{TARGET_ID}_embedding.npz"
    print(f"6. Embedding file exists   : {emb_file.exists()} ({emb_file})")
    if emb_file.exists():
        data = np.load(str(emb_file), allow_pickle=True)
        emb = data["embedding"]
        print(f"7. Embedding shape         : {emb.shape}")
        print(f"8. Embedding dtype         : {emb.dtype}")
        norm_val = float(np.linalg.norm(emb))
        print(f"9. L2 norm                 : {norm_val:.6f}")
        is_finite = bool(np.all(np.isfinite(emb)))
        print(f"10. Values are finite      : {is_finite}")
        print(f"    Vector preview (first 5): {emb[:5]}")
    else:
        print("   Embedding file missing!")


def run_phase_2():
    print("\n" + "=" * 70)
    print("PHASE 2 -- CANONICAL FAISS INDEX")
    print("=" * 70)
    engine = get_face_engine()
    index = get_face_index()

    print(f"FaceEngine object ID: {id(engine)}")
    print(f"FaceIndex object ID : {id(index)}")
    print(f"FAISS Index ntotal  : {index.total_identities}")
    print(f"Metadata count      : {len(index.metadata)}")

    runtime_match = any(m.get("person_id") == TARGET_ID for m in index.metadata)
    print(f"partho_d3f3e5 in runtime index metadata: {'YES' if runtime_match else 'NO'}")

    disk_idx = FaceIndex(512)
    loaded = disk_idx.load(INDEX_PATH, METADATA_PATH)
    print(f"Persisted index loaded successfully : {loaded}")
    print(f"Persisted FAISS ntotal              : {disk_idx.total_identities}")
    print(f"Persisted metadata count            : {len(disk_idx.metadata)}")
    disk_match = any(m.get("person_id") == TARGET_ID for m in disk_idx.metadata)
    print(f"partho_d3f3e5 in disk metadata      : {'YES' if disk_match else 'NO'}")


def run_phase_3():
    print("\n" + "=" * 70)
    print("PHASE 3 -- EXACT SELF-MATCH TEST")
    print("=" * 70)
    emb_file = EMBEDDINGS_DIR / f"{TARGET_ID}_embedding.npz"
    if not emb_file.exists():
        print(f"ERROR: {emb_file} not found!")
        return

    data = np.load(str(emb_file), allow_pickle=True)
    template_emb = data["embedding"]

    index = get_face_index()
    q = normalize_embedding(template_emb)
    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)

    sims, idxs = index.index.search(q.astype(np.float32), min(5, index.index.ntotal))

    print(f"{'Rank':<5} | {'Person ID':<25} | {'Name':<20} | {'Similarity':<10} | {'Threshold':<10} | {'Status'}")
    print("-" * 85)
    for rank, (s, i) in enumerate(zip(sims[0], idxs[0]), start=1):
        m = index.metadata[i] if (0 <= i < len(index.metadata)) else {}
        pid = m.get("person_id", "Unknown")
        name = m.get("name", "Unknown")
        sim_val = float(s)
        match_str = "MATCH" if sim_val >= DEFAULT_SIMILARITY_THRESHOLD else "NO MATCH"
        print(f"{rank:<5} | {pid:<25} | {name:<20} | {sim_val:<10.6f} | {DEFAULT_SIMILARITY_THRESHOLD:<10.2f} | {match_str}")


def run_phase_4_and_5():
    print("\n" + "=" * 70)
    print("PHASE 4 -- INDIVIDUAL REFERENCE IMAGE TEST")
    print("=" * 70)
    engine = get_face_engine()
    index = get_face_index()

    ref_dir = PROJECT_ROOT / "data" / "reference_photos" / TARGET_ID
    ref_photos = sorted(list(ref_dir.glob("*.*"))) if ref_dir.exists() else []

    emb_file = EMBEDDINGS_DIR / f"{TARGET_ID}_embedding.npz"
    centroid_emb = None
    if emb_file.exists():
        centroid_emb = normalize_embedding(np.load(str(emb_file), allow_pickle=True)["embedding"])

    print(f"{'Photo':<12} | {'SCRFD':<6} | {'Top Identity':<25} | {'Top Sim':<8} | {'Target Sim':<10} | {'Thresh':<6} | {'Decision'}")
    print("-" * 85)

    pairwise_sims_to_centroid = []

    for p in ref_photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = engine.detect_and_embed(img)
        if not faces:
            print(f"{p.name:<12} | 0.00   | {'NO FACE DETECTED':<25} | 0.0000   | 0.0000     | 0.89   | REJECTED")
            continue

        face = faces[0]
        q = normalize_embedding(face.normalized_embedding)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)

        sims, idxs = index.index.search(q.astype(np.float32), min(10, index.index.ntotal))

        top_pid = "Unknown"
        top_name = "Unknown"
        top_sim = float(sims[0][0])
        target_sim = -1.0

        if 0 <= idxs[0][0] < len(index.metadata):
            top_meta = index.metadata[idxs[0][0]]
            top_pid = top_meta.get("person_id", "Unknown")
            top_name = top_meta.get("name", "Unknown")

        for s, i in zip(sims[0], idxs[0]):
            if 0 <= i < len(index.metadata):
                if index.metadata[i].get("person_id") == TARGET_ID:
                    target_sim = float(s)
                    break

        if centroid_emb is not None:
            # Pairwise cosine between this single face embedding and registered centroid
            single_emb = normalize_embedding(face.normalized_embedding)
            pairwise_dot = float(np.dot(single_emb.flatten(), centroid_emb.flatten()))
            pairwise_sims_to_centroid.append(pairwise_dot)

        decision = "MATCH" if target_sim >= DEFAULT_SIMILARITY_THRESHOLD else "NO MATCH"
        print(f"{p.name:<12} | {face.confidence:<6.2f} | {f'{top_name} ({top_pid})'[:25]:<25} | {top_sim:<8.4f} | {target_sim:<10.4f} | {DEFAULT_SIMILARITY_THRESHOLD:<6.2f} | {decision}")

    print("\n" + "=" * 70)
    print("PHASE 5 -- TEMPLATE VS INDIVIDUAL REFERENCE EMBEDDINGS")
    print("=" * 70)
    for idx, sim in enumerate(pairwise_sims_to_centroid, 1):
        print(f"  Reference {idx} -> Registered Centroid Dot Product : {sim:.4f}")
    if pairwise_sims_to_centroid:
        print(f"  Min Similarity   : {min(pairwise_sims_to_centroid):.4f}")
        print(f"  Max Similarity   : {max(pairwise_sims_to_centroid):.4f}")
        print(f"  Mean Similarity  : {np.mean(pairwise_sims_to_centroid):.4f}")
        print(f"  Median Similarity: {np.median(pairwise_sims_to_centroid):.4f}")


def run_phase_6():
    print("\n" + "=" * 70)
    print("PHASE 6 -- LIVE WEBCAM (C_CAM1 / Index 0) - 20 Frames")
    print("=" * 70)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam index 0 could not be opened (hardware device not accessible or in use).")
        return []

    engine = get_face_engine()
    index = get_face_index()

    print(f"{'Sample #':<8} | {'SCRFD':<6} | {'Top Identity':<25} | {'Top Sim':<8} | {'Target Sim':<10} | {'Decision'}")
    print("-" * 75)

    target_sims = []
    sampled = 0

    for i in range(25):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        faces = engine.detect_and_embed(frame)
        if not faces:
            continue

        sampled += 1
        face = faces[0]
        q = normalize_embedding(face.normalized_embedding)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)

        sims, idxs = index.index.search(q.astype(np.float32), min(10, index.index.ntotal))

        top_pid = "Unknown"
        top_name = "Unknown"
        top_sim = float(sims[0][0])
        target_sim = -1.0

        if 0 <= idxs[0][0] < len(index.metadata):
            top_meta = index.metadata[idxs[0][0]]
            top_pid = top_meta.get("person_id", "Unknown")
            top_name = top_meta.get("name", "Unknown")

        for s, idx in zip(sims[0], idxs[0]):
            if 0 <= idx < len(index.metadata):
                if index.metadata[idx].get("person_id") == TARGET_ID:
                    target_sim = float(s)
                    break

        target_sims.append(target_sim)
        decision = "MATCH" if target_sim >= DEFAULT_SIMILARITY_THRESHOLD else "NO MATCH"
        print(f"{sampled:<8} | {face.confidence:<6.2f} | {f'{top_name} ({top_pid})'[:25]:<25} | {top_sim:<8.4f} | {target_sim:<10.4f} | {decision}")

        if sampled >= 20:
            break

    cap.release()

    valid_sims = [s for s in target_sims if s > 0]
    print("-" * 75)
    print(f"Total sampled frames with faces : {len(target_sims)}")
    if valid_sims:
        print(f"Highest similarity to {TARGET_ID} : {max(valid_sims):.4f}")
        print(f"Mean similarity to {TARGET_ID}    : {np.mean(valid_sims):.4f}")
        print(f"Median similarity to {TARGET_ID}  : {np.median(valid_sims):.4f}")
        matches_count = sum(1 for s in valid_sims if s >= DEFAULT_SIMILARITY_THRESHOLD)
        print(f"Number of frames >= 0.89        : {matches_count}")
    return target_sims


def run_phase_7():
    print("\n" + "=" * 70)
    print("PHASE 7 -- HISTORICAL FOOTAGE FORENSIC ANALYSIS")
    print("=" * 70)

    # Videos to analyze
    v_paths = [
        PROJECT_ROOT / "data" / "videos" / "WIN_20261001_15_42_05_Pro.mp4",
        PROJECT_ROOT / "data" / "videos" / "camera_1.mp4",
    ]

    engine = get_face_engine()
    index = get_face_index()

    for v_path in v_paths:
        if not v_path.exists():
            continue
        print(f"\n--- Testing Video: {v_path.name} ---")
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_interval = 1.0 / 2.0  # 2 FPS
        next_sample_time = 0.0
        frame_idx = 0
        sampled_frames = 0
        detected_faces = 0

        frame_records = []

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            cur_time = frame_idx / fps
            if cur_time + 1e-5 >= next_sample_time:
                sampled_frames += 1
                next_sample_time = cur_time + sample_interval
                faces = engine.detect_and_embed(frame)
                detected_faces += len(faces)

                for f_idx, face in enumerate(faces):
                    q = normalize_embedding(face.normalized_embedding)
                    if q.ndim == 1:
                        q = np.expand_dims(q, axis=0)

                    sims, idxs = index.index.search(q.astype(np.float32), min(10, index.index.ntotal))
                    top_pid = "Unknown"
                    top_name = "Unknown"
                    top_sim = float(sims[0][0])
                    target_sim = -1.0

                    if 0 <= idxs[0][0] < len(index.metadata):
                        top_meta = index.metadata[idxs[0][0]]
                        top_pid = top_meta.get("person_id", "Unknown")
                        top_name = top_meta.get("name", "Unknown")

                    for s, idx in zip(sims[0], idxs[0]):
                        if 0 <= idx < len(index.metadata):
                            if index.metadata[idx].get("person_id") == TARGET_ID:
                                target_sim = float(s)
                                break

                    decision = "MATCH" if target_sim >= DEFAULT_SIMILARITY_THRESHOLD else "NO MATCH"
                    frame_records.append({
                        "frame": frame_idx,
                        "time": cur_time,
                        "confidence": face.confidence,
                        "top_sim": top_sim,
                        "top_identity": f"{top_name} ({top_pid})",
                        "target_sim": target_sim,
                        "decision": decision,
                    })

            frame_idx += 1
        cap.release()

        target_sims = [r["target_sim"] for r in frame_records if r["target_sim"] > 0]
        print(f"Total sampled frames : {sampled_frames}")
        print(f"Total faces detected : {detected_faces}")
        if target_sims:
            print(f"Best similarity to {TARGET_ID}   : {max(target_sims):.4f}")
            print(f"Mean similarity to {TARGET_ID}   : {np.mean(target_sims):.4f}")
            print(f"Median similarity to {TARGET_ID} : {np.median(target_sims):.4f}")
            print(f"Number of frames >= 0.89       : {sum(1 for s in target_sims if s >= DEFAULT_SIMILARITY_THRESHOLD)}")
        else:
            print(f"No similarity scores > 0 for {TARGET_ID}")

        print("\nTop 10 Strongest Detected Frames in Video:")
        sorted_records = sorted(frame_records, key=lambda x: x["target_sim"] if x["target_sim"] > 0 else x["top_sim"], reverse=True)[:10]
        print(f"{'Frame':<6} | {'Time(s)':<7} | {'SCRFD':<6} | {'Top Identity':<25} | {'Top Sim':<8} | {'Target Sim':<10} | {'Decision'}")
        print("-" * 85)
        for r in sorted_records:
            print(f"{r['frame']:<6} | {r['time']:<7.2f} | {r['confidence']:<6.2f} | {r['top_identity'][:25]:<25} | {r['top_sim']:<8.4f} | {r['target_sim']:<10.4f} | {r['decision']}")


def run_phase_9_and_10():
    print("\n" + "=" * 70)
    print("PHASE 9 & 10 -- PREPROCESSING & MODEL CONSISTENCY")
    print("=" * 70)
    engine = get_face_engine()
    print(f"Model Name           : {MODEL_NAME}")
    print(f"Execution Providers  : {EXECUTION_PROVIDERS}")
    print(f"Detection Size       : {DETECTION_SIZE}")
    print(f"Detection Threshold  : {DET_THRESH}")
    print(f"Embedding Dimension  : {EMBEDDING_DIM}")
    print(f"FaceEngine Singleton : {id(engine)}")

    # Check FaceRegistrar initialization
    reg = FaceRegistrar(face_engine=engine)
    print(f"FaceRegistrar engine ID: {id(reg.engine)}")
    print(f"FaceEngine app models:")
    for model_name, model_obj in getattr(engine.app, "models", {}).items():
        print(f"  - Model: {model_name} -> {type(model_obj)}")


def run_phase_11():
    print("\n" + "=" * 70)
    print("PHASE 11 -- IDENTITY COLLISION CHECK")
    print("=" * 70)
    index = get_face_index()
    partho_identities = [m for m in index.metadata if "partho" in m.get("name", "").lower() or "partho" in m.get("person_id", "").lower()]
    print(f"All 'Partho'-related identities in FAISS metadata ({len(partho_identities)} found):")
    for m in partho_identities:
        print(f"  - Index ID: {m.get('index_id')} | Person ID: {m.get('person_id')} | Name: {m.get('name')} | NPZ: {m.get('npz_file')}")


if __name__ == "__main__":
    run_phase_1()
    run_phase_2()
    run_phase_3()
    run_phase_4_and_5()
    run_phase_6()
    run_phase_7()
    run_phase_9_and_10()
    run_phase_11()
