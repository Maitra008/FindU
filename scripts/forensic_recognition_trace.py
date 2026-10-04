"""
FINDU -- Forensic Recognition Pipeline Diagnostic
Traces the complete pipeline on historical video files and live/recorded frames:
Frame -> SCRFD Detection -> ArcFace Embedding -> FAISS Cosine Search -> FaceTracker -> AlertService -> DB -> WS.

Specifically analyzes similarity against 'partho_b55022' across every frame.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_engine import get_face_engine
from src.index import get_face_index, normalize_embedding
from src.tracker import FaceTracker
from src.services.alert_service import get_alert_service
from src.pipeline import recognize_frame
from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_SAMPLE_FPS


def analyze_video(video_path: Path, camera_id: str = "C_HIST_01", sample_fps: float = 2.0, target_person_id: str = "partho_b55022"):
    print("=" * 70)
    print(f"FORENSIC TRACE: {video_path.name}")
    print(f"Camera/Worker ID: {camera_id}")
    print(f"File Path: {video_path}")
    print(f"Target Person ID: {target_person_id}")
    print("=" * 70)

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return

    engine = get_face_engine()
    index = get_face_index()
    alert_service = get_alert_service()

    print(f"\n[CANONICAL STATE]")
    print(f"  FaceEngine Object ID: {id(engine)}")
    print(f"  FaceIndex Object ID : {id(index)}")
    print(f"  FAISS Index ntotal  : {index.total_identities}")
    print(f"  Metadata Count      : {len(index.metadata)}")
    
    # Check if target person is in index
    target_in_index = any(m.get("person_id") == target_person_id for m in index.metadata)
    print(f"  Target '{target_person_id}' in FAISS: {target_in_index}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cv2.VideoCapture failed to open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    print(f"  Video FPS: {fps:.1f}, Total Frames: {total_frames}, Duration: {duration_sec:.2f}s")
    print(f"  Sample Rate: {sample_fps} FPS (Sample interval: {1.0/sample_fps:.2f}s)")
    print("-" * 70)

    tracker = FaceTracker(
        iou_threshold=0.30,
        max_missed_frames=5,
        top_k=3,
    )

    sample_interval = 1.0 / sample_fps
    next_sample_time = 0.0
    frame_idx = 0
    sampled_frames = 0
    total_faces_detected = 0
    total_matches = 0
    max_similarity_overall = -1.0
    max_similarity_target = -1.0
    best_target_frame_info = None

    print(f"{'Frame':<6} | {'Time(s)':<7} | {'Face':<4} | {'SCRFD':<6} | {'Top Identity':<25} | {'Sim':<6} | {'Thresh':<6} | {'Decision':<8} | {'Target Sim':<10}")
    print("-" * 90)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_timestamp = frame_idx / fps
        if current_timestamp + 1e-5 >= next_sample_time:
            sampled_frames += 1
            next_sample_time = current_timestamp + sample_interval

            # Step 1: Detect and embed
            faces = engine.detect_and_embed(frame)
            total_faces_detected += len(faces)

            for face_idx, face in enumerate(faces):
                # Cosine similarity against all index vectors
                search_results = index.search(
                    query_embedding=face.normalized_embedding,
                    k=1,
                    threshold=DEFAULT_SIMILARITY_THRESHOLD,
                )
                match = search_results[0] if search_results else None

                # Also specifically find similarity to target_person_id
                target_sim = -1.0
                top_name = "Unknown"
                top_pid = "None"
                sim_val = 0.0

                # Query top 5 to get actual metadata and target similarity
                query_norm = normalize_embedding(face.normalized_embedding)
                if query_norm.ndim == 1:
                    query_norm = np.expand_dims(query_norm, axis=0)
                sims, idxs = index.index.search(query_norm.astype(np.float32), min(10, index.index.ntotal))

                if len(sims) > 0 and len(sims[0]) > 0:
                    top_idx = int(idxs[0][0])
                    sim_val = float(sims[0][0])
                    if top_idx >= 0 and top_idx < len(index.metadata):
                        top_meta = index.metadata[top_idx]
                        top_pid = top_meta.get("person_id", "Unknown")
                        top_name = top_meta.get("name", "Unknown")

                    # Look for target_person_id
                    for s, i in zip(sims[0], idxs[0]):
                        if i >= 0 and i < len(index.metadata):
                            m = index.metadata[i]
                            if m.get("person_id") == target_person_id:
                                target_sim = float(s)
                                break

                is_match = (sim_val >= DEFAULT_SIMILARITY_THRESHOLD)
                decision = "MATCH" if is_match else "NO MATCH"

                if sim_val > max_similarity_overall:
                    max_similarity_overall = sim_val

                if target_sim > max_similarity_target:
                    max_similarity_target = target_sim
                    best_target_frame_info = {
                        "frame": frame_idx,
                        "time": current_timestamp,
                        "similarity": target_sim,
                        "bbox": face.bbox,
                        "confidence": face.confidence,
                    }

                print(f"{frame_idx:<6} | {current_timestamp:<7.2f} | {face_idx+1:<4} | {face.confidence:<6.2f} | {f'{top_name} ({top_pid})'[:25]:<25} | {sim_val:<6.4f} | {DEFAULT_SIMILARITY_THRESHOLD:<6.2f} | {decision:<8} | {target_sim:<10.4f}")

            # Also feed through canonical recognize_frame to test tracker & alert pipeline
            res = recognize_frame(
                frame=frame,
                frame_idx=frame_idx,
                timestamp_sec=current_timestamp,
                camera_id=camera_id,
                tracker=tracker,
                face_engine=engine,
                face_index=index,
                alert_service=alert_service,
                threshold=DEFAULT_SIMILARITY_THRESHOLD,
                dispatch_alerts=True,
            )
            total_matches += res.potential_matches

        frame_idx += 1

    cap.release()
    all_tracks = tracker.get_all_tracks()

    print("=" * 70)
    print(f"SUMMARY FOR {video_path.name}")
    print(f"  Sampled Frames        : {sampled_frames}")
    print(f"  Total Faces Detected  : {total_faces_detected}")
    print(f"  Total Tracks Created  : {len(all_tracks)}")
    print(f"  Canonical Matches (>= {DEFAULT_SIMILARITY_THRESHOLD}): {total_matches}")
    print(f"  Max Similarity Overall: {max_similarity_overall:.4f}")
    print(f"  Max Similarity Target ({target_person_id}): {max_similarity_target:.4f}")
    if best_target_frame_info:
        print(f"  Best Target Frame Details: Frame {best_target_frame_info['frame']} (t={best_target_frame_info['time']:.2f}s), Sim={best_target_frame_info['similarity']:.4f}, Conf={best_target_frame_info['confidence']:.2f}, BBox={best_target_frame_info['bbox']}")

    print("\nTRACK STATES:")
    for t in all_tracks:
        print(f"  Track ID: {t.track_id} | Detections: {t.detection_count} | Person: {t.matched_person_id} ({t.matched_person_name}) | Max Sim: {t.max_similarity:.4f} | Mean Sim: {t.mean_similarity:.4f} | Alert Triggered: {t.alert_triggered}")

    print("=" * 70)
    return {
        "max_similarity_overall": max_similarity_overall,
        "max_similarity_target": max_similarity_target,
        "total_faces": total_faces_detected,
        "total_matches": total_matches,
        "tracks": len(all_tracks),
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "partho_b55022"
    
    # Trace 1: Historical video WIN_20261001_15_42_05_Pro.mp4
    v1 = PROJECT_ROOT / "data" / "videos" / "WIN_20261001_15_42_05_Pro.mp4"
    if v1.exists():
        analyze_video(v1, camera_id="C_HIST_01", sample_fps=2.0, target_person_id=target)

    # Trace 2: Historical video WIN_20260928_03_29_37_Pro.mp4
    v2 = PROJECT_ROOT / "data" / "historical_uploads" / "JOB-30811653_WIN_20260928_03_29_37_Pro.mp4"
    if v2.exists():
        analyze_video(v2, camera_id="C_HIST_02", sample_fps=2.0, target_person_id=target)

    # Trace 3: camera_1.mp4 (golden sample with person_01..05 / person_x / person_y)
    v3 = PROJECT_ROOT / "data" / "videos" / "camera_1.mp4"
    if v3.exists():
        analyze_video(v3, camera_id="C1", sample_fps=2.0, target_person_id=target)
