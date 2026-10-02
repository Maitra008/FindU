"""
Automated Acceptance and Unit Tests for Recognition Core MVP.
Covers Test A, Test B, Test C, Test D, Test E, FAISS indexing, face_id, and normalization math.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.config import RecognitionConfig
from src.face_engine import FaceEngine, normalize_embedding
from src.index import FaceIndex
from src.registration import FaceRegistrar
from src.recognition import VideoRecognizer


@pytest.fixture(scope="session")
def face_engine():
    """Session-scoped FaceEngine to avoid repeatedly loading weights."""
    return FaceEngine()


@pytest.fixture(scope="session")
def setup_test_environment(face_engine, tmp_path_factory):
    """Create test registration images and test video."""
    import insightface.data

    test_dir = tmp_path_factory.mktemp("rec_test_env")
    reg_dir = test_dir / "registration"
    p_x_dir = reg_dir / "person_x"
    p_y_dir = reg_dir / "person_y"
    p_x_dir.mkdir(parents=True)
    p_y_dir.mkdir(parents=True)
    embeddings_dir = test_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)

    img_t1 = insightface.data.get_image("t1")
    faces = sorted(face_engine.detect_and_embed(img_t1), key=lambda f: f.bbox[0])
    assert len(faces) >= 4, "Expected at least 4 faces in sample image"

    h, w, _ = img_t1.shape

    # Generate 4 registration images for Person X
    bx = faces[1].bbox
    for i in range(4):
        pad_x = 40 + i * 10
        pad_y = 50 + i * 10
        crop = img_t1[max(0, bx[1]-pad_y):min(h, bx[3]+pad_y), max(0, bx[0]-pad_x):min(w, bx[2]+pad_x)]
        cv2.imwrite(str(p_x_dir / f"0{i+1}.jpg"), crop)

    # Generate 4 registration images for Person Y
    by = faces[3].bbox
    for i in range(4):
        pad_x = 40 + i * 10
        pad_y = 50 + i * 10
        crop = img_t1[max(0, by[1]-pad_y):min(h, by[3]+pad_y), max(0, by[0]-pad_x):min(w, by[2]+pad_x)]
        cv2.imwrite(str(p_y_dir / f"0{i+1}.jpg"), crop)

    # Generate multi-segment test video
    video_path = test_dir / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    crop_x = cv2.resize(img_t1[max(0, bx[1]-20):min(h, bx[3]+20), max(0, bx[0]-20):min(w, bx[2]+20)], (160, 200))
    crop_y = cv2.resize(img_t1[max(0, by[1]-20):min(h, by[3]+20), max(0, by[0]-20):min(w, by[2]+20)], (160, 200))

    # Segment 1: No face (frames 0-29) -> Test D
    for _ in range(30):
        out.write(np.zeros((480, 640, 3), dtype=np.uint8))

    # Segment 2: Person X (frames 30-59) -> Test A
    for _ in range(30):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        frame[100:300, 100:260] = crop_x
        out.write(frame)

    # Segment 3: Unknown Person Y (frames 60-89) -> Test B
    for _ in range(30):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        frame[100:300, 240:400] = crop_y
        out.write(frame)

    # Segment 4: Multiple faces X & Y (frames 90-119) -> Test C
    for _ in range(30):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        frame[100:300, 80:240] = crop_x
        frame[100:300, 400:560] = crop_y
        out.write(frame)

    # Segment 5: Blurred Face (frames 120-149) -> Test E
    blurred_x = cv2.GaussianBlur(crop_x, (17, 17), 0)
    for _ in range(30):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        frame[100:300, 240:400] = blurred_x
        out.write(frame)

    out.release()

    return {
        "test_dir": test_dir,
        "p_x_dir": p_x_dir,
        "p_y_dir": p_y_dir,
        "embeddings_dir": embeddings_dir,
        "video_path": video_path,
        "faces": faces,
    }


def test_normalization():
    """Verify L2 normalization properties."""
    vec = np.array([3.0, 4.0], dtype=np.float32)
    norm_vec = normalize_embedding(vec)
    assert np.isclose(np.linalg.norm(norm_vec), 1.0)
    assert np.allclose(norm_vec, [0.6, 0.8])

    # Zero vector handling
    zero_vec = np.zeros(512, dtype=np.float32)
    norm_zero = normalize_embedding(zero_vec)
    assert not np.isnan(norm_zero).any()


def test_face_engine_edge_cases(face_engine):
    """Test face engine with empty and invalid frames."""
    assert face_engine.detect_and_embed(None) == []
    assert face_engine.detect_and_embed(np.array([])) == []
    assert face_engine.load_image("non_existent_image_path.jpg") is None


def test_registration_pipeline(face_engine, setup_test_environment):
    """Test registration of 3-5 images, embedding averaging, and double L2 normalization."""
    env = setup_test_environment
    registrar = FaceRegistrar(face_engine)

    result = registrar.register_person(
        person_id="person_x",
        name="Person X",
        image_dir_or_paths=env["p_x_dir"],
        output_dir=env["embeddings_dir"],
    )

    assert result is not None
    assert result.person_id == "person_x"
    assert result.name == "Person X"
    assert result.num_valid_embeddings == 4
    assert result.identity_embedding.shape == (512,)
    # Verify double normalization results in unit norm
    assert np.isclose(np.linalg.norm(result.identity_embedding), 1.0, atol=1e-5)
    assert result.output_path.exists()


def test_faiss_index_search(face_engine, setup_test_environment):
    """Test FAISS IndexFlatIP cosine similarity and metadata mapping."""
    env = setup_test_environment
    index = FaceIndex(dimension=512)
    indexed_count = index.build_from_directory(env["embeddings_dir"])
    assert indexed_count == 1

    # Search with registered Person X face
    p_x_emb = env["faces"][1].normalized_embedding
    res_match = index.search(p_x_emb, k=1, threshold=0.40)
    assert len(res_match) == 1
    assert res_match[0].is_match is True
    assert res_match[0].person_id == "person_x"
    assert res_match[0].name == "Person X"
    assert res_match[0].similarity > 0.80

    # Search with unregistered Person Y face
    p_y_emb = env["faces"][3].normalized_embedding
    res_unknown = index.search(p_y_emb, k=1, threshold=0.40)
    assert len(res_unknown) == 1
    assert res_unknown[0].is_match is False
    assert res_unknown[0].similarity < 0.40


def test_video_recognition_all_cases(face_engine, setup_test_environment):
    """
    Complete end-to-end video recognition test covering:
    - Test A (Same person -> potential match)
    - Test B (Unknown person -> unknown)
    - Test C (Multiple faces -> processed independently)
    - Test D (No face frames -> no crash)
    - Test E (Blurred/partial face -> no crash, handled)
    - Unique detection ID (face_id) assignment
    - Duplicate suppression
    """
    env = setup_test_environment
    index = FaceIndex(dimension=512)
    index.build_from_directory(env["embeddings_dir"])

    recognizer = VideoRecognizer(
        face_engine=face_engine,
        face_index=index,
        threshold=0.40,
        sample_fps=2.0,
        verbose_debug=True,
    )

    summary = recognizer.process_video(env["video_path"])

    assert summary.frames_processed > 0
    assert summary.faces_detected > 0
    assert summary.potential_matches >= 1
    assert summary.unknown_faces >= 1

    # Verify unique sequential detection IDs
    detection_ids = [e.detection_id for e in summary.events]
    assert len(detection_ids) == summary.faces_detected
    assert detection_ids[0] == "001"
    assert detection_ids[1] == "002"

    # Verify Test D: frames with no faces did not crash and were processed
    assert summary.frames_processed == 10  # 150 frames @ 30 FPS = 5.0s -> 10 sampled frames at 2 FPS

    # Verify duplicate suppression suppressed consecutive frame matches within 3.0s window
    suppressed_events = [e for e in summary.events if e.is_suppressed]
    assert len(suppressed_events) > 0

