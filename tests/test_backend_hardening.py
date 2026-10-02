"""Release-gate tests for Part 3/4 backend hardening."""

from pathlib import Path

from src.api.websocket import WebSocketConnectionManager
from src.config import DEFAULT_SIMILARITY_THRESHOLD
from src.tracker import DetectionItem, FaceTracker


def _det(bbox, similarity=0.2, person_id=None, name="Unknown"):
    matched = person_id is not None
    return DetectionItem(
        bbox=list(bbox),
        confidence=0.99,
        similarity=similarity,
        person_id=person_id,
        person_name=name,
        is_match=matched,
    )


def test_frozen_runtime_threshold():
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.89


def test_new_track_starts_with_zero_missed_frames():
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=1)
    tracks, alerts = tracker.update(
        [_det([0, 0, 100, 100], similarity=0.2)],
        frame_idx=0,
        timestamp_sec=0.0,
        threshold=0.89,
    )
    assert len(tracks) == 1
    assert tracks[0].missed_frames == 0
    assert tracks[0].active is True
    assert alerts == []


def test_new_track_does_not_terminate_on_creation_frame():
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=0)
    tracks, _ = tracker.update(
        [_det([0, 0, 100, 100])],
        frame_idx=0,
        timestamp_sec=0.0,
        threshold=0.89,
    )
    assert tracks[0].active is True
    assert len(tracker.terminated_tracks) == 0


def test_single_track_produces_one_alert():
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=2)
    det = _det([0, 0, 100, 100], similarity=0.93, person_id="person_x", name="Person X")

    _, first_alerts = tracker.update([det], 0, 0.0, 0.89)
    _, second_alerts = tracker.update([det], 1, 0.5, 0.89)

    assert len(first_alerts) == 1
    assert len(second_alerts) == 0
    assert tracker.active_tracks[0].alert_triggered is True


def test_c3_c4_sources_are_not_enabled_by_default():
    from src.db.database import init_db
    from src.db.models import Camera, Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    init_db(engine=engine, seed_defaults=True)

    with sessionmaker(bind=engine)() as session:
        cameras = {c.camera_id: c for c in session.query(Camera).all()}
        assert cameras["C1"].enabled is True
        assert cameras["C2"].enabled is True
        assert cameras["C3"].enabled is False
        assert cameras["C4"].enabled is False


def test_websocket_manager_imports_cleanly():
    manager = WebSocketConnectionManager()
    assert manager.active_connections == set()
