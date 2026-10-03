"""
Historical video processing jobs database repository.
Provides CRUD and progress update operations for background video searches.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session

from src.db.models import HistoricalJob

logger = logging.getLogger(__name__)


class HistoricalJobRepository:
    """Repository handling CRUD operations for HistoricalJob entities."""

    def __init__(self, session: Session):
        self.session = session

    def get_all(self, limit: int = 50, offset: int = 0) -> List[HistoricalJob]:
        """Fetch historical jobs ordered by creation date descending."""
        return (
            self.session.query(HistoricalJob)
            .order_by(HistoricalJob.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_by_job_id(self, job_id: str) -> Optional[HistoricalJob]:
        """Fetch job by unique job_id (e.g. 'JOB-2026-001')."""
        return self.session.query(HistoricalJob).filter(HistoricalJob.job_id == job_id).first()

    def create(
        self,
        job_id: str,
        camera_id: str,
        file_path: str,
        sample_fps: float = 2.0,
        total_duration_sec: float = 0.0,
    ) -> HistoricalJob:
        """Create a new historical processing job record."""
        job = HistoricalJob(
            job_id=job_id,
            camera_id=camera_id,
            file_path=file_path,
            sample_fps=sample_fps,
            status="QUEUED",
            total_duration_sec=total_duration_sec,
            processed_duration_sec=0.0,
            progress_percent=0.0,
        )
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        logger.info("Created historical search job '%s' for camera '%s'", job_id, camera_id)
        return job

    def update_progress(
        self,
        job_id: str,
        status: str,
        processed_duration_sec: float,
        progress_percent: float,
        frames_sampled: int,
        faces_detected: int,
        tracks_created: int,
        potential_matches: int,
        verified_matches: int = 0,
        processing_speed_fps: float = 0.0,
        error_message: Optional[str] = None,
        completed: bool = False,
    ) -> Optional[HistoricalJob]:
        """Update job progress telemetry and completion status."""
        job = self.get_by_job_id(job_id)
        if job is None:
            return None

        job.status = status
        job.processed_duration_sec = processed_duration_sec
        job.progress_percent = min(100.0, max(0.0, progress_percent))
        job.frames_sampled = frames_sampled
        job.faces_detected = faces_detected
        job.tracks_created = tracks_created
        job.potential_matches = potential_matches
        job.verified_matches = verified_matches
        job.processing_speed_fps = processing_speed_fps
        if error_message:
            job.error_message = error_message
        if completed:
            job.completed_at = datetime.now(timezone.utc)

        self.session.commit()
        self.session.refresh(job)
        return job

