"""
Camera Workers package for multi-stream video recognition (C1 to C4).
"""

from src.workers.camera_worker import CameraWorker, CameraWorkerMetrics
from src.workers.worker_manager import WorkerManager, get_worker_manager

__all__ = ["CameraWorker", "CameraWorkerMetrics", "WorkerManager", "get_worker_manager"]

