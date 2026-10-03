# FindU: Camera Integration & Command Center Architecture Audit

**Date:** October 4, 2026  
**Repository:** `Maitra008/FindU`  
**Target:** Camera Command Center + Live & Historical Replay Integration

---

## 1. Executive Summary & Existing Architecture

FindU is an authorized AI-assisted missing-person investigation platform built with FastAPI, OpenCV, InsightFace (SCRFD detector + ArcFace 512-d embeddings), FAISS similarity indexing, temporal `FaceTracker`, SQLAlchemy (SQLite/PostgreSQL), and a React + Vite + Tailwind operator portal.

The existing system provides:
- **Core Pipeline:** Face Detection (SCRFD) $\rightarrow$ Alignment $\rightarrow$ 512-d Normalized Embedding (ArcFace) $\rightarrow$ FAISS Cosine Indexing $\rightarrow$ Temporal Tracking (`FaceTracker` with IoU association and top-k similarity pooling) $\rightarrow$ Strict Threshold Verification ($0.89$) $\rightarrow$ Human Verification Alert Triaging $\rightarrow$ Cross-Camera Chronological Trajectory Mapping.
- **Security & RBAC:** JWT authentication (Argon2 password hashing via `pwdlib`), 4 roles (`ADMIN`, `POLICE`, `HOSPITAL`, `NGO`), and immutable `AuditLog` records.
- **Case Management:** Searchable cases, 1–10 reference photograph registration, dynamic FAISS index propagation, media preview with lightbox.
- **WebSocket & Real-time Stream:** Authenticated `/ws/alerts?token=...` broadcasting alerts, status changes, and registration events.

---

## 2. Reusable Core Components (DO NOT REWRITE)

| Subsystem | Existing Module | Current Interface & Purpose | Extension Strategy |
|---|---|---|---|
| **Face Engine** | `src/face_engine.py` | `FaceEngine.detect_and_embed(frame) -> List[DetectedFace]` | Reused directly across all camera streams (singleton/shared). |
| **FAISS Face Index** | `src/index.py` | `FaceIndex.search(embedding, k, threshold) -> List[MatchResult]` | Reused directly; dynamic template injection via `register_face_identity`. |
| **Temporal Tracker** | `src/tracker.py` | `FaceTracker.update(detections, frame_idx, timestamp_sec, threshold, source_name)` | Instantiated per worker stream; handles IoU tracking and threshold match aggregation. |
| **Alert & Track DB** | `src/services/alert_service.py` | `create_alert_from_track_alert`, `update_track_state` | Direct pipeline target for potential match alerts. |
| **Auth & Security** | `src/auth/` | `get_current_user`, `require_role("ADMIN")`, `create_access_token` | Protects camera modification and historical footage job dispatch. |
| **WebSocket Hub** | `src/api/websocket.py` | `ConnectionManager.broadcast_sync(message)` | Extended with `camera.status`, `camera.metrics`, `job.progress`. |
| **Media Service** | `src/api/routes/media.py` | Safe snapshot and reference photo serving | Pattern reused for uploaded historical footage storage. |

---

## 3. Current Camera Capabilities & Limitations

### Current Capabilities:
- Fixed seeding for cameras `C1` to `C4` in SQLite database.
- Dedicated `CameraWorker` thread per camera polling OpenCV `cv2.VideoCapture`.
- `WorkerManager` singleton with lifecycle controls (`start_worker`, `stop_worker`, `start_all`, `stop_all`, `get_statuses`).
- Real-time telemetry metrics (`frames_read`, `frames_processed`, `fps`, `faces_detected`, `tracks_created`, `alerts_emitted`).

### Current Limitations:
1. **Source Coupling:** `CameraWorker` directly calls `cv2.VideoCapture(src)` without an abstraction layer for stream protocols, reconnect backoff, timeout handling, or video replay pacing.
2. **Missing RTSP & USB Configuration:** No explicit configuration for RTSP transport (TCP vs UDP), timeout, reconnect strategies, or USB device indices.
3. **Database Schema:** `Camera` model only contains basic fields (`camera_id`, `name`, `location`, `source`, `enabled`, `sample_fps`, `created_at`). Lacks `source_type`, `status`, `stream_fps`, `ai_fps`, `latency_ms`, `reconnect_count`, `error_message`, and timestamps.
4. **Historical Replay & Jobs:** No asynchronous background job system for batch/long video historical analysis; historical files run synchronously or lack job tracking models.
5. **UI & Command Center:** Live CCTV Operations view displays a 4-feed status bar, Leaflet map, and timeline, but lacks a dedicated Camera Command Center network grid, camera cards with live metric gauges, camera add/edit modals, test connection utilities, and historical footage replay controls.

---

## 4. Protected Files vs Files to Extend

### Files that MUST Remain Untouched (Core Stability):
- `src/recognition.py` (Core pipeline entrypoint)
- `src/face_engine.py` (SCRFD & ArcFace model inference)
- `src/index.py` (FAISS indexing and cosine similarity search)
- `src/tracker.py` (IoU tracking and temporal alert generation)
- `src/auth/security.py` & `src/auth/dependencies.py` (JWT RBAC authentication)
- `src/services/alert_service.py` & `src/services/audit_service.py` (Alert persistence and audit trail)

### Files to Extend / Add:
- **NEW `src/cameras/source.py` / `src/cameras/sources.py`**:
  Abstract `CameraSource` interface with `FileSource`, `USBSource`, and `RTSPSource` (TCP preferred, backoff reconnection, FPS rate limiter).
- **`src/db/models.py` & `src/db/database.py`**:
  Extend `Camera` model with `source_type`, `status`, `stream_fps`, `ai_fps`, `latency_ms`, `reconnect_count`, `error_message`, `updated_at`. Add `HistoricalJob` model for background video processing. Add additive SQLite schema migrations in `_migrate_sqlite_columns`.
- **`src/db/repositories/cameras.py` & `src/api/routes/cameras.py`**:
  Extend with full CRUD, test connection endpoint (`POST /api/cameras/{id}/test`), and ADMIN-only permission guards.
- **`src/workers/camera_worker.py` & `src/workers/worker_manager.py`**:
  Integrate `CameraSource` abstraction, decouple stream FPS from AI inference sampling FPS, ensure bounded frame queueing, and isolate camera crashes.
- **NEW `src/services/historical_service.py` & `src/api/routes/historical.py`**:
  Asynchronous historical footage processing with progress calculation and WebSocket events (`job.progress`, `job.completed`).
- **`frontend/src/`**:
  - `types/index.ts`: Add `CameraSourceType`, `CameraStatus`, `HistoricalJobRecord`, `CameraMetricsMessage`.
  - `api/client.ts`: Add camera management API calls (`testCameraConnection`, `createCamera`, `updateCamera`, `deleteCamera`, `createHistoricalJob`, `fetchHistoricalJobs`).
  - `components/CameraCommandCenter.tsx`: Professional Command Center grid with network summary bar, camera cards, status badges, video preview, FPS/latency telemetry, and test connection action.
  - `components/AddCameraModal.tsx`: Admin-only camera creation & source configuration modal (RTSP, USB, Video Replay).
  - `components/HistoricalReplayModal.tsx`: Admin historical footage upload and background search dispatcher.
  - `components/MapView.tsx`: Dynamic marker styling based on camera status (`ONLINE`, `OFFLINE`, `PROCESSING`, `ALERT`) and click drawer integration.
  - `components/Header.tsx` & `App.tsx`: Tab navigation integration for Camera Command Center, Live Operations, and Case Management.

---

## 5. Database Compatibility & Safety

- Existing database `data/missing_person.db` will be preserved without data loss or dropping tables.
- Additive SQLite column migration checks (`PRAGMA table_info(cameras)`) will automatically add new columns with safe defaults (`source_type='file'`, `status='OFFLINE'`, etc.).
- Default cameras `C1` to `C4` will remain pre-configured and immediately operational.

---

## 6. Test Coverage & Verification Strategy

- **Unit Tests:** `CameraSource` adapters (`FileSource`, `USBSource`, `RTSPSource`), `CameraWorker` frame sampling, failure isolation, RBAC enforcement on camera CRUD, historical job runner.
- **Integration Tests:** REST endpoints (`/api/cameras`, `/api/cameras/{id}/test`, `/api/historical`), WebSocket telemetry messages.
- **E2E Browser Tests:** Real Playwright browser testing verifying Admin camera addition, test connection diagnostics, non-admin permission restrictions, and seamless Live Operations tracking.

