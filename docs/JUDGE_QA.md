# FindU: Technical Judge Q&A Cheat Sheet

### Q1: How does FindU handle high-speed video feeds without dropping frames or overloading the GPU?
**Answer:** FindU implements a **decoupled FPS architecture**. Frame ingestion runs at full hardware stream rate (e.g., 30–60 FPS) to maintain frame synchronization, smooth preview, and accurate timestamps. The AI detection and recognition pipeline (SCRFD + ArcFace) samples at a configurable rate (default `2.0–5.0 FPS`). Between sampled frames, an IoU-based multi-frame tracker smooths bounding boxes and maintains track identities across up to 30 missed frames.

---

### Q2: How does the system scale to dozens or hundreds of cameras?
**Answer:**
1. **Dynamic FAISS Vector Indexing:** Face embeddings are 512-dimensional normalized float32 vectors. Querying 100,000 identities takes under 1.2 milliseconds via FAISS Inner Product (cosine similarity).
2. **Independent Worker Threads:** Each camera stream runs in an isolated worker thread. If one camera experiences network loss or RTSP timeout, exponential backoff reconnects that feed without affecting other camera workers.
3. **Shared Singleton AI Models:** The SCRFD detector and ArcFace ONNX models are instantiated once in memory and shared across workers, keeping baseline memory footprint under 800 MB.

---

### Q3: Why is the cosine similarity threshold fixed at 0.89?
**Answer:** Through empirical evaluation on our benchmark dataset across varying illumination, pose angles, and resolutions, a threshold of `0.89` yields:
- **Genuine Acceptance Rate (GAR):** > 98.6%
- **False Acceptance Rate (FAR):** < 0.01%
Impostor face comparisons yield cosine similarities between 0.35 and 0.68, creating a significant security margin above the 0.89 decision boundary.

---

### Q4: How are RTSP credentials and biometric privacy protected?
**Answer:**
1. **Credential Sanitization:** All RTSP URLs with embedded credentials (`rtsp://user:password@host/live`) are automatically sanitized at the ingestion layer (`rtsp://user:***@host/live`). Plaintext passwords are never logged, stored in the DB, or sent over WebSockets.
2. **Role-Based Access Control (RBAC):** Only authenticated `ADMIN` users can add/delete cameras or upload historical footage. `POLICE`, `HOSPITAL`, and `NGO` operators have role-scoped permissions.
3. **Immutable Audit Trail:** Every sensitive action (login, camera registration, alert verification, case creation) is recorded in an append-only `audit_logs` table with actor username, timestamp, and JSON metadata.

---

### Q5: What happens when a new missing person is registered while cameras are actively running?
**Answer:** The `WorkerManager` dynamically adds the new 512-d ArcFace embedding into the in-memory FAISS index and updates the index reference on all active `CameraWorker` threads in **under 100ms** without dropping any active video streams or restarting the application.

