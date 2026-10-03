# FindU: Hackathon Live Demonstration & Judge Script

## 1. System Quick Start

### 1.1 Backend Startup
```bat
python -m src.main serve --host 127.0.0.1 --port 8000 --start-workers
```

### 1.2 Frontend Startup
```bat
cd frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 2. Demo User Matrix (Pre-configured Credentials)

| Role | Username | Password | Purpose & Scope in Demo |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin` (or `demo_admin`) | `admin-demo-CHANGE-ME` | Full access: Register missing persons, add/edit RTSP/USB cameras, upload historical footage, delete feeds. |
| **Police** | `police` (or `demo_police`) | `police-demo-CHANGE-ME` | Investigation workflow: Register cases, verify/dismiss alerts, test camera probes, search missing persons. |
| **Hospital** | `hospital` (or `demo_hospital`) | `hospital-demo-CHANGE-ME` | Healthcare triage: View alert feeds, verify incoming patient sightings, read-only camera telemetry. |
| **NGO** | `ngo` (or `demo_ngo`) | `ngo-demo-CHANGE-ME` | Missing person advocacy: View public case status, search records, read-only camera view. |

---

## 3. End-to-End Live Demo Walkthrough (5-Minute Script)

### Step 1: Login & Role-Based Access Control (0:00 - 0:45)
1. Navigate to `http://localhost:5173`.
2. Login as `admin` (`admin-demo-CHANGE-ME`).
3. Point out the top header showing authenticated role badge `ADMIN` and system recognition threshold (`0.89`).

### Step 2: Camera Command Center & Real-Time Telemetry (0:45 - 2:00)
1. Click the **Camera Command Center** tab in the top navigation.
2. Highlight the **Network Status Summary Bar**:
   - `[4] Cameras` total registered.
   - `[2] Active / Online` (C1 North Entrance & C2 Parking Lot East).
   - Live FPS, AI Inference FPS, and Latency metrics.
3. Demonstrate **Instant Stream Probing**:
   - Click **⚡ Probe** on Camera C1 (`data/videos/camera_1.mp4`).
   - Notice the instant probe badge: `✅ Probe OK (<50ms, 30.0 FPS, 1280x720)`.
4. Demonstrate **Adding a New Feed**:
   - Click **+ Add Camera**.
   - Select **RTSP Stream** -> enter `rtsp://admin:demo123@192.168.1.100:554/live`.
   - Show that credentials are automatically masked.
   - Click **Test Probe** (or select **Video File** `data/videos/camera_2.mp4` for a live test).
   - Click **Register Camera Feed**. The new camera immediately appears in the grid.

### Step 3: Historical Footage Replay & Deep Search (2:00 - 3:15)
1. In the Command Center, click **📼 Upload Footage**.
2. Select a video file from `data/videos/`.
3. Choose **Asynchronous Search** mode (or Replay mode) and click **Upload & Start Processing**.
4. Show the **Historical Footage Analysis Jobs** section at the bottom:
   - Real-time progress bar updating as frames are processed.
   - Live counters for **Faces Found** and **Potential Matches**.

### Step 4: Missing Persons Case Management & Dynamic FAISS Registration (3:15 - 4:15)
1. Click **Cases & Registration** tab.
2. View existing registered cases (e.g. Partho Maitra, Sarah Connor, Person 01–05).
3. Click **+ Register Person**:
   - Fill in Person ID: `TEST_CASE_09`, Name: `John Doe`, Age: `28`, Gender: `Male`.
   - Upload reference photo (`data/reference_photos/p1/ref_01.jpg`).
   - Click **Register & Extract Biometrics**.
4. The system automatically computes ArcFace 512-d embeddings and hot-reloads the FAISS index across all active camera workers in < 100ms without restarting the server.

### Step 5: Real-Time Alerts & Interactive Map Verification (4:15 - 5:00)
1. Switch back to **Live Operations** tab.
2. Click **⚡ Seed Demo Movement** (simulating target moving C2 -> C3 -> C1).
3. Live alerts immediately appear in the left feed via WebSockets.
4. Click an alert:
   - Interactive Leaflet map draws the cross-camera breadcrumb path.
   - Timeline nodes show exact timestamps and cosine similarity scores (> 0.92).
   - Click **Verify Match** to commit the positive identification to the immutable audit trail.

