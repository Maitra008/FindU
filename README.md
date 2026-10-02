# Recognition Core MVP (Part 1)

Headless face recognition core designed for the Missing-Person Alert Platform.

> **Disclaimer & Verification Policy**: This is a controlled prototype developed with consenting/test dataset samples. Any match produced by this pipeline represents a **potential match** based on similarity score thresholds and **strictly requires human verification** before any alert or action.

---

## 1. Pipeline Architecture

### Registration Pipeline

```
Registration Images (3-5 JPG/PNG)
       │
       ▼
OpenCV Image Decoding
       │
       ▼
SCRFD Face Detection (InsightFace)
       │
       ▼
ArcFace Feature Extraction (512-D)
       │
       ▼
Individual L2 Normalization:  v_i = e_i / ||e_i||_2
       │
       ▼
Identity Mean Template:       v_mean = mean(v_1, ..., v_N)
       │
       ▼
Final L2 Normalization:       v_identity = v_mean / ||v_mean||_2
       │
       ▼
Saved to Compressed Identity Archive (.npz)
```

### Video Recognition Pipeline

```
Video Stream / File (.mp4)
       │
       ▼
OpenCV Frame Sampling (e.g. 2 FPS)
       │
       ▼
SCRFD Face Detection (Single / Multiple / No Face)
       │
       ▼
Assign Unique Sequential Detection ID (e.g. face_id=001)
       │
       ▼
ArcFace Embedding Extraction for Each Detected Face
       │
       ▼
L2 Normalization:  q_norm = q / ||q||_2
       │
       ▼
FAISS Exact Search (`faiss.IndexFlatIP` Cosine Similarity)
       │
       ▼
Threshold Comparison (e.g. sim >= 0.40)
       ├── True  ──► Check Alert Cooldown (Duplicate Suppression)
       │                 ├── Outside Cooldown ──► Print Potential Match Alert Block
       │                 └── Inside Cooldown  ──► Console Alert Suppressed
       └── False ──► Print Unknown Detection Block
```

---

## 2. Project Structure

```
recognition_core/
│
├── data/
│   ├── registration/
│   │   ├── person_x/
│   │   │   ├── 01.jpg
│   │   │   ├── 02.jpg
│   │   │   ├── 03.jpg
│   │   │   └── 04.jpg
│   │   └── person_y/
│   └── videos/
│       └── test_video.mp4
│
├── models/
│   ├── embeddings/
│   │   └── person_x_embedding.npz
│   ├── face_index.faiss
│   └── face_metadata.json
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Centralized configuration & defaults
│   ├── face_engine.py      # InsightFace SCRFD + ArcFace wrapper
│   ├── registration.py     # Registration & double-normalized identity template
│   ├── index.py            # FAISS IndexFlatIP & metadata mapping
│   ├── recognition.py      # Video sampling, matching & duplicate suppression
│   └── main.py             # Unified CLI entry point
│
├── tests/
│   └── test_recognition_core.py # Automated test suite (Tests A - E)
│
├── requirements.txt
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.10 - 3.14)
* CPU Execution Provider via ONNX Runtime

### Install Dependencies
```bash
pip install -r requirements.txt
```

The system uses InsightFace's `buffalo_l` model pack (SCRFD face detector + ArcFace recognition model). Weights are downloaded and cached automatically upon first run.

---

## 4. Usage Guide (CLI)

All operations are managed through a unified CLI interface: `python -m src.main <command>`.

### Step 1: Register a Person
Provide 3–5 portrait images of the known person:
```bash
python -m src.main register \
    --person-id person_x \
    --name "Person X" \
    --images data/registration/person_x
```

* Multi-face policy: If multiple faces are detected in a registration image, a warning is logged and the highest-confidence face is selected.
* Zero-face policy: If no face is detected, the image is skipped with a warning.
* Normalized identity embeddings are saved to `models/embeddings/person_x_embedding.npz`.

### Step 2: Build FAISS Search Index
Index all registered identity embeddings into a `faiss.IndexFlatIP` structure:
```bash
python -m src.main build-index
```
This writes:
* `models/face_index.faiss`: Binary FAISS inner product index
* `models/face_metadata.json`: Position-to-identity metadata mapping

### Step 3: Run Video Recognition
Process a target video with configurable frame sampling, similarity threshold, and duplicate alert cooldown:
```bash
python -m src.main recognize \
    --video data/videos/test_video.mp4 \
    --threshold 0.40 \
    --sample-fps 2.0 \
    --cooldown 5.0
```

To view per-frame bounding boxes and detection confidence in real time:
```bash
python -m src.main recognize \
    --video data/videos/test_video.mp4 \
    --debug
```

---

## 5. Example Terminal Output

```text
Loaded FAISS index with 1 registered identities.
Loading InsightFace...
Model loaded.

Processing video: test_video.mp4
FPS: 30.0
Sampling: 2.0 FPS (Threshold: 0.40, Cooldown: 5.0s)

face_id=001
potential match: Person X
similarity=0.93
timestamp=1.0s

face_id=004
unknown
similarity=0.08
timestamp=2.5s

face_id=005
unknown
similarity=0.08
timestamp=3.0s

face_id=006
unknown
similarity=0.07
timestamp=3.5s

face_id=008
unknown
similarity=0.07
timestamp=4.0s

face_id=010
unknown
similarity=0.07
timestamp=4.5s

Processing complete.
Frames processed: 12
Faces detected: 13
Potential matches: 1
```

---

## 6. Key Features & Design Choices

1. **Unique Detection ID (`face_id`)**:
   - Each detected face receives a sequential zero-padded unique detection ID across the video stream (`face_id=001`, `face_id=002`, ...), facilitating downstream alert tracking, logging, and multi-camera reconciliation.

2. **Strict Double Normalization**:
   - Each individual registration image embedding is L2-normalized: $\hat{e}_i = \frac{e_i}{\|e_i\|_2}$.
   - The identity template is the mean: $\bar{e} = \frac{1}{N} \sum_{i=1}^N \hat{e}_i$.
   - The mean is normalized again: $e_{\text{identity}} = \frac{\bar{e}}{\|\bar{e}\|_2}$.
   - Query vectors from video frames are also L2-normalized: $\hat{q} = \frac{q}{\|q\|_2}$.

3. **Cosine Similarity Search**:
   - Uses `faiss.IndexFlatIP` where inner product $\langle \hat{q}, e_{\text{identity}} \rangle$ is equivalent to exact cosine similarity $[-1.0, 1.0]$.

4. **Centralized Model Management**:
   - `FaceEngine` initializes InsightFace exactly once during startup, preventing memory leaks and frame-by-frame overhead.

5. **Configurable Similarity Threshold**:
   - Default experimental threshold is set to `0.40`. Easily tunable via `--threshold <val>`.

6. **Duplicate Alert Suppression**:
   - Suppresses repeated console alerts for the same identity within a configurable cooldown window (`--cooldown <seconds>`).

---

## 7. Running Automated Acceptance Tests

Execute the test suite verifying Tests A through E:
```bash
pytest -v
```

### Tested Scenarios:
* **Test A (Known Person Match)**: Target identity correctly produces `potential match` with high similarity (>0.70).
* **Test B (Unknown Person)**: Unregistered faces yield low similarity and are labeled `unknown`.
* **Test C (Multiple Faces)**: Every detected face in a crowded frame is evaluated independently.
* **Test D (No Face Frames)**: Frames containing no faces pass through without error.
* **Test E (Degraded / Blurred Detection)**: Partial and blurry faces are processed smoothly.
* **Detection ID Verification**: Sequential `detection_id` / `face_id` tracking.
* **Alert Cooldown**: Duplicate matches within the cooldown window are suppressed from terminal spam.

---

## 8. Known Limitations

* **Headless Prototype**: Contains no UI, web backend, database, or WebSocket server (as specified for Part 1).
* **Lighting & Severe Occlusion**: Extreme angles (>60° yaw) or heavy occlusion may reduce detection confidence below the SCRFD threshold.
* **Tracking**: Uses timestamp cooldown deduplication rather than multi-object tracking (MOT / ByteTrack).
