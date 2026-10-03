# FindU Demo Script

**System:** FindU Missing Person Recognition & Alert System  
**Audience:** Operators, evaluators, demonstration observers  
**Prerequisite:** Backend running on :8000, frontend on :5173 or :3000

---

## Pre-Demo Checklist

- [ ] Backend started: `python -m src.main serve --host 0.0.0.0 --port 8000`
- [ ] Frontend started: `cd frontend && npm run dev`
- [ ] Browser open: `http://localhost:5173`
- [ ] FAISS index built: `python -m src.main build-index`
- [ ] Demo video files present in `data/videos/`

---

## Demo Flow (10 minutes)

### Step 1 — Login (30 seconds)

1. The browser shows the **FindU login screen**.
2. Enter credentials: `officer01` / `<POLICE_PASSWORD from .env>`
3. Click **Sign In**.
4. Portal loads — note the `officer01 · POLICE` badge in the header.

**Talking point:** JWT token issued server-side; stored in browser localStorage. All subsequent API calls and the WebSocket connection carry this token. The backend validates it on every request.

---

### Step 2 — Portal Overview (1 minute)

Point out the four regions:
- **Left:** Real-time alert feed
- **Center:** Leaflet camera map showing C1–C4 positions
- **Center bottom:** Cross-camera movement timeline
- **Right:** Alert detail and human verification panel

Point out the **LIVE** badge in the header — this is the WebSocket connection status.

**Talking point:** The WebSocket (`/ws/alerts?token=<JWT>`) is authenticated. Unauthenticated clients receive close code 4001 — no data leaks.

---

### Step 3 — Seed Demo Movement (2 minutes)

1. Click **Demo Movement (C2→C3→C1)** in the header.
2. Three alerts appear in the feed (C2 → C3 → C1).
3. Select the **C2** alert — the map highlights Camera 2.
4. The **track timeline** shows: `C2 → C3 → C1` movement path.
5. Select the **C3** alert — map moves to Camera 3.

**Talking point:** All three alerts were created above the 0.89 threshold. Sub-threshold detections produce zero records — privacy by design.

---

### Step 4 — Human Verification (1 minute)

1. With the C1 alert selected, click **Confirm** in the right panel.
2. Alert status changes from `NEW` → `VERIFIED` (WebSocket broadcasts the update instantly).
3. In the alert feed, the badge turns green.

**Talking point:** Only POLICE and ADMIN roles can change alert status. HOSPITAL and NGO users receive 403. This is enforced server-side — no frontend-only gating.

---

### Step 5 — Role Demonstration (optional, 2 minutes)

1. Click **Sign Out** (logout icon in header).
2. Log in as `hospital01` / `<HOSPITAL_PASSWORD>`.
3. Observe: `hospital01 · HOSPITAL` badge.
4. Select an alert and attempt **Confirm** — the button is hidden (UI hint).
5. Open browser DevTools and attempt `PATCH /api/alerts/<id>` manually — server returns **403 Forbidden**.

**Talking point:** Backend is the real security boundary. Frontend hides the button as a UX courtesy, but the API enforces the constraint regardless.

---

### Step 6 — Audit Log (ADMIN only, 1 minute)

1. Sign out, log in as `admin` / `<ADMIN_PASSWORD>`.
2. Open `http://localhost:8000/api/audit` (or DevTools fetch).
3. Show audit entries: `login.success`, `alert.verified`, `login.failure` (if any wrong password was tried).
4. Show that `detail` fields contain no passwords or tokens.

**Talking point:** Complete, tamper-evident audit trail. Every security event is recorded. ADMIN-only via JWT RBAC.

---

### Step 7 — Camera Workers (1 minute)

1. Open `http://localhost:8000/api/workers/status` to show worker telemetry.
2. In the portal header, the C1/C2/C3/C4 badges show live status.
3. If a video file is present, click a camera to start a worker — face recognition runs in real time.

---

### Step 8 — Privacy Assurance (30 seconds)

Open `http://localhost:8000/api/alerts` — all entries have `similarity >= 0.89`.  
**There are no entries with similarity below the threshold** — they were never created.

---

## Key Technical Facts for Q&A

| Question | Answer |
|----------|--------|
| Why 0.89? | Precision 100%, FPR 0% on independent test set |
| What happens below 0.89? | Nothing — no alert, no track, no DB row |
| Where is the token stored? | Browser `localStorage` — cleared on logout |
| What's in the audit log? | Event type, actor, resource, timestamp — NO passwords or embeddings |
| How is CORS configured? | Explicit origins only — no wildcard with credentials |
| What DB does prod use? | PostgreSQL 15 (SQLite for local dev) |

---

## Cleanup After Demo

```bash
# Stop backend
Ctrl+C

# Stop frontend
Ctrl+C

# Optional: clear test DB
rm data/missing_person.db
```
