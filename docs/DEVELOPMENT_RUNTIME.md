# FindU Development Runtime & Diagnostics Guide

This document outlines the architecture, startup procedures, ports, editor conflict prevention, and troubleshooting instructions for the FindU development environment.

---

## 1. System Topology & Canonical Endpoints

| Component | Port / Protocol | Working Directory | Command | Canonical URL |
|---|---|---|---|---|
| **Vite Dev Server** | `HTTP / 5173` | `./frontend` | `npm run dev` | `http://localhost:5173` |
| **FastAPI Backend** | `HTTP / 8000` | `./` | `python -m src.main serve --host 127.0.0.1 --port 8000 --start-workers` | `http://127.0.0.1:8000` |
| **WebSocket Stream** | `WS / 8000` | `./` | Integrated in FastAPI app | `ws://127.0.0.1:8000/ws/alerts` |
| **SQLite Database** | `data/findu.db` | `./` | Auto-migrated on startup (`src/db/database.py`) | N/A |

---

## 2. Editor / Agent File Conflict Prevention

When pair-programming with autonomous agents or external tools, in-memory editor buffers (e.g. in VS Code) can silently overwrite disk files upon auto-save or tab focus. Follow these rules to prevent stale buffer reversions:

1. **Do not keep stale copies of actively agent-edited files open**:
   - Close or reload tabs for files being modified by the agent (`Header.tsx`, `App.tsx`, `CameraCommandCenter.tsx`, etc.).
2. **Reload/reopen files after agent modifications**:
   - In VS Code, run `File: Revert File` or close and reopen the file from disk after agent updates.
3. **Verify disk contents before saving**:
   - Check the physical file on disk using PowerShell (`Get-Content .\frontend\src\App.tsx`).
4. **Verify `git diff` and `git status`**:
   - Run `git status --short` and `git diff` to confirm the working tree reflects only the intended changes.
5. **Verify the running browser after UI changes**:
   - Always verify the live browser and query the raw served module over HTTP (`http://localhost:5173/src/App.tsx`).

---

## 3. Recommended Startup Order

### Step 1: Start the Backend Server
```powershell
# From project root:
python -m src.main serve --host 127.0.0.1 --port 8000 --start-workers
```
* **Authentication**: Default demo credentials are documented in `DEMO_CREDENTIALS.md`.
* **Database Migration**: Columns (`source_type`, `status`, `stream_fps`, `ai_fps`, etc.) are auto-checked and migrated on startup.

### Step 2: Start the Frontend Development Server
```powershell
# From ./frontend:
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 4. Verifying Serving State & Resolving Mismatches

If browser updates or new features do not appear:

1. **Verify Backend Connection**:
   ```powershell
   curl http://127.0.0.1:8000/health
   ```
2. **Verify Frontend HMR & File Serving**:
   Query the raw served modules directly from Vite:
   ```powershell
   curl http://localhost:5173/src/App.tsx
   curl http://localhost:5173/src/components/Header.tsx
   ```
3. **Verify Runtime Build Diagnostic**:
   Open browser DevTools Console (`F12`), where `[FindU Runtime]` reports the active build ID and mode.
4. **Check for Stale Vite / Node Processes**:
   ```powershell
   Get-NetTCPConnection -LocalPort 5173 | Format-Table OwningProcess, State
   Get-NetTCPConnection -LocalPort 8000 | Format-Table OwningProcess, State
   ```
   If needed, kill the owning process ID:
   ```powershell
   Stop-Process -Id <PID> -Force
   ```
5. **Hard Reload Browser**:
   Press `Ctrl + Shift + R` (or `Ctrl + F5`) to clear browser asset caching.

---

## 5. Testing & Build Verification

* **Backend Test Suite (76 tests)**:
  ```powershell
  python -m pytest -q
  ```
* **Frontend TypeScript & Production Build**:
  ```powershell
  cd frontend
  npm run build
  ```
