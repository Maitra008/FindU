# FindU Development Runtime Guide

This document outlines the architecture, startup procedures, ports, and troubleshooting instructions for the FindU development environment.

---

## 1. System Topology & Default Ports

| Component | Port / Protocol | Working Directory | Command |
|---|---|---|---|
| **FastAPI Backend** | `http://127.0.0.1:8000` | Project root (`./`) | `python -m src.main serve --host 127.0.0.1 --port 8000 --start-workers` |
| **WebSocket Stream** | `ws://127.0.0.1:8000/ws/alerts` | Project root (`./`) | Integrated in FastAPI app |
| **Vite Dev Server** | `http://localhost:5173` | `./frontend` | `npm run dev` |
| **SQLite Database** | `data/findu.db` | Project root (`./`) | Auto-migrated on startup (`src/db/database.py`) |

---

## 2. Recommended Startup Order

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

## 3. Verifying Serving State & Resolving Mismatches

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
3. **Check for Stale Vite / Node Processes**:
   ```powershell
   Get-NetTCPConnection -LocalPort 5173 | Format-Table OwningProcess, State
   Get-NetTCPConnection -LocalPort 8000 | Format-Table OwningProcess, State
   ```
   If needed, kill the owning process ID:
   ```powershell
   Stop-Process -Id <PID> -Force
   ```
4. **Hard Reload Browser**:
   Press `Ctrl + Shift + R` (or `Ctrl + F5`) to clear browser service workers and client-side asset caching.

---

## 4. Testing & Verification

* **Backend Suite**:
  ```powershell
  python -m pytest -q
  ```
* **Frontend TypeScript & Production Build**:
  ```powershell
  cd frontend
  npm run build
  ```
