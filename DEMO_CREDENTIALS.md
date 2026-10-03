# FindU — Demo Credentials & Role Matrix

> **IMPORTANT SECURITY NOTICE**  
> The accounts listed below are **pre-configured demo credentials intended strictly for local development, hackathon judging, and testing**.  
> **Never commit production passwords or real credentials to GitHub.**  
> In production environments, set strong unique passwords via environment variables.

---

## 1. Demo Account Matrix

| Role | Username | Default Password (Dev/Demo) | Demo Purpose & Permissions |
| :--- | :--- | :--- | :--- |
| **Admin** | `demo_admin` | `admin-demo-CHANGE-ME` | **Full system access**: Camera configuration, worker management, live alert verification, and audit log inspection (`GET /api/audit`). |
| **Police** | `demo_police` | `police-demo-CHANGE-ME` | **Investigation workflow**: Live feed monitoring, map tracking, alert status triage/updates (`PATCH /api/alerts/{id}` to `VERIFIED`/`DISMISSED`), and report generation. |
| **Hospital** | `demo_hospital` | `hospital-demo-CHANGE-ME` | **Hospital intake workflow**: Patient admission verification, read-only alert and track stream monitoring (cannot dismiss or alter alert statuses). |
| **NGO** | `demo_ngo` | `ngo-demo-CHANGE-ME` | **Field assistance workflow**: Field team coordination, read-only alerts and missing person sighting validation. |

*Note: Legacy shorthand aliases (`admin`, `officer01`, `hospital01`, `ngo01`) are also seeded for backward compatibility with existing automated scripts.*

---

## 2. Role-Based Access Control (RBAC) Matrix

| Endpoint / Feature | ADMIN | POLICE | HOSPITAL | NGO |
| :--- | :---: | :---: | :---: | :---: |
| **System Health** (`/health`) | Public | Public | Public | Public |
| **Auth Login** (`/api/auth/login`) | Public | Public | Public | Public |
| **Current User Info** (`/api/auth/me`) | Yes | Yes | Yes | Yes |
| **List/View Alerts** (`GET /api/alerts`) | Yes | Yes | Yes | Yes |
| **Verify/Update Alert Status** (`PATCH /api/alerts/{id}`) | Yes | Yes | No (403) | No (403) |
| **List/View Tracks** (`GET /api/tracks`) | Yes | Yes | Yes | Yes |
| **List/View Cameras** (`GET /api/cameras`) | Yes | Yes | Yes | Yes |
| **Create/Edit Cameras** (`POST /api/cameras`) | Yes | No (403) | No (403) | No (403) |
| **Worker Status / Controls** (`/api/workers/*`) | Yes | Yes | No (403) | No (403) |
| **Audit Logs** (`GET /api/audit`) | Yes | No (403) | No (403) | No (403) |
| **Live WebSocket Alerts** (`/ws/alerts`) | Yes | Yes | Yes | Yes |

---

## 3. Environment Variable Configuration

To override default demo credentials in staging or production, define the following in your `.env` or system environment:

```bash
# JWT Token Settings
JWT_SECRET_KEY=your-secure-random-jwt-secret-at-least-32-chars-long
ACCESS_TOKEN_EXPIRE_MINUTES=480

# User Password Overrides
ADMIN_PASSWORD=your_production_admin_password
POLICE_PASSWORD=your_production_police_password
HOSPITAL_PASSWORD=your_production_hospital_password
NGO_PASSWORD=your_production_ngo_password
```

---

## 4. How Password Hashing Works

- Passwords are encrypted using **Argon2id** (`pwdlib[argon2,bcrypt]`) with automatic salt generation and memory hardness.
- Plaintext passwords are never persisted to the database or logged in audit trails.
- All login attempts (both successful and failed) are recorded in the tamper-evident `audit_logs` table without exposing credentials.

