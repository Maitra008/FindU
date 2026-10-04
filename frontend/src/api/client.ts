/**
 * Backend API Client & WebSocket Connection Provider.
 */

import type {
  AlertRecord,
  AlertStatus,
  Camera,
  CameraSourceType,
  CameraTestResult,
  HealthResponse,
  HistoricalJobRecord,
  PersonRecord,
  TrackRecord,
  WorkerMetrics,
} from '../types'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const TOKEN_KEY = 'findu_auth_token'
const USER_KEY = 'findu_user_info'

export interface UserInfo {
  username: string
  role: string
}

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setStoredToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function getStoredUserInfo(): UserInfo | null {
  const raw = localStorage.getItem(USER_KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

export function setStoredUserInfo(username: string, role: string): void {
  localStorage.setItem(USER_KEY, JSON.stringify({ username, role }))
}

export function clearStoredAuth(): void {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

export function getWebSocketUrl(): string {
  const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss:' : 'ws:'
  const host = API_BASE_URL.replace(/^https?:\/\//, '')
  const token = getStoredToken()
  const tokenQuery = token ? `?token=${encodeURIComponent(token)}` : ''
  return `${wsProtocol}//${host}/ws/alerts${tokenQuery}`
}

function getAuthHeaders(includeJson = true): HeadersInit {
  const token = getStoredToken()
  const headers: Record<string, string> = {}
  if (includeJson) {
    headers['Content-Type'] = 'application/json'
  }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

export function getMediaUrl(relativePath: string | null | undefined): string {
  if (!relativePath) return ''
  if (relativePath.startsWith('http://') || relativePath.startsWith('https://')) {
    return relativePath
  }
  // Convert relative data path to backend media endpoint
  // e.g. data/reference_photos/p1/ref_01.jpg -> /api/media/reference/p1/ref_01.jpg
  if (relativePath.includes('reference_photos/')) {
    const parts = relativePath.split('reference_photos/')[1]?.split('/')
    if (parts && parts.length >= 2) {
      return `${API_BASE_URL}/api/media/reference/${parts[0]}/${parts[1]}`
    }
  }
  if (relativePath.includes('snapshots/')) {
    const filename = relativePath.split('snapshots/')[1]
    return `${API_BASE_URL}/api/media/snapshots/${filename}`
  }
  return `${API_BASE_URL}/${relativePath.replace(/^\/+/, '')}`
}

export interface LoginResponse {
  access_token: string
  token_type: string
  username: string
  role: string
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const formData = new URLSearchParams()
  formData.append('username', username)
  formData.append('password', password)

  const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData.toString(),
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || `Login failed: ${res.statusText}`)
  }

  return res.json()
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
  return res.json()
}

export async function fetchCameras(enabledOnly = false): Promise<Camera[]> {
  const url = `${API_BASE_URL}/api/cameras${enabledOnly ? '?enabled_only=true' : ''}`
  const res = await fetch(url, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(`Failed to fetch cameras: ${res.statusText}`)
  return res.json()
}

export async function fetchAlerts(params?: {
  camera_id?: string
  person_id?: string
  status?: string
}): Promise<AlertRecord[]> {
  const query = new URLSearchParams()
  if (params?.camera_id) query.append('camera_id', params.camera_id)
  if (params?.person_id) query.append('person_id', params.person_id)
  if (params?.status) query.append('status', params.status)

  const url = `${API_BASE_URL}/api/alerts${query.toString() ? `?${query.toString()}` : ''}`
  const res = await fetch(url, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(`Failed to fetch alerts: ${res.statusText}`)
  return res.json()
}

export async function fetchAlertById(alertId: string | number): Promise<AlertRecord> {
  const res = await fetch(`${API_BASE_URL}/api/alerts/${alertId}`, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(`Failed to fetch alert ${alertId}: ${res.statusText}`)
  return res.json()
}

export async function updateAlertStatus(
  alertId: string | number,
  status: AlertStatus
): Promise<AlertRecord> {
  const res = await fetch(`${API_BASE_URL}/api/alerts/${alertId}`, {
    method: 'PATCH',
    headers: getAuthHeaders(),
    body: JSON.stringify({ status }),
  })
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || `Failed to update alert: ${res.statusText}`)
  }
  return res.json()
}

export async function fetchTracks(params?: {
  camera_id?: string
  person_id?: string
  status?: string
}): Promise<TrackRecord[]> {
  const query = new URLSearchParams()
  if (params?.camera_id) query.append('camera_id', params.camera_id)
  if (params?.person_id) query.append('person_id', params.person_id)
  if (params?.status) query.append('status', params.status)

  const url = `${API_BASE_URL}/api/tracks${query.toString() ? `?${query.toString()}` : ''}`
  const res = await fetch(url, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(`Failed to fetch tracks: ${res.statusText}`)
  return res.json()
}

export async function fetchWorkerStatuses(): Promise<WorkerMetrics[]> {
  const res = await fetch(`${API_BASE_URL}/api/workers/status`, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(`Failed to fetch worker statuses: ${res.statusText}`)
  return res.json()
}

export async function startWorker(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/workers/${cameraId}/start`, {
    method: 'POST',
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(`Failed to start worker ${cameraId}`)
}

export async function stopWorker(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/workers/${cameraId}/stop`, {
    method: 'POST',
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(`Failed to stop worker ${cameraId}`)
}

export async function seedDemoMovement(): Promise<AlertRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/demo/seed-movement`, {
    method: 'POST',
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(`Failed to seed demo movement`)
  return res.json()
}

// ---------------------------------------------------------------------------
// Missing Persons & Case Management API
// ---------------------------------------------------------------------------

export async function fetchPersons(params?: {
  status?: string
  search?: string
}): Promise<PersonRecord[]> {
  const query = new URLSearchParams()
  if (params?.status && params.status !== 'ALL') query.append('status', params.status)
  if (params?.search) query.append('search', params.search)

  const url = `${API_BASE_URL}/api/persons${query.toString() ? `?${query.toString()}` : ''}`
  const res = await fetch(url, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(`Failed to fetch persons: ${res.statusText}`)
  return res.json()
}

export async function fetchPersonDetails(personId: string): Promise<PersonRecord> {
  const res = await fetch(`${API_BASE_URL}/api/persons/${encodeURIComponent(personId)}`, {
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(`Failed to fetch person details for ${personId}: ${res.statusText}`)
  return res.json()
}

export async function registerMissingPerson(formData: FormData): Promise<PersonRecord> {
  const res = await fetch(`${API_BASE_URL}/api/persons`, {
    method: 'POST',
    headers: getAuthHeaders(false), // FormData manages its own Content-Type boundary
    body: formData,
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || `Registration failed: ${res.statusText}`)
  }

  return res.json()
}

export async function updatePerson(
  personId: string,
  updates: Partial<PersonRecord>
): Promise<PersonRecord> {
  const res = await fetch(`${API_BASE_URL}/api/persons/${encodeURIComponent(personId)}`, {
    method: 'PATCH',
    headers: getAuthHeaders(true),
    body: JSON.stringify(updates),
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || `Failed to update case: ${res.statusText}`)
  }

  return res.json()
}

// ---------------------------------------------------------------------------
// Camera Command Center & Video Feed Management API
// ---------------------------------------------------------------------------

export async function createCamera(camera: {
  camera_id: string
  name: string
  source: string
  source_type?: CameraSourceType | string
  location?: string
  enabled?: boolean
  sample_fps?: number
  transport?: string
}): Promise<Camera> {
  const res = await fetch(`${API_BASE_URL}/api/cameras`, {
    method: 'POST',
    headers: getAuthHeaders(true),
    body: JSON.stringify(camera),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to create camera: ${res.statusText}`)
  }
  return res.json()
}

export async function updateCamera(
  cameraId: string,
  updates: Partial<Camera> & Record<string, any>
): Promise<Camera> {
  const res = await fetch(`${API_BASE_URL}/api/cameras/${encodeURIComponent(cameraId)}`, {
    method: 'PATCH',
    headers: getAuthHeaders(true),
    body: JSON.stringify(updates),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to update camera: ${res.statusText}`)
  }
  return res.json()
}

export async function deleteCamera(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/cameras/${encodeURIComponent(cameraId)}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  })
  if (!res.ok && res.status !== 204) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to delete camera: ${res.statusText}`)
  }
}

export async function testCameraSource(payload: {
  source: string
  source_type?: string
  transport?: string
}): Promise<CameraTestResult> {
  const res = await fetch(`${API_BASE_URL}/api/cameras/test-source`, {
    method: 'POST',
    headers: getAuthHeaders(true),
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Test probe failed: ${res.statusText}`)
  }
  return res.json()
}

export async function testExistingCamera(cameraId: string): Promise<CameraTestResult> {
  const res = await fetch(`${API_BASE_URL}/api/cameras/${encodeURIComponent(cameraId)}/test`, {
    method: 'POST',
    headers: getAuthHeaders(),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Camera probe failed: ${res.statusText}`)
  }
  return res.json()
}

export async function uploadHistoricalFootage(formData: FormData): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/historical/upload`, {
    method: 'POST',
    headers: getAuthHeaders(false),
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to upload footage: ${res.statusText}`)
  }
  return res.json()
}

export async function fetchHistoricalJobs(limit = 50, offset = 0): Promise<HistoricalJobRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/historical/jobs?limit=${limit}&offset=${offset}`, {
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(`Failed to fetch historical jobs: ${res.statusText}`)
  return res.json()
}

export async function cancelHistoricalJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/historical/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
    headers: getAuthHeaders(),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to cancel historical job: ${res.statusText}`)
  }
}

export async function pauseWorker(cameraId: string): Promise<void> {
  return stopWorker(cameraId)
}

export async function resumeWorker(cameraId: string): Promise<void> {
  return startWorker(cameraId)
}

export async function setWorkerSpeed(cameraId: string, speed: number): Promise<void> {
  await updateCamera(cameraId, { sample_fps: Math.max(0.5, Math.min(30, speed * 2.0)) })
}

export interface ResetEnvironmentResponse {
  success: boolean
  message: string
  deleted: {
    persons: number
    alerts: number
    tracks: number
    historical_jobs: number
    historical_files: number
    reference_photos: number
    embeddings: number
  }
  preserved: {
    cameras: number
    users: number
    baseline_identities: number
  }
}

export async function resetTestEnvironment(): Promise<ResetEnvironmentResponse> {
  const res = await fetch(`${API_BASE_URL}/api/admin/reset-test-environment`, {
    method: 'POST',
    headers: getAuthHeaders(),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to reset test environment: ${res.statusText}`)
  }
  return res.json()
}

