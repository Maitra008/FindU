/**
 * Backend API Client & WebSocket Connection Provider.
 */

import type {
  AlertRecord,
  AlertStatus,
  Camera,
  HealthResponse,
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

function getAuthHeaders(): HeadersInit {
  const token = getStoredToken()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
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

