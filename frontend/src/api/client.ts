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

export function getWebSocketUrl(): string {
  const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss:' : 'ws:'
  const host = API_BASE_URL.replace(/^https?:\/\//, '')
  return `${wsProtocol}//${host}/ws/alerts`
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
  return res.json()
}

export async function fetchCameras(enabledOnly = false): Promise<Camera[]> {
  const url = `${API_BASE_URL}/api/cameras${enabledOnly ? '?enabled_only=true' : ''}`
  const res = await fetch(url)
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
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch alerts: ${res.statusText}`)
  return res.json()
}

export async function fetchAlertById(alertId: string | number): Promise<AlertRecord> {
  const res = await fetch(`${API_BASE_URL}/api/alerts/${alertId}`)
  if (!res.ok) throw new Error(`Failed to fetch alert ${alertId}: ${res.statusText}`)
  return res.json()
}

export async function updateAlertStatus(
  alertId: string | number,
  status: AlertStatus
): Promise<AlertRecord> {
  const res = await fetch(`${API_BASE_URL}/api/alerts/${alertId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
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
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch tracks: ${res.statusText}`)
  return res.json()
}

export async function fetchWorkerStatuses(): Promise<WorkerMetrics[]> {
  const res = await fetch(`${API_BASE_URL}/api/workers/status`)
  if (!res.ok) throw new Error(`Failed to fetch worker statuses: ${res.statusText}`)
  return res.json()
}

export async function startWorker(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/workers/${cameraId}/start`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to start worker ${cameraId}`)
}

export async function stopWorker(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/workers/${cameraId}/stop`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to stop worker ${cameraId}`)
}

export async function seedDemoMovement(): Promise<AlertRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/demo/seed-movement`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to seed demo movement`)
  return res.json()
}
