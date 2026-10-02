/**
 * Type definitions matching the backend REST API and WebSocket contract.
 */

export type AlertStatus = 'NEW' | 'VERIFIED' | 'DISMISSED'

export type TrackStatus = 'ACTIVE' | 'TERMINATED'

export type WorkerStatus = 'STOPPED' | 'STARTING' | 'RUNNING' | 'COMPLETED' | 'ERROR'

export interface Camera {
  id: number
  camera_id: string
  name: string
  location: string | null
  source: string
  enabled: boolean
  sample_fps: number
  created_at: string | null
}

export interface AlertRecord {
  id: number
  alert_id: string
  camera_id: string
  track_id: string
  person_id: string
  person_name: string
  similarity: number
  max_similarity: number
  mean_similarity: number
  threshold: number
  timestamp: number
  frame_idx: number
  bbox: [number, number, number, number] | number[]
  status: AlertStatus
  snapshot_path: string | null
  created_at: string | null
}

export interface TrackRecord {
  id: number
  track_id: string
  camera_id: string
  person_id: string | null
  person_name: string
  first_seen: number
  last_seen: number
  first_frame: number
  last_frame: number
  detection_count: number
  max_similarity: number
  mean_similarity: number
  top_k_similarity: number
  threshold_matches: number
  alert_triggered: boolean
  status: TrackStatus
  created_at: string | null
  updated_at: string | null
}

export interface WorkerMetrics {
  camera_id: string
  name: string
  source: string
  status: WorkerStatus
  frames_read: number
  frames_processed: number
  faces_detected: number
  tracks_created: number
  alerts_emitted: number
  fps: number
  last_frame_timestamp: number
  error_message: string | null
}

export interface WebSocketAlertMessage {
  event: 'alert.created' | 'alert.updated' | 'pong'
  alert?: AlertRecord
}

export interface HealthResponse {
  status: string
  application: string
  database: string
}

export interface CameraLocation {
  lat: number
  lng: number
  zone: string
}
