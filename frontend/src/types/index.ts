/**
 * Type definitions matching the backend REST API and WebSocket contract.
 */

export type AlertStatus = 'NEW' | 'VERIFIED' | 'DISMISSED'

export type TrackStatus = 'ACTIVE' | 'TERMINATED'

export type WorkerStatus = 'STOPPED' | 'STARTING' | 'RUNNING' | 'COMPLETED' | 'ERROR'

export type PersonCaseStatus = 'ACTIVE' | 'FOUND' | 'CLOSED'

export type CameraSourceType = 'file' | 'usb' | 'rtsp'

export type CameraStatus = 'ONLINE' | 'CONNECTING' | 'RECONNECTING' | 'OFFLINE' | 'PROCESSING' | 'ERROR'

export type HistoricalJobStatus = 'QUEUED' | 'PROCESSING' | 'PAUSED' | 'COMPLETED' | 'FAILED' | 'CANCELLED'

export interface Camera {
  id: number
  camera_id: string
  name: string
  location: string | null
  source_type: CameraSourceType
  source: string
  enabled: boolean
  sample_fps: number
  status?: CameraStatus | string
  stream_fps?: number
  ai_fps?: number
  latency_ms?: number
  reconnect_count?: number
  frames_processed?: number
  alerts_emitted?: number
  tracks_created?: number
  error_message?: string | null
  last_connected_at?: string | null
  last_frame_at?: string | null
  created_at: string | null
  updated_at?: string | null
}

export interface CameraTestResult {
  success: boolean
  latency_ms: number
  fps?: number
  width?: number
  height?: number
  error?: string | null
  masked_source?: string
}

export interface HistoricalJobRecord {
  id: number
  job_id: string
  camera_id: string
  file_path: string
  sample_fps: number
  status: HistoricalJobStatus
  total_duration_sec: number
  processed_duration_sec: number
  progress_percent: number
  frames_sampled: number
  faces_detected: number
  tracks_created: number
  potential_matches: number
  verified_matches: number
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface PersonRecord {
  id: number
  person_id: string
  name: string
  case_id: string | null
  age: number | null
  gender: string | null
  date_last_seen: string | null
  last_known_location: string | null
  notes: string | null
  status: PersonCaseStatus
  photo_paths: string[]
  embedding_path: string | null
  is_active: boolean
  created_at: string | null
  updated_at: string | null
  total_sightings?: number
  verified_sightings?: number
  alerts?: AlertRecord[]
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
  event: 'alert.created' | 'alert.updated' | 'person.registered' | 'person.updated' | 'job.progress' | 'job.completed' | 'pong'
  alert?: AlertRecord
  person?: PersonRecord
  job?: HistoricalJobRecord
  data?: Record<string, unknown>
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
