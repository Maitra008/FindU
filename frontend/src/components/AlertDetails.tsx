import React from 'react'
import { AlertTriangle, CheckCircle, Clock, Crosshair, Shield, User, Video, XCircle } from 'lucide-react'
import type { AlertRecord, AlertStatus, TrackRecord } from '../types'
import { CAMERA_COORDINATES } from '../config/mapConfig'

interface AlertDetailsProps {
  alert: AlertRecord | null
  track: TrackRecord | null
  onUpdateStatus: (alertId: string | number, newStatus: AlertStatus) => Promise<void>
  isUpdating: boolean
}

export const AlertDetails: React.FC<AlertDetailsProps> = ({
  alert,
  track,
  onUpdateStatus,
  isUpdating,
}) => {
  if (!alert) {
    return (
      <div className="w-80 lg:w-96 bg-slate-900 border-l border-slate-800 p-6 flex flex-col items-center justify-center text-center text-slate-500 select-none">
        <Crosshair className="w-8 h-8 mb-2 text-slate-600" />
        <p className="text-xs">Select an alert from the feed to review details and perform verification.</p>
      </div>
    )
  }

  const camGeo = CAMERA_COORDINATES[alert.camera_id]

  return (
    <div className="w-80 lg:w-96 bg-slate-900 border-l border-slate-800 flex flex-col h-full select-none shrink-0">
      {/* Panel Header */}
      <div className="p-3 border-b border-slate-800 bg-slate-950 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Shield className="w-4 h-4 text-sky-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Alert Verification & Details
          </h2>
        </div>
        <span
          className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded uppercase ${
            alert.status === 'VERIFIED'
              ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
              : alert.status === 'DISMISSED'
              ? 'bg-slate-800 text-slate-400 border border-slate-700'
              : 'bg-amber-950 text-amber-300 border border-amber-800 animate-pulse'
          }`}
        >
          {alert.status}
        </span>
      </div>

      {/* Main Content Info */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs font-mono">
        {/* Person Identity Card */}
        <div className="bg-slate-950 p-3 rounded border border-slate-800">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Candidate Identity
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <User className="w-4 h-4 text-sky-400" />
              <span className="text-sm font-bold text-slate-100">{alert.person_name}</span>
            </div>
            <span className="text-[11px] text-slate-400">ID: {alert.person_id}</span>
          </div>

          <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between">
            <span className="text-slate-400">Cosine Similarity:</span>
            <span className="font-bold text-sky-400 text-sm">
              {(alert.similarity * 100).toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center justify-between text-[11px] text-slate-500 mt-0.5">
            <span>Threshold Cutoff:</span>
            <span>{(alert.threshold * 100).toFixed(1)}% (Frozen)</span>
          </div>
        </div>

        {/* Location & Capture Metadata */}
        <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-2">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Capture Location
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center space-x-1">
              <Video className="w-3.5 h-3.5 text-slate-500" />
              <span>Camera Feed:</span>
            </span>
            <span className="font-bold text-slate-200">
              {alert.camera_id} — {camGeo?.name || 'CCTV Stream'}
            </span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400">Zone / Gate:</span>
            <span className="text-slate-300">{camGeo?.zone || 'Main Facility'}</span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center space-x-1">
              <Clock className="w-3.5 h-3.5 text-slate-500" />
              <span>Timestamp:</span>
            </span>
            <span className="text-slate-200">{alert.timestamp}s (Frame #{alert.frame_idx})</span>
          </div>
        </div>

        {/* Track Aggregation Summary */}
        <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-2">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Temporal Track Summary
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400">Track ID:</span>
            <span className="text-slate-200">{alert.track_id}</span>
          </div>

          {track && (
            <>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Detections on Track:</span>
                <span className="text-slate-200">{track.detection_count} frames</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Top-k Mean Sim:</span>
                <span className="text-slate-200">{(track.top_k_similarity * 100).toFixed(2)}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Track Lifespan:</span>
                <span className="text-slate-200">{track.first_seen}s - {track.last_seen}s</span>
              </div>
            </>
          )}

          <div className="flex items-center justify-between">
            <span className="text-slate-400">Bounding Box:</span>
            <span className="text-slate-300 text-[11px]">
              [{alert.bbox ? alert.bbox.join(', ') : 'N/A'}]
            </span>
          </div>
        </div>

        {/* Mandatory Verification Policy Notice */}
        <div className="p-2.5 rounded bg-sky-950/30 border border-sky-900/40 text-[11px] text-sky-300 leading-relaxed font-sans">
          <div className="flex items-center space-x-1 font-bold text-sky-400 mb-1">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Human Verification Policy</span>
          </div>
          Potential matches require human verification prior to dispatch. Confirming or rejecting
          persists the decision to the backend audit log.
        </div>
      </div>

      {/* Action Buttons: Confirm / Reject */}
      <div className="p-3 border-t border-slate-800 bg-slate-950 grid grid-cols-2 gap-2">
        <button
          onClick={() => onUpdateStatus(alert.id, 'VERIFIED')}
          disabled={isUpdating || alert.status === 'VERIFIED'}
          className={`flex items-center justify-center space-x-1.5 py-2 px-3 rounded text-xs font-bold font-mono transition cursor-pointer ${
            alert.status === 'VERIFIED'
              ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/80 cursor-default opacity-80'
              : 'bg-emerald-700 hover:bg-emerald-600 text-white shadow-md hover:shadow-emerald-900/30 disabled:opacity-50'
          }`}
        >
          <CheckCircle className="w-3.5 h-3.5" />
          <span>{alert.status === 'VERIFIED' ? 'Verified' : 'Confirm'}</span>
        </button>

        <button
          onClick={() => onUpdateStatus(alert.id, 'DISMISSED')}
          disabled={isUpdating || alert.status === 'DISMISSED'}
          className={`flex items-center justify-center space-x-1.5 py-2 px-3 rounded text-xs font-bold font-mono transition cursor-pointer ${
            alert.status === 'DISMISSED'
              ? 'bg-slate-800 text-slate-400 border border-slate-700 cursor-default opacity-80'
              : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-600 disabled:opacity-50'
          }`}
        >
          <XCircle className="w-3.5 h-3.5" />
          <span>{alert.status === 'DISMISSED' ? 'Dismissed' : 'Reject'}</span>
        </button>
      </div>
    </div>
  )
}
