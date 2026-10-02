import React from 'react'
import { ArrowRight, Clock, Navigation } from 'lucide-react'
import type { AlertRecord, TrackRecord } from '../types'
import { CAMERA_COORDINATES } from '../config/mapConfig'

interface TrackTimelineProps {
  personName: string
  personId?: string | null
  tracks: TrackRecord[]
  alerts: AlertRecord[]
  selectedTrackId: string | null
  onSelectTrackNode: (track: TrackRecord, alert?: AlertRecord) => void
}

export const TrackTimeline: React.FC<TrackTimelineProps> = ({
  personName,
  tracks,
  alerts,
  selectedTrackId,
  onSelectTrackNode,
}) => {
  // Sort tracks chronologically by first_seen
  const sortedTracks = [...tracks].sort((a, b) => a.first_seen - b.first_seen)

  if (sortedTracks.length === 0) {
    return (
      <div className="p-4 bg-slate-900 border-t border-slate-800 text-center text-xs text-slate-500">
        No active tracking timeline recorded. Select an alert to inspect journey.
      </div>
    )
  }

  return (
    <div className="p-3 bg-slate-950 border-t border-slate-800 select-none">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Navigation className="w-4 h-4 text-sky-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Cross-Camera Movement Timeline: <span className="text-sky-400 font-semibold">{personName}</span>
          </h3>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          {sortedTracks.length} Camera {sortedTracks.length === 1 ? 'Hop' : 'Hops'}
        </span>
      </div>

      {/* Horizontal Timeline Strip */}
      <div className="flex items-center space-x-2 overflow-x-auto py-1">
        {sortedTracks.map((track, idx) => {
          const isSelected = selectedTrackId === track.track_id
          const matchingAlert = alerts.find(
            (a) => a.camera_id === track.camera_id && a.track_id === track.track_id
          )
          const camGeo = CAMERA_COORDINATES[track.camera_id]

          return (
            <React.Fragment key={track.id || track.track_id}>
              {idx > 0 && (
                <div className="flex items-center text-slate-600 px-1 shrink-0">
                  <ArrowRight className="w-4 h-4 text-sky-500/70 animate-pulse" />
                </div>
              )}

              <div
                onClick={() => onSelectTrackNode(track, matchingAlert)}
                className={`p-2.5 rounded border transition cursor-pointer shrink-0 min-w-[170px] ${
                  isSelected
                    ? 'bg-slate-800 border-sky-400 ring-1 ring-sky-400'
                    : 'bg-slate-900/90 hover:bg-slate-850 border-slate-800'
                }`}
              >
                {/* Node Header: Camera & Zone */}
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold font-mono text-slate-100 px-1.5 py-0.5 bg-slate-800 rounded">
                    {track.camera_id}
                  </span>
                  <span className="text-[10px] text-slate-400 truncate max-w-[90px]" title={camGeo?.name}>
                    {camGeo?.name || 'CCTV'}
                  </span>
                </div>

                {/* Score & Time */}
                <div className="flex items-center justify-between text-xs mt-1 font-mono">
                  <span className="text-sky-400 font-bold">
                    {(track.max_similarity * 100).toFixed(1)}%
                  </span>
                  <span className="text-slate-400 text-[11px] flex items-center space-x-1">
                    <Clock className="w-3 h-3 text-slate-500" />
                    <span>{track.first_seen}s</span>
                  </span>
                </div>

                {/* Track ID & Detections count */}
                <div className="flex items-center justify-between text-[10px] text-slate-500 mt-1 font-mono">
                  <span>{track.track_id}</span>
                  <span>{track.detection_count} dets</span>
                </div>
              </div>
            </React.Fragment>
          )
        })}
      </div>
    </div>
  )
}
