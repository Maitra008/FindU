import React, { useState } from 'react'
import { AlertCircle, Clock, Filter, User, Video } from 'lucide-react'
import type { AlertRecord, AlertStatus } from '../types'

interface AlertFeedProps {
  alerts: AlertRecord[]
  selectedAlert: AlertRecord | null
  onSelectAlert: (alert: AlertRecord) => void
  isLoading: boolean
}

export const AlertFeed: React.FC<AlertFeedProps> = ({
  alerts,
  selectedAlert,
  onSelectAlert,
  isLoading,
}) => {
  const [statusFilter, setStatusFilter] = useState<string>('ALL')

  const filteredAlerts = alerts.filter((a) => {
    if (statusFilter === 'ALL') return true
    return a.status === statusFilter
  })

  const getStatusBadge = (status: AlertStatus) => {
    switch (status) {
      case 'NEW':
        return (
          <span className="px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-amber-950 text-amber-300 border border-amber-800 rounded">
            NEW
          </span>
        )
      case 'VERIFIED':
        return (
          <span className="px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800 rounded">
            VERIFIED
          </span>
        )
      case 'DISMISSED':
        return (
          <span className="px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700 rounded">
            DISMISSED
          </span>
        )
    }
  }

  return (
    <div className="flex flex-col h-full bg-slate-900/90 border-r border-slate-800 w-80 lg:w-96 select-none shrink-0">
      {/* Feed Header */}
      <div className="p-3 border-b border-slate-800 flex items-center justify-between bg-slate-950">
        <div className="flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-sky-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Alert Feed ({filteredAlerts.length})
          </h2>
        </div>

        {/* Filter dropdown */}
        <div className="flex items-center space-x-1 text-xs">
          <Filter className="w-3 h-3 text-slate-400" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-900 text-slate-300 text-xs border border-slate-700 rounded px-1.5 py-0.5 focus:outline-none focus:border-sky-500 cursor-pointer"
          >
            <option value="ALL">All ({alerts.length})</option>
            <option value="NEW">New</option>
            <option value="VERIFIED">Verified</option>
            <option value="DISMISSED">Dismissed</option>
          </select>
        </div>
      </div>

      {/* Alert List */}
      <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60">
        {isLoading && alerts.length === 0 ? (
          <div className="p-6 text-center text-xs text-slate-500">Loading alerts...</div>
        ) : filteredAlerts.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-500">
            No alerts match current filter.
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const isSelected = selectedAlert?.id === alert.id
            return (
              <div
                key={alert.id}
                onClick={() => onSelectAlert(alert)}
                className={`p-3 transition cursor-pointer border-l-2 ${
                  isSelected
                    ? 'bg-slate-800/90 border-sky-500'
                    : 'bg-slate-900 hover:bg-slate-850 border-transparent'
                }`}
              >
                {/* Person Name & Status */}
                <div className="flex items-start justify-between mb-1">
                  <div className="flex items-center space-x-1.5">
                    <User className="w-3.5 h-3.5 text-slate-400" />
                    <span className="text-sm font-semibold text-slate-100 truncate">
                      {alert.person_name}
                    </span>
                  </div>
                  {getStatusBadge(alert.status)}
                </div>

                {/* Similarity & Camera */}
                <div className="flex items-center justify-between text-xs text-slate-400 mt-1">
                  <div className="flex items-center space-x-1">
                    <Video className="w-3 h-3 text-slate-500" />
                    <span className="font-mono text-slate-300">{alert.camera_id}</span>
                  </div>

                  <div className="flex items-center space-x-1 font-mono">
                    <span className="text-slate-500 text-[10px]">SIM:</span>
                    <span
                      className={`font-bold ${
                        alert.similarity >= 0.89 ? 'text-sky-400' : 'text-slate-400'
                      }`}
                    >
                      {(alert.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Track ID & Timestamp */}
                <div className="flex items-center justify-between text-[11px] text-slate-500 mt-1.5 font-mono">
                  <span className="truncate">{alert.track_id}</span>
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3 h-3 text-slate-600" />
                    <span>{alert.timestamp}s</span>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
