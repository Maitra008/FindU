import React from 'react'
import { Camera as CameraIcon, LogOut, Play, RefreshCw, Wifi, WifiOff } from 'lucide-react'
import type { WorkerMetrics } from '../types'

interface HeaderProps {
  backendConnected: boolean
  wsConnected: boolean
  workers: WorkerMetrics[]
  onRefresh: () => void
  onSeedDemo: () => void
  isSeeding: boolean
  currentUsername: string | null
  currentRole: string | null
  onLogout: () => void
}

const ROLE_COLORS: Record<string, string> = {
  ADMIN: 'bg-red-950 text-red-300 border-red-800/60',
  POLICE: 'bg-blue-950 text-blue-300 border-blue-800/60',
  HOSPITAL: 'bg-emerald-950 text-emerald-300 border-emerald-800/60',
  NGO: 'bg-purple-950 text-purple-300 border-purple-800/60',
}

export const Header: React.FC<HeaderProps> = ({
  backendConnected,
  wsConnected,
  workers,
  onRefresh,
  onSeedDemo,
  isSeeding,
  currentUsername,
  currentRole,
  onLogout,
}) => {
  const cameraIds = ['C1', 'C2', 'C3', 'C4']
  const roleColor = currentRole ? (ROLE_COLORS[currentRole] ?? 'bg-slate-800 text-slate-400 border-slate-700') : ''

  return (
    <header className="bg-slate-900 border-b border-slate-800 px-4 py-2.5 flex items-center justify-between select-none">
      {/* Brand & System Mode */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 bg-sky-500 rounded-full animate-pulse" />
          <h1 className="text-base font-bold tracking-wider text-slate-100 uppercase">
            FindU <span className="text-slate-400 font-normal text-xs ml-1">Operator Console</span>
          </h1>
        </div>
        <span className="text-xs px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700 font-mono">
          Threshold: 0.89
        </span>
      </div>

      {/* Camera Live Status Bar */}
      <div className="hidden md:flex items-center space-x-3 bg-slate-950 px-3 py-1 rounded-md border border-slate-800">
        <div className="text-xs font-semibold text-slate-400 flex items-center space-x-1">
          <CameraIcon className="w-3.5 h-3.5" />
          <span>Feeds:</span>
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono">
          {cameraIds.map((cid) => {
            const w = workers.find((item) => item.camera_id === cid)
            const isOnline = w && w.status === 'RUNNING'
            return (
              <div
                key={cid}
                className={`flex items-center space-x-1 px-2 py-0.5 rounded ${
                  isOnline
                    ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/60'
                    : 'bg-slate-900 text-slate-500 border border-slate-800'
                }`}
                title={w ? `${w.name} (${w.status}) - ${w.fps} FPS` : `${cid} Offline`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-slate-600'}`} />
                <span className="font-bold">{cid}</span>
                {isOnline && <span className="text-[10px] text-emerald-400/80">{w.fps}f</span>}
              </div>
            )
          })}
        </div>
      </div>

      {/* Connection Health, User Info & Actions */}
      <div className="flex items-center space-x-3">
        {/* WebSocket Status */}
        <div
          className={`flex items-center space-x-1.5 text-xs px-2 py-1 rounded ${
            wsConnected
              ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/50'
              : 'bg-amber-950 text-amber-400 border border-amber-800/50'
          }`}
          title={wsConnected ? 'WebSocket Live Feed Connected' : 'WebSocket Disconnected'}
        >
          {wsConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
          <span className="font-mono">{wsConnected ? 'LIVE' : 'DISCONNECTED'}</span>
        </div>

        {/* User Role Badge */}
        {currentUsername && currentRole && (
          <div className={`hidden sm:flex items-center space-x-1.5 text-xs px-2 py-1 rounded border ${roleColor}`}>
            <span className="font-medium">{currentUsername}</span>
            <span className="opacity-60">·</span>
            <span className="font-mono">{currentRole}</span>
          </div>
        )}

        {/* Demo Action Button */}
        <button
          onClick={onSeedDemo}
          disabled={isSeeding || !backendConnected}
          className="flex items-center space-x-1 text-xs px-2.5 py-1 bg-indigo-950 hover:bg-indigo-900 text-indigo-300 border border-indigo-700/60 rounded transition disabled:opacity-50 cursor-pointer"
          title="Seed controlled C2 → C3 → C1 movement scenario"
        >
          <Play className="w-3 h-3 text-indigo-400" />
          <span>Demo Movement (C2→C3→C1)</span>
        </button>

        {/* Refresh button */}
        <button
          onClick={onRefresh}
          className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition cursor-pointer"
          title="Refresh feeds and telemetry"
        >
          <RefreshCw className="w-4 h-4" />
        </button>

        {/* Logout button */}
        <button
          onClick={onLogout}
          className="p-1 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded transition cursor-pointer"
          title="Sign out"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}

