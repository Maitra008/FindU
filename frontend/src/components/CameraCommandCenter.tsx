import React, { useEffect, useState, useCallback } from 'react'
import {
  Activity,
  AlertTriangle,
  Camera as CameraIcon,
  CheckCircle2,
  FastForward,
  Film,
  MapPin,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  Upload,
  Video,
  WifiOff,
  X,
  XCircle,
  Zap,
} from 'lucide-react'
import type { Camera, HistoricalJobRecord } from '../types'
import type { UserInfo } from '../api/client'
import {
  cancelHistoricalJob,
  deleteCamera,
  fetchHistoricalJobs,
  pauseWorker,
  resumeWorker,
  setWorkerSpeed,
  testExistingCamera,
  updateCamera,
} from '../api/client'
import { AddCameraModal } from './AddCameraModal'
import { HistoricalUploadModal } from './HistoricalUploadModal'

interface CameraCommandCenterProps {
  cameras: Camera[]
  currentUser: UserInfo | null
  onRefreshCameras: () => void
}

export const CameraCommandCenter: React.FC<CameraCommandCenterProps> = ({
  cameras,
  currentUser,
  onRefreshCameras,
}) => {
  const [selectedSourceType, setSelectedSourceType] = useState<string>('ALL')
  const [selectedStatus, setSelectedStatus] = useState<string>('ALL')
  const [searchQuery, setSearchQuery] = useState('')

  const [isAddModalOpen, setIsAddModalOpen] = useState(false)
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)

  const [testingCameraId, setTestingCameraId] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, any>>({})
  const [historicalJobs, setHistoricalJobs] = useState<HistoricalJobRecord[]>([])
  const [loadingJobs, setLoadingJobs] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)

  const isAdmin = currentUser?.role === 'ADMIN'

  const loadJobs = useCallback(async () => {
    try {
      setLoadingJobs(true)
      const jobs = await fetchHistoricalJobs()
      setHistoricalJobs(jobs)
    } catch {
      // safe fallback
    } finally {
      setLoadingJobs(false)
    }
  }, [])

  useEffect(() => {
    loadJobs()
    const interval = setInterval(() => {
      loadJobs()
      onRefreshCameras()
    }, 5000)
    return () => clearInterval(interval)
  }, [loadJobs, onRefreshCameras])

  const handleTestCamera = async (cameraId: string) => {
    setTestingCameraId(cameraId)
    setActionError(null)
    try {
      const res = await testExistingCamera(cameraId)
      setTestResults((prev) => ({ ...prev, [cameraId]: res }))
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Test failed')
    } finally {
      setTestingCameraId(null)
    }
  }

  const handleToggleEnable = async (cam: Camera) => {
    try {
      await updateCamera(cam.camera_id, { enabled: !cam.enabled })
      onRefreshCameras()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to update camera status')
    }
  }

  const handleDeleteCamera = async (cameraId: string) => {
    if (!window.confirm(`Are you sure you want to remove Camera ${cameraId}?`)) return
    try {
      await deleteCamera(cameraId)
      onRefreshCameras()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to delete camera')
    }
  }

  const handlePauseResume = async (cam: Camera, isPaused: boolean) => {
    try {
      if (isPaused) {
        await resumeWorker(cam.camera_id)
      } else {
        await pauseWorker(cam.camera_id)
      }
      onRefreshCameras()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Replay control failed')
    }
  }

  const handleSpeedChange = async (cameraId: string, speed: number) => {
    try {
      await setWorkerSpeed(cameraId, speed)
      onRefreshCameras()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Speed control failed')
    }
  }

  const handleCancelJob = async (jobId: string) => {
    try {
      await cancelHistoricalJob(jobId)
      loadJobs()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to cancel job')
    }
  }

  // Filter cameras
  const filteredCameras = cameras.filter((cam) => {
    if (selectedSourceType !== 'ALL' && cam.source_type !== selectedSourceType) return false
    const status = cam.status || (cam.enabled ? 'ONLINE' : 'OFFLINE')
    if (selectedStatus !== 'ALL' && status !== selectedStatus) return false
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      return (
        cam.camera_id.toLowerCase().includes(q) ||
        cam.name.toLowerCase().includes(q) ||
        (cam.location && cam.location.toLowerCase().includes(q))
      )
    }
    return true
  })

  // Summary counts
  const totalCameras = cameras.length
  const onlineCameras = cameras.filter(
    (c) => (c.status || (c.enabled ? 'ONLINE' : 'OFFLINE')) === 'ONLINE' || c.status === 'RUNNING'
  ).length
  const processingCameras = cameras.filter((c) => c.status === 'PROCESSING' || c.status === 'CONNECTING').length
  const totalMatches = cameras.reduce((acc, c) => acc + (c.alerts_emitted || 0), 0)

  return (
    <div className="flex-1 h-full overflow-y-auto p-6 space-y-6 bg-slate-950 text-slate-100">
      {/* Top Network Status Summary Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-sm backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Total Feeds</span>
            <div className="rounded-lg bg-slate-800 p-2 text-slate-300">
              <CameraIcon className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 text-2xl font-bold text-white font-mono">{totalCameras}</div>
          <div className="text-[11px] text-slate-400 mt-0.5">RTSP, USB & File Ingestion</div>
        </div>

        <div className="rounded-xl border border-emerald-950/60 bg-emerald-950/20 p-4 shadow-sm backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400">Active / Online</span>
            <span className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
          </div>
          <div className="mt-2 text-2xl font-bold text-emerald-300 font-mono">{onlineCameras}</div>
          <div className="text-[11px] text-emerald-400/80 mt-0.5">Real-time inference active</div>
        </div>

        <div className="rounded-xl border border-sky-950/60 bg-sky-950/20 p-4 shadow-sm backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-sky-400">Deep Processing</span>
            <div className="rounded-lg bg-sky-950 p-2 text-sky-400">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 text-2xl font-bold text-sky-300 font-mono">{processingCameras}</div>
          <div className="text-[11px] text-sky-400/80 mt-0.5">AI ingestion & sampling</div>
        </div>

        <div className="rounded-xl border border-purple-950/60 bg-purple-950/20 p-4 shadow-sm backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-purple-400">Alert Matches</span>
            <div className="rounded-lg bg-purple-950 p-2 text-purple-400">
              <Zap className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 text-2xl font-bold text-purple-300 font-mono">{totalMatches}</div>
          <div className="text-[11px] text-purple-400/80 mt-0.5">Cross-camera detections</div>
        </div>
      </div>

      {actionError && (
        <div className="rounded-lg bg-red-950/60 border border-red-800 p-3 text-sm text-red-300 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span>{actionError}</span>
          </div>
          <button onClick={() => setActionError(null)} className="text-red-400 hover:text-white p-1">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Control & Filter Bar */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur">
        <div className="flex flex-wrap items-center gap-3">
          {/* Source Type Filter */}
          <div className="flex items-center gap-1 rounded-lg bg-slate-950 p-1 border border-slate-800 text-xs">
            {['ALL', 'rtsp', 'usb', 'file'].map((type) => (
              <button
                key={type}
                onClick={() => setSelectedSourceType(type)}
                className={`rounded-md px-3 py-1 font-semibold transition cursor-pointer ${
                  selectedSourceType === type
                    ? 'bg-sky-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {type === 'ALL' ? 'All Types' : type.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Status Filter */}
          <div className="flex items-center gap-1 rounded-lg bg-slate-950 p-1 border border-slate-800 text-xs">
            {['ALL', 'ONLINE', 'PROCESSING', 'OFFLINE'].map((st) => (
              <button
                key={st}
                onClick={() => setSelectedStatus(st)}
                className={`rounded-md px-2.5 py-1 font-semibold transition cursor-pointer ${
                  selectedStatus === st
                    ? 'bg-sky-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {st}
              </button>
            ))}
          </div>

          {/* Search Input */}
          <div className="relative min-w-[220px]">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search camera name, ID, location..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:border-sky-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onRefreshCameras()}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white transition cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>

          {isAdmin && (
            <>
              <button
                onClick={() => setIsUploadModalOpen(true)}
                className="flex items-center gap-1.5 rounded-lg border border-sky-500/40 bg-sky-950/60 px-3 py-1.5 text-xs font-semibold text-sky-300 hover:bg-sky-900 transition cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Upload Footage</span>
              </button>

              <button
                onClick={() => setIsAddModalOpen(true)}
                className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3.5 py-1.5 text-xs font-bold text-white hover:bg-sky-500 shadow-md transition cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add Camera</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Camera Feeds Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredCameras.map((cam) => {
          const status = cam.status || (cam.enabled ? 'ONLINE' : 'OFFLINE')
          const isOnline = status === 'ONLINE' || status === 'RUNNING'
          const isProcessing = status === 'PROCESSING' || status === 'CONNECTING'
          const testRes = testResults[cam.camera_id]

          return (
            <div
              key={cam.camera_id}
              className="flex flex-col rounded-xl border border-slate-800 bg-slate-900/90 shadow-lg overflow-hidden transition hover:border-slate-700"
            >
              {/* Card Header */}
              <div className="flex items-center justify-between border-b border-slate-800/80 bg-slate-950/40 px-4 py-3">
                <div className="flex items-center gap-2.5">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${
                      isOnline
                        ? 'bg-emerald-500 animate-pulse'
                        : isProcessing
                        ? 'bg-sky-500 animate-pulse'
                        : 'bg-slate-600'
                    }`}
                  />
                  <div>
                    <span className="font-mono text-xs font-bold text-sky-400">[{cam.camera_id}]</span>{' '}
                    <span className="text-sm font-semibold text-slate-200">{cam.name}</span>
                  </div>
                </div>

                <div className="flex items-center gap-1.5">
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-mono font-bold uppercase text-sky-300 border border-slate-700">
                    {cam.source_type === 'rtsp' ? '📡 RTSP' : cam.source_type === 'usb' ? '📷 USB' : '🎞️ FILE'}
                  </span>
                  <span
                    className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase ${
                      isOnline
                        ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800'
                        : isProcessing
                        ? 'bg-sky-950/80 text-sky-300 border border-sky-800'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}
                  >
                    {status}
                  </span>
                </div>
              </div>

              {/* Feed Video / Simulation Container */}
              <div className="relative aspect-video w-full bg-slate-950 flex items-center justify-center border-b border-slate-800">
                {isOnline ? (
                  <div className="relative w-full h-full flex items-center justify-center overflow-hidden bg-slate-950">
                    <div className="text-center p-4">
                      <Video className="w-10 h-10 mx-auto mb-2 text-emerald-500 opacity-80" />
                      <p className="text-xs font-mono text-emerald-400 font-bold">FEED LIVE & INGESTING</p>
                      <p className="text-[11px] text-slate-400">{cam.location || 'Surveillance Channel'}</p>
                    </div>

                    {/* HUD Telemetry Overlay */}
                    <div className="absolute top-2 left-2 flex flex-col gap-1">
                      <div className="rounded bg-black/75 px-2 py-0.5 text-[10px] font-mono text-emerald-400 border border-emerald-900/50 backdrop-blur-sm">
                        FPS: {cam.stream_fps ? cam.stream_fps.toFixed(1) : '30.0'} • AI: {(cam.sample_fps ?? 2.0).toFixed(1)} FPS
                      </div>
                      {cam.latency_ms !== undefined && cam.latency_ms !== null && (
                        <div className="rounded bg-black/75 px-2 py-0.5 text-[10px] font-mono text-slate-300 border border-slate-800 backdrop-blur-sm">
                          Latency: {cam.latency_ms.toFixed(0)} ms
                        </div>
                      )}
                    </div>

                    <div className="absolute bottom-2 right-2 flex gap-1">
                      <div className="rounded bg-black/75 px-2 py-0.5 text-[10px] font-mono text-sky-300 border border-sky-900/50 backdrop-blur-sm">
                        Tracks: {cam.tracks_created || 0} • Alerts: {cam.alerts_emitted || 0}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center p-4 text-slate-400">
                    <WifiOff className="w-8 h-8 mx-auto mb-1 text-slate-600" />
                    <p className="text-xs font-semibold">Feed Standby / Offline</p>
                    <p className="text-[10px] text-slate-400 mt-0.5">Toggle enable to begin worker ingestion</p>
                  </div>
                )}
              </div>

              {/* Channel Meta & Live Info */}
              <div className="p-4 space-y-3 flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                    <span className="flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-slate-500" />
                      <span>Source Location:</span>
                    </span>
                    <span className="text-slate-200 font-medium">{cam.location || 'Unspecified'}</span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400 truncate bg-slate-950 px-2 py-1 rounded border border-slate-800">
                    {cam.source}
                  </div>
                </div>

                {/* Inline Connection Test Status */}
                {testRes && (
                  <div
                    className={`rounded p-2 text-[11px] border flex items-center gap-2 ${
                      testRes.success
                        ? 'border-emerald-800 bg-emerald-950/40 text-emerald-300'
                        : 'border-red-800 bg-red-950/40 text-red-300'
                    }`}
                  >
                    {testRes.success ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                    )}
                    <span>
                      {testRes.success
                        ? `Probe OK (${testRes.latency_ms}ms, ${testRes.fps ?? 30} FPS, ${testRes.width ?? 0}x${testRes.height ?? 0})`
                        : `Probe Error: ${testRes.error}`}
                    </span>
                  </div>
                )}

                {/* Replay Controls for File Sources */}
                {cam.source_type === 'file' && isOnline && (
                  <div className="flex items-center justify-between bg-slate-950/60 p-2 rounded-lg border border-slate-800 text-xs">
                    <span className="text-[11px] text-slate-400 font-semibold flex items-center gap-1">
                      <Film className="w-3 h-3" />
                      <span>Playback:</span>
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => handlePauseResume(cam, false)}
                        className="flex items-center gap-1 rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-700 cursor-pointer"
                        title="Pause Ingestion"
                      >
                        <Pause className="w-2.5 h-2.5" />
                        <span>Pause</span>
                      </button>
                      <button
                        onClick={() => handlePauseResume(cam, true)}
                        className="flex items-center gap-1 rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-700 cursor-pointer"
                        title="Resume Ingestion"
                      >
                        <Play className="w-2.5 h-2.5" />
                        <span>Resume</span>
                      </button>
                      <button
                        onClick={() => handleSpeedChange(cam.camera_id, 1.0)}
                        className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 hover:bg-slate-700 cursor-pointer"
                      >
                        1x
                      </button>
                      <button
                        onClick={() => handleSpeedChange(cam.camera_id, 2.0)}
                        className="flex items-center gap-0.5 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 hover:bg-slate-700 cursor-pointer"
                      >
                        <FastForward className="w-2.5 h-2.5" />
                        <span>2x</span>
                      </button>
                    </div>
                  </div>
                )}

                {/* Card Action Controls */}
                <div className="flex items-center justify-between border-t border-slate-800 pt-3 text-xs">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleTestCamera(cam.camera_id)}
                      disabled={testingCameraId === cam.camera_id}
                      className="flex items-center gap-1 rounded border border-slate-700 bg-slate-800 px-2.5 py-1 font-semibold text-slate-300 hover:bg-slate-700 disabled:opacity-50 transition cursor-pointer"
                    >
                      <Zap className="w-3 h-3 text-amber-400" />
                      <span>{testingCameraId === cam.camera_id ? 'Probing...' : 'Probe'}</span>
                    </button>

                    {isAdmin && (
                      <button
                        onClick={() => handleToggleEnable(cam)}
                        className={`rounded px-2.5 py-1 font-semibold transition cursor-pointer ${
                          cam.enabled
                            ? 'bg-amber-950/60 text-amber-300 border border-amber-800 hover:bg-amber-900'
                            : 'bg-emerald-950/60 text-emerald-300 border border-emerald-800 hover:bg-emerald-900'
                        }`}
                      >
                        {cam.enabled ? 'Disable' : 'Enable'}
                      </button>
                    )}
                  </div>

                  {isAdmin && (
                    <button
                      onClick={() => handleDeleteCamera(cam.camera_id)}
                      className="flex items-center gap-1 text-red-400 hover:text-red-300 text-xs font-semibold px-2 py-1 rounded hover:bg-red-950/40 transition cursor-pointer"
                    >
                      <Trash2 className="w-3 h-3" />
                      <span>Delete</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Historical Background Jobs Drawer */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 backdrop-blur">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Film className="w-4 h-4 text-sky-400" />
              <span>Historical Footage Analysis Jobs</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Asynchronous background scans over recorded CCTV archives and deep face matching runs.
            </p>
          </div>
          <button
            onClick={loadJobs}
            disabled={loadingJobs}
            className="flex items-center gap-1.5 rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs text-slate-300 hover:text-white cursor-pointer"
          >
            <RefreshCw className={`w-3 h-3 ${loadingJobs ? 'animate-spin' : ''}`} />
            <span>{loadingJobs ? 'Refreshing...' : 'Refresh Jobs'}</span>
          </button>
        </div>

        {historicalJobs.length === 0 ? (
          <div className="text-center py-6 text-xs text-slate-400">
            No historical search jobs currently queued or running. Upload footage above to run deep historical scans.
          </div>
        ) : (
          <div className="mt-4 space-y-3">
            {historicalJobs.map((job) => {
              const isRunning = job.status === 'PROCESSING' || job.status === 'QUEUED'
              return (
                <div
                  key={job.job_id}
                  className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 flex flex-col md:flex-row md:items-center justify-between gap-3"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold text-sky-400">{job.job_id}</span>
                      <span className="text-xs text-slate-300 font-semibold">• Cam {job.camera_id}</span>
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                          job.status === 'COMPLETED'
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                            : job.status === 'PROCESSING'
                            ? 'bg-sky-950 text-sky-300 border border-sky-800 animate-pulse'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {job.status}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400">
                      File: <span className="font-mono text-slate-300">{job.file_path}</span> • Faces Found:{' '}
                      <span className="text-white font-bold">{job.faces_detected}</span> • Potential Matches:{' '}
                      <span className="text-emerald-400 font-bold">{job.potential_matches}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="w-32">
                      <div className="flex justify-between text-[10px] text-slate-400 mb-0.5">
                        <span>Progress</span>
                        <span>{job.progress_percent.toFixed(0)}%</span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                        <div
                          className="h-full bg-sky-500 transition-all duration-300"
                          style={{ width: `${job.progress_percent}%` }}
                        ></div>
                      </div>
                    </div>

                    {isRunning && isAdmin && (
                      <button
                        onClick={() => handleCancelJob(job.job_id)}
                        className="rounded border border-red-900/60 bg-red-950/40 px-2 py-1 text-[11px] font-semibold text-red-300 hover:bg-red-900/60 cursor-pointer"
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Modals */}
      <AddCameraModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onSuccess={onRefreshCameras}
      />

      <HistoricalUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onSuccess={() => {
          loadJobs()
          onRefreshCameras()
        }}
      />
    </div>
  )
}
