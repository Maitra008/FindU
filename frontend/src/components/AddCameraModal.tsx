import React, { useState } from 'react'
import type { CameraSourceType, CameraTestResult } from '../types'
import { createCamera, testCameraSource } from '../api/client'

interface AddCameraModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

export const AddCameraModal: React.FC<AddCameraModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [cameraId, setCameraId] = useState('')
  const [name, setName] = useState('')
  const [sourceType, setSourceType] = useState<CameraSourceType>('rtsp')
  const [source, setSource] = useState('')
  const [location, setLocation] = useState('')
  const [sampleFps, setSampleFps] = useState<number>(2.0)
  const [transport, setTransport] = useState<'tcp' | 'udp'>('tcp')
  const [enabled, setEnabled] = useState(true)

  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<CameraTestResult | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const handleTestConnection = async () => {
    if (!source.trim()) {
      setError('Please provide a source URL, device index, or file path to test')
      return
    }
    setTesting(true)
    setError(null)
    setTestResult(null)
    try {
      const res = await testCameraSource({
        source: source.trim(),
        source_type: sourceType,
        transport,
      })
      setTestResult(res)
      if (!res.success && res.error) {
        setError(`Probe Error: ${res.error}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection test failed')
    } finally {
      setTesting(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!cameraId.trim() || !name.trim() || !source.trim()) {
      setError('Camera ID, Name, and Source are required')
      return
    }

    setSaving(true)
    setError(null)
    try {
      await createCamera({
        camera_id: cameraId.trim().toUpperCase(),
        name: name.trim(),
        source: source.trim(),
        source_type: sourceType,
        location: location.trim() || undefined,
        enabled,
        sample_fps: sampleFps,
        transport,
      })
      onSuccess()
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to register camera')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm overflow-y-auto">
      <div className="relative w-full max-w-2xl rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl my-8">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-emerald-500"></span>
              Add Camera Feed
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Register RTSP stream, USB webcam, or pre-recorded video source into the network.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition"
          >
            ✕
          </button>
        </div>

        {error && (
          <div className="mt-4 rounded-lg bg-red-950/60 border border-red-800 p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          {/* Source Type Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Feed Source Type
            </label>
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => {
                  setSourceType('rtsp')
                  if (!source.startsWith('rtsp://')) setSource('rtsp://')
                }}
                className={`flex flex-col items-center justify-center p-3 rounded-lg border text-sm font-medium transition ${
                  sourceType === 'rtsp'
                    ? 'border-indigo-500 bg-indigo-950/40 text-indigo-300 ring-1 ring-indigo-500'
                    : 'border-slate-800 bg-slate-800/40 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <span className="text-lg mb-1">📡</span>
                <span>RTSP Stream</span>
                <span className="text-[10px] text-slate-400">IP Camera / NVR</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setSourceType('usb')
                  setSource('0')
                }}
                className={`flex flex-col items-center justify-center p-3 rounded-lg border text-sm font-medium transition ${
                  sourceType === 'usb'
                    ? 'border-indigo-500 bg-indigo-950/40 text-indigo-300 ring-1 ring-indigo-500'
                    : 'border-slate-800 bg-slate-800/40 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <span className="text-lg mb-1">📷</span>
                <span>USB Webcam</span>
                <span className="text-[10px] text-slate-400">Local Device (0, 1)</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setSourceType('file')
                  if (!source || source.startsWith('rtsp://') || source === '0') {
                    setSource('data/videos/camera_1.mp4')
                  }
                }}
                className={`flex flex-col items-center justify-center p-3 rounded-lg border text-sm font-medium transition ${
                  sourceType === 'file'
                    ? 'border-indigo-500 bg-indigo-950/40 text-indigo-300 ring-1 ring-indigo-500'
                    : 'border-slate-800 bg-slate-800/40 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <span className="text-lg mb-1">🎞️</span>
                <span>Video File</span>
                <span className="text-[10px] text-slate-400">MP4 / AVI Simulation</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Camera ID <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                required
                placeholder="e.g. C5, CAM-NORTH"
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Display Name <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                required
                placeholder="e.g. West Perimeter Gate"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
              {sourceType === 'rtsp'
                ? 'RTSP Stream URL'
                : sourceType === 'usb'
                ? 'Device Index'
                : 'Video File Path'}{' '}
              <span className="text-red-400">*</span>
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                required
                placeholder={
                  sourceType === 'rtsp'
                    ? 'rtsp://user:password@192.168.1.100:554/stream1'
                    : sourceType === 'usb'
                    ? '0 (primary webcam) or 1 (secondary)'
                    : 'data/videos/camera_1.mp4'
                }
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white font-mono placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
              <button
                type="button"
                onClick={handleTestConnection}
                disabled={testing || !source.trim()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-500/50 bg-indigo-950/60 px-4 py-2 text-xs font-semibold text-indigo-300 hover:bg-indigo-900/80 disabled:opacity-50 transition"
              >
                {testing ? (
                  <>
                    <span className="animate-spin text-xs">🌀</span>
                    Testing...
                  </>
                ) : (
                  <>
                    <span>⚡</span>
                    Test Probe
                  </>
                )}
              </button>
            </div>
            {sourceType === 'rtsp' && (
              <p className="text-[11px] text-slate-400 mt-1">
                Passwords are automatically masked for operators and never broadcast.
              </p>
            )}
          </div>

          {/* Test Result Indicator */}
          {testResult && (
            <div
              className={`rounded-lg border p-3 text-xs flex items-center justify-between ${
                testResult.success
                  ? 'border-emerald-800 bg-emerald-950/40 text-emerald-300'
                  : 'border-red-800 bg-red-950/40 text-red-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-base">{testResult.success ? '✅' : '❌'}</span>
                <div>
                  <div className="font-semibold">
                    {testResult.success ? 'Feed Connected Successfully' : 'Feed Probe Failed'}
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Latency: {testResult.latency_ms}ms
                    {testResult.fps ? ` • Stream FPS: ${testResult.fps}` : ''}
                    {testResult.width ? ` • Res: ${testResult.width}x${testResult.height}` : ''}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Physical Location
              </label>
              <input
                type="text"
                placeholder="e.g. Building B - Level 2"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  AI Sampling Rate
                </label>
                <span className="text-xs font-bold text-indigo-400">{sampleFps} FPS</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.5"
                value={sampleFps}
                onChange={(e) => setSampleFps(parseFloat(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                <span>0.5 (Low CPU)</span>
                <span>2.0 (Standard)</span>
                <span>10.0 (High Speed)</span>
              </div>
            </div>
          </div>

          {sourceType === 'rtsp' && (
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                RTSP Network Transport
              </label>
              <div className="flex gap-4 text-xs text-slate-300">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="radio"
                    name="transport"
                    value="tcp"
                    checked={transport === 'tcp'}
                    onChange={() => setTransport('tcp')}
                    className="accent-indigo-500"
                  />
                  <span>TCP (Recommended — zero packet drop)</span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="radio"
                    name="transport"
                    value="udp"
                    checked={transport === 'udp'}
                    onChange={() => setTransport('udp')}
                    className="accent-indigo-500"
                  />
                  <span>UDP (Low latency)</span>
                </label>
              </div>
            </div>
          )}

          <div className="flex items-center gap-2 pt-2">
            <input
              type="checkbox"
              id="enable_now"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              className="rounded accent-indigo-500 h-4 w-4"
            />
            <label htmlFor="enable_now" className="text-xs font-medium text-slate-300 cursor-pointer">
              Enable camera worker immediately upon registration
            </label>
          </div>

          <div className="flex items-center justify-end gap-3 border-t border-slate-800 pt-4 mt-6">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700 transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2 text-xs font-bold text-white hover:bg-indigo-500 disabled:opacity-50 transition"
            >
              {saving ? 'Registering...' : 'Register Camera Feed'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
