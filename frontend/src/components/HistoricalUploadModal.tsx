import React, { useState } from 'react'
import { uploadHistoricalFootage } from '../api/client'

interface HistoricalUploadModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: (jobInfo: any) => void
}

export const HistoricalUploadModal: React.FC<HistoricalUploadModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [cameraId, setCameraId] = useState('C_HIST_01')
  const [cameraName, setCameraName] = useState('')
  const [location, setLocation] = useState('Archived Footage')
  const [mode, setMode] = useState<'replay' | 'search'>('search')
  const [sampleFps, setSampleFps] = useState<number>(2.0)

  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedFile(file)
      if (!cameraName) {
        setCameraName(`Archive - ${file.name.replace(/\.[^/.]+$/, '')}`)
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedFile) {
      setError('Please select a video file to upload')
      return
    }

    setUploading(true)
    setError(null)
    setUploadProgress(20)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('camera_id', cameraId.trim().toUpperCase())
      if (cameraName) formData.append('camera_name', cameraName.trim())
      if (location) formData.append('location', location.trim())
      formData.append('mode', mode)
      formData.append('sample_fps', sampleFps.toString())

      setUploadProgress(60)
      const res = await uploadHistoricalFootage(formData)
      setUploadProgress(100)
      onSuccess(res)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm overflow-y-auto">
      <div className="relative w-full max-w-2xl rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl my-8">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span className="text-xl">📼</span>
              Upload Historical Footage
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Ingest offline video files (MP4, AVI, MKV, MOV) for deep face matching or live replay.
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
          {/* File Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Video File (Max 2GB) <span className="text-red-400">*</span>
            </label>
            <div className="flex justify-center rounded-lg border border-dashed border-slate-700 px-6 py-6 bg-slate-800/50 hover:border-indigo-500 transition">
              <div className="text-center">
                <span className="text-3xl mb-2 block">📁</span>
                <div className="mt-2 flex text-xs leading-6 text-slate-400 justify-center">
                  <label
                    htmlFor="file-upload"
                    className="relative cursor-pointer rounded-md font-semibold text-indigo-400 hover:text-indigo-300 focus-within:outline-none"
                  >
                    <span>{selectedFile ? selectedFile.name : 'Select a video file'}</span>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      accept=".mp4,.avi,.mkv,.mov,.webm"
                      className="sr-only"
                      onChange={handleFileChange}
                    />
                  </label>
                  {!selectedFile && <p className="pl-1">or drag and drop</p>}
                </div>
                <p className="text-[11px] leading-5 text-slate-400">
                  {selectedFile
                    ? `Size: ${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`
                    : 'MP4, AVI, MKV, MOV up to 2GB'}
                </p>
              </div>
            </div>
          </div>

          {/* Processing Mode Selection */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Processing Mode
            </label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setMode('search')}
                className={`p-3 rounded-lg border text-left transition ${
                  mode === 'search'
                    ? 'border-indigo-500 bg-indigo-950/40 text-indigo-200 ring-1 ring-indigo-500'
                    : 'border-slate-800 bg-slate-800/40 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center gap-2 font-semibold text-sm">
                  <span>🔍</span> Asynchronous Search
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Processes full video rapidly in the background, extracts tracks, and flags potential missing person matches.
                </p>
              </button>

              <button
                type="button"
                onClick={() => setMode('replay')}
                className={`p-3 rounded-lg border text-left transition ${
                  mode === 'replay'
                    ? 'border-indigo-500 bg-indigo-950/40 text-indigo-200 ring-1 ring-indigo-500'
                    : 'border-slate-800 bg-slate-800/40 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center gap-2 font-semibold text-sm">
                  <span>▶️</span> Replay as Live Camera
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Mounts footage as a looping camera channel with real-time speed pacing and live alert broadcasting.
                </p>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Camera Identifier
              </label>
              <input
                type="text"
                required
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Display Label
              </label>
              <input
                type="text"
                placeholder="e.g. CCTV Archive 14-Oct"
                value={cameraName}
                onChange={(e) => setCameraName(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Recorded Location
              </label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Analysis Sample Rate
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
            </div>
          </div>

          {uploading && (
            <div className="space-y-1.5 pt-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Uploading & Initializing...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-indigo-500 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}

          <div className="flex items-center justify-end gap-3 border-t border-slate-800 pt-4 mt-6">
            <button
              type="button"
              onClick={onClose}
              disabled={uploading}
              className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700 transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={uploading || !selectedFile}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2 text-xs font-bold text-white hover:bg-indigo-500 disabled:opacity-50 transition"
            >
              {uploading ? 'Processing...' : 'Upload & Start Processing'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

