import React, { useState } from 'react'
import { AlertTriangle, CheckCircle2, Loader2, Trash2, X } from 'lucide-react'
import { resetTestEnvironment, type ResetEnvironmentResponse } from '../api/client'

interface ResetTestEnvironmentModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

const REQUIRED_CONFIRM_TEXT = 'RESET TEST DATA'

export const ResetTestEnvironmentModal: React.FC<ResetTestEnvironmentModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [confirmInput, setConfirmInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ResetEnvironmentResponse | null>(null)

  if (!isOpen) return null

  const isConfirmed = confirmInput.trim() === REQUIRED_CONFIRM_TEXT

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!isConfirmed || loading) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await resetTestEnvironment()
      setResult(res)
      onSuccess()
    } catch (err: any) {
      setError(err.message || 'Failed to reset test environment.')
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    if (loading) return
    setConfirmInput('')
    setError(null)
    setResult(null)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-red-900/60 rounded-xl shadow-2xl max-w-lg w-full overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-red-900/40 bg-red-950/30 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 bg-red-950/80 border border-red-800/80 rounded-lg text-red-400">
              <AlertTriangle className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">Reset Test Environment</h2>
              <p className="text-xs text-red-400 font-mono">ADMIN PRIVILEGED ACTION</p>
            </div>
          </div>
          {!loading && (
            <button
              onClick={handleClose}
              className="text-slate-400 hover:text-slate-200 transition p-1 rounded-lg hover:bg-slate-800"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Content */}
        <div className="p-5 overflow-y-auto space-y-4">
          {result ? (
            <div className="space-y-4">
              <div className="p-4 bg-emerald-950/50 border border-emerald-800/60 rounded-lg flex items-start space-x-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                <div className="text-xs space-y-1">
                  <p className="font-semibold text-emerald-300 text-sm">{result.message}</p>
                  <p className="text-slate-400">The test environment has been restored to a clean state.</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-1.5">
                  <p className="font-semibold text-red-400 uppercase tracking-wider text-[10px]">Deleted Data</p>
                  <ul className="space-y-1 text-slate-300 font-mono text-[11px]">
                    <li>• Persons: {result.deleted.persons}</li>
                    <li>• Alerts: {result.deleted.alerts}</li>
                    <li>• Tracks: {result.deleted.tracks}</li>
                    <li>• Historical Jobs: {result.deleted.historical_jobs}</li>
                    <li>• Historical Files: {result.deleted.historical_files}</li>
                    <li>• Photo Folders: {result.deleted.reference_photos}</li>
                    <li>• Test Embeddings: {result.deleted.embeddings}</li>
                  </ul>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-1.5">
                  <p className="font-semibold text-emerald-400 uppercase tracking-wider text-[10px]">Preserved Infrastructure</p>
                  <ul className="space-y-1 text-slate-300 font-mono text-[11px]">
                    <li>• Cameras: {result.preserved.cameras}</li>
                    <li>• User Accounts: {result.preserved.users}</li>
                    <li>• Baseline Identities: {result.preserved.baseline_identities}</li>
                    <li>• Recognition Core: Buffalo_L</li>
                    <li>• Cosine Threshold: 0.55</li>
                  </ul>
                </div>
              </div>

              <button
                onClick={handleClose}
                className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium rounded-lg text-xs transition cursor-pointer"
              >
                Done
              </button>
            </div>
          ) : (
            <form onSubmit={handleReset} className="space-y-4">
              <div className="p-3.5 bg-red-950/40 border border-red-900/50 rounded-lg text-xs text-red-300 space-y-2">
                <p className="font-semibold text-red-200">Warning: Destructive Reset Action</p>
                <p className="leading-relaxed text-slate-300">
                  This will permanently delete all test-generated data including:
                </p>
                <ul className="list-disc list-inside space-y-0.5 text-slate-300">
                  <li>Registered missing persons & uploaded reference photos</li>
                  <li>All alert history and track records</li>
                  <li>Historical video analysis jobs & uploaded footage files</li>
                  <li>Dynamic face embeddings (synchronizing FAISS back to baseline)</li>
                </ul>
                <p className="text-emerald-400 text-[11px] pt-1">
                  ✓ Preserves user accounts, camera configs (C1-C4, webcams), and baseline models.
                </p>
              </div>

              {error && (
                <div className="p-3 bg-red-950/80 border border-red-800 text-red-200 text-xs rounded-lg flex items-center space-x-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <div className="space-y-2">
                <label className="block text-xs font-semibold text-slate-300">
                  To confirm, type <span className="font-mono text-red-400 bg-red-950/60 px-1.5 py-0.5 rounded border border-red-900/60">{REQUIRED_CONFIRM_TEXT}</span> below:
                </label>
                <input
                  type="text"
                  value={confirmInput}
                  onChange={(e) => setConfirmInput(e.target.value)}
                  placeholder="RESET TEST DATA"
                  disabled={loading}
                  className="w-full bg-slate-950 border border-slate-700 focus:border-red-500 focus:ring-1 focus:ring-red-500 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono placeholder:text-slate-600 outline-none transition"
                />
              </div>

              <div className="flex items-center justify-end space-x-3 pt-2">
                <button
                  type="button"
                  onClick={handleClose}
                  disabled={loading}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-lg text-xs transition disabled:opacity-50 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!isConfirmed || loading}
                  className="flex items-center space-x-1.5 px-4 py-2 bg-red-600 hover:bg-red-500 text-white font-semibold rounded-lg text-xs transition disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer shadow-lg shadow-red-950"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Resetting Environment...</span>
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>Reset Test Environment</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  )
}
