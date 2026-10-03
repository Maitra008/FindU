/**
 * Missing Persons & Case Management View.
 * Displays searchable cases, reference photo galleries, sighting histories, and case triage.
 */

import React, { useEffect, useState } from 'react'
import {
  Calendar,
  CheckCircle2,
  ExternalLink,
  MapPin,
  RefreshCw,
  Search,
  Shield,
  ShieldCheck,
  Tag,
  User,
  UserCheck,
  UserPlus,
  Video,
  X,
  XCircle,
} from 'lucide-react'
import { fetchPersonDetails, fetchPersons, updateAlertStatus, updatePerson } from '../api/client'
import type { AlertRecord, AlertStatus, PersonCaseStatus, PersonRecord } from '../types'
import { MediaPreview } from './MediaPreview'

interface MissingPersonsViewProps {
  currentRole: string | null
  onOpenRegisterModal: () => void
  onSelectAlertForOperation?: (alert: AlertRecord) => void
  refreshKey?: number
}

const STATUS_BADGES: Record<PersonCaseStatus, string> = {
  ACTIVE: 'bg-emerald-950/80 text-emerald-300 border-emerald-700',
  FOUND: 'bg-blue-950/80 text-blue-300 border-blue-700',
  CLOSED: 'bg-slate-800 text-slate-400 border-slate-700',
}

export const MissingPersonsView: React.FC<MissingPersonsViewProps> = ({
  currentRole,
  onOpenRegisterModal,
  onSelectAlertForOperation,
  refreshKey,
}) => {
  const [persons, setPersons] = useState<PersonRecord[]>([])
  const [selectedPerson, setSelectedPerson] = useState<PersonRecord | null>(null)
  const [statusFilter, setStatusFilter] = useState<string>('ALL')
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [isUpdatingStatus, setIsUpdatingStatus] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const isAuthorized = currentRole === 'ADMIN' || currentRole === 'POLICE'

  const loadPersons = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await fetchPersons({
        status: statusFilter !== 'ALL' ? statusFilter : undefined,
        search: searchQuery.trim() || undefined,
      })
      setPersons(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch missing persons')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadPersons()
  }, [statusFilter, refreshKey])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    loadPersons()
  }

  const handleOpenDetails = async (p: PersonRecord) => {
    try {
      const detailed = await fetchPersonDetails(p.person_id)
      setSelectedPerson(detailed)
    } catch {
      setSelectedPerson(p)
    }
  }

  const handleUpdateCaseStatus = async (newStatus: PersonCaseStatus) => {
    if (!selectedPerson || !isAuthorized) return
    setIsUpdatingStatus(true)
    try {
      const updated = await updatePerson(selectedPerson.person_id, { status: newStatus })
      const detailed = await fetchPersonDetails(updated.person_id)
      setSelectedPerson(detailed)
      setPersons((prev) => prev.map((item) => (item.person_id === updated.person_id ? updated : item)))
    } catch (err) {
      alert(`Error updating case status: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsUpdatingStatus(false)
    }
  }

  const handleVerifySighting = async (alertId: string | number, status: AlertStatus) => {
    if (!isAuthorized) return
    try {
      await updateAlertStatus(alertId, status)
      if (selectedPerson) {
        const refreshed = await fetchPersonDetails(selectedPerson.person_id)
        setSelectedPerson(refreshed)
      }
    } catch (err) {
      alert(`Error updating alert: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  return (
    <div className="flex flex-1 h-full overflow-hidden bg-slate-950 text-slate-100">
      {/* Left / Main: Cases List & Grid */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Controls Bar */}
        <div className="p-4 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <UserCheck className="w-5 h-5 text-sky-400" />
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-100">
                Missing Persons Database
              </h2>
            </div>
            <span className="text-xs px-2 py-0.5 bg-slate-800 text-slate-300 rounded-full border border-slate-700 font-mono">
              {persons.length} Cases
            </span>
          </div>

          <div className="flex items-center flex-wrap gap-2.5">
            {/* Search Input */}
            <form onSubmit={handleSearch} className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search name, person ID, case ID..."
                className="bg-slate-950 border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 w-48 sm:w-64 transition"
              />
            </form>

            {/* Status Filter Buttons */}
            <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-xs">
              {['ALL', 'ACTIVE', 'FOUND', 'CLOSED'].map((st) => (
                <button
                  key={st}
                  type="button"
                  onClick={() => setStatusFilter(st)}
                  className={`px-2.5 py-1 rounded-md transition font-medium text-[11px] ${
                    statusFilter === st
                      ? 'bg-blue-600 text-white shadow'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {st}
                </button>
              ))}
            </div>

            {/* Refresh */}
            <button
              type="button"
              onClick={loadPersons}
              className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition"
              title="Refresh case records"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>

            {/* Register Person Button */}
            {isAuthorized && (
              <button
                type="button"
                onClick={onOpenRegisterModal}
                className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-3 py-1.5 rounded-lg transition shadow-md shadow-blue-950/60"
              >
                <UserPlus className="w-3.5 h-3.5" />
                <span>+ Register Person</span>
              </button>
            )}
          </div>
        </div>

        {/* Cases Grid Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-500 gap-2">
              <RefreshCw className="w-6 h-6 animate-spin text-blue-500" />
              <p className="text-xs font-mono">Loading missing person case records…</p>
            </div>
          ) : error ? (
            <div className="bg-red-950/40 border border-red-800 rounded-xl p-4 text-red-300 text-xs text-center">
              {error}
            </div>
          ) : persons.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-500 text-center space-y-3">
              <User className="w-10 h-10 text-slate-700" />
              <div>
                <p className="text-sm font-semibold text-slate-300">No missing person records found</p>
                <p className="text-xs text-slate-500">
                  {statusFilter !== 'ALL'
                    ? `No records matching status "${statusFilter}"`
                    : 'Register an identity with reference photographs to initiate recognition tracking.'}
                </p>
              </div>
              {isAuthorized && (
                <button
                  type="button"
                  onClick={onOpenRegisterModal}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-lg transition"
                >
                  + Register First Missing Person
                </button>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {persons.map((person) => {
                const primaryPhoto = person.photo_paths?.[0]
                const isSelected = selectedPerson?.person_id === person.person_id

                return (
                  <div
                    key={person.person_id}
                    onClick={() => handleOpenDetails(person)}
                    className={`bg-slate-900 border rounded-xl p-4 cursor-pointer transition-all hover:border-slate-600 hover:shadow-lg space-y-3 ${
                      isSelected
                        ? 'border-blue-500 ring-1 ring-blue-500 shadow-blue-950/50'
                        : 'border-slate-800'
                    }`}
                  >
                    {/* Card Top: Thumbnail + Header Info */}
                    <div className="flex gap-3">
                      <div className="w-20 h-20 shrink-0 rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
                        <MediaPreview
                          src={primaryPhoto}
                          alt={person.name}
                          thumbnailClassName="w-full h-full object-cover"
                          allowFullscreen={false}
                        />
                      </div>

                      <div className="flex-1 min-w-0 space-y-1">
                        <div className="flex items-start justify-between gap-1">
                          <h3 className="text-sm font-bold text-slate-100 truncate">
                            {person.name}
                          </h3>
                          <span
                            className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                              STATUS_BADGES[person.status] || 'bg-slate-800 text-slate-400'
                            }`}
                          >
                            {person.status}
                          </span>
                        </div>

                        <div className="text-[11px] font-mono text-slate-400 flex items-center gap-2">
                          <span>ID: {person.person_id}</span>
                        </div>

                        {person.case_id && (
                          <div className="text-[10px] font-mono text-sky-400 flex items-center gap-1">
                            <Tag className="w-3 h-3 text-sky-500" />
                            <span>{person.case_id}</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Meta details */}
                    <div className="grid grid-cols-2 gap-2 text-[11px] bg-slate-950/70 p-2.5 rounded-lg border border-slate-800/80 font-mono">
                      {person.age !== null && (
                        <div>
                          <span className="text-slate-500">Age: </span>
                          <span className="text-slate-300">{person.age} yrs</span>
                        </div>
                      )}
                      {person.gender && (
                        <div>
                          <span className="text-slate-500">Gender: </span>
                          <span className="text-slate-300">{person.gender}</span>
                        </div>
                      )}
                      {person.last_known_location && (
                        <div className="col-span-2 flex items-center gap-1 text-slate-400 truncate">
                          <MapPin className="w-3 h-3 text-slate-500 shrink-0" />
                          <span className="truncate">{person.last_known_location}</span>
                        </div>
                      )}
                      {person.date_last_seen && (
                        <div className="col-span-2 flex items-center gap-1 text-slate-400">
                          <Calendar className="w-3 h-3 text-slate-500 shrink-0" />
                          <span>Last seen: {person.date_last_seen}</span>
                        </div>
                      )}
                    </div>

                    {/* Card Footer: Photo count & Actions */}
                    <div className="flex items-center justify-between pt-1 border-t border-slate-800/60 text-xs">
                      <span className="text-slate-500 text-[11px] font-mono">
                        {person.photo_paths?.length || 0} Reference Photos
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleOpenDetails(person)
                        }}
                        className="text-blue-400 hover:text-blue-300 text-[11px] font-semibold flex items-center gap-1"
                      >
                        <span>View Details</span>
                        <span>&rarr;</span>
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Right Drawer: Selected Case Details & Sightings */}
      {selectedPerson && (
        <div className="w-96 lg:w-[420px] bg-slate-900 border-l border-slate-800 flex flex-col h-full overflow-hidden shrink-0 shadow-2xl">
          {/* Drawer Header */}
          <div className="p-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-sky-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Case Investigation Record
              </h3>
            </div>
            <button
              type="button"
              onClick={() => setSelectedPerson(null)}
              className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Drawer Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs font-mono">
            {/* Identity Info Card */}
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2.5">
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="text-sm font-bold text-slate-100">{selectedPerson.name}</h4>
                  <p className="text-[11px] text-slate-400">ID: {selectedPerson.person_id}</p>
                  {selectedPerson.case_id && (
                    <p className="text-[11px] text-sky-400 mt-0.5">Case: {selectedPerson.case_id}</p>
                  )}
                </div>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase ${
                    STATUS_BADGES[selectedPerson.status] || 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {selectedPerson.status}
                </span>
              </div>

              {/* Status Change Triage (Admin/Police) */}
              {isAuthorized && (
                <div className="pt-2 border-t border-slate-800 flex items-center justify-between gap-2">
                  <span className="text-[11px] text-slate-400">Update Status:</span>
                  <div className="flex gap-1.5">
                    {(['ACTIVE', 'FOUND', 'CLOSED'] as PersonCaseStatus[]).map((st) => (
                      <button
                        key={st}
                        type="button"
                        disabled={isUpdatingStatus || selectedPerson.status === st}
                        onClick={() => handleUpdateCaseStatus(st)}
                        className={`text-[10px] font-bold px-2 py-0.5 rounded transition ${
                          selectedPerson.status === st
                            ? 'bg-blue-600 text-white cursor-default'
                            : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700'
                        }`}
                      >
                        {st}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Facial Recognition Readiness */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-emerald-950 border border-emerald-600/60 flex items-center justify-center text-emerald-400">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <div className="text-[11px]">
                <div className="font-bold text-emerald-300">ArcFace 512-d Template Active</div>
                <div className="text-slate-500 text-[10px]">
                  Cosine Similarity Cutoff: 0.89 (Frozen Threshold)
                </div>
              </div>
            </div>

            {/* Reference Photos Gallery */}
            <div className="space-y-2">
              <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
                <span>Reference Photographs</span>
                <span className="text-slate-500 font-normal">
                  {selectedPerson.photo_paths?.length || 0} Photos
                </span>
              </div>

              <div className="grid grid-cols-4 gap-2">
                {selectedPerson.photo_paths?.map((path: string, idx: number) => (
                  <div key={idx} className="aspect-square rounded-lg overflow-hidden border border-slate-800">
                    <MediaPreview
                      src={path}
                      alt={`${selectedPerson.name} ref photo ${idx + 1}`}
                      title={`${selectedPerson.name} — Reference Photo #${idx + 1}`}
                      thumbnailClassName="w-full h-full object-cover"
                      allowFullscreen={true}
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Description & Case Notes */}
            {selectedPerson.notes && (
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                <div className="text-[10px] uppercase text-slate-500 tracking-wider">
                  Physical Notes & Case Context
                </div>
                <p className="text-xs text-slate-300 font-sans leading-relaxed">
                  {selectedPerson.notes}
                </p>
              </div>
            )}

            {/* Sightings & Alerts Stream for this Person */}
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
                  CCTV Sighting Alerts ({selectedPerson.alerts?.length || 0})
                </span>
                {selectedPerson.verified_sightings ? (
                  <span className="text-[10px] text-emerald-400 font-bold">
                    {selectedPerson.verified_sightings} Verified
                  </span>
                ) : null}
              </div>

              {selectedPerson.alerts && selectedPerson.alerts.length > 0 ? (
                <div className="space-y-2">
                  {selectedPerson.alerts.map((alt: AlertRecord) => (
                    <div
                      key={alt.alert_id}
                      className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 text-slate-300">
                          <Video className="w-3.5 h-3.5 text-sky-400" />
                          <span className="font-bold">Camera {alt.camera_id}</span>
                        </div>
                        <span
                          className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase ${
                            alt.status === 'VERIFIED'
                              ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                              : alt.status === 'DISMISSED'
                              ? 'bg-slate-800 text-slate-400'
                              : 'bg-amber-950 text-amber-300 border border-amber-800'
                          }`}
                        >
                          {alt.status}
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-slate-400">Match Similarity:</span>
                        <span className="text-sky-400 font-bold">
                          {(alt.similarity * 100).toFixed(2)}%
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[10px] text-slate-500">
                        <span>Timestamp: {alt.timestamp}s</span>
                        <span>Track: {alt.track_id}</span>
                      </div>

                      {/* Action buttons for sightings triage */}
                      {isAuthorized && (
                        <div className="flex gap-2 pt-1 border-t border-slate-800/80">
                          <button
                            type="button"
                            onClick={() => handleVerifySighting(alt.id, 'VERIFIED')}
                            disabled={alt.status === 'VERIFIED'}
                            className="flex-1 py-1 rounded bg-emerald-950 hover:bg-emerald-900 border border-emerald-800 text-emerald-300 text-[10px] font-bold transition disabled:opacity-50 flex items-center justify-center gap-1"
                          >
                            <CheckCircle2 className="w-3 h-3" />
                            <span>Confirm</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => handleVerifySighting(alt.id, 'DISMISSED')}
                            disabled={alt.status === 'DISMISSED'}
                            className="flex-1 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-bold transition disabled:opacity-50 flex items-center justify-center gap-1"
                          >
                            <XCircle className="w-3 h-3" />
                            <span>Reject</span>
                          </button>
                          {onSelectAlertForOperation && (
                            <button
                              type="button"
                              onClick={() => onSelectAlertForOperation(alt)}
                              className="px-2 py-1 bg-blue-950 hover:bg-blue-900 border border-blue-800 text-blue-300 rounded text-[10px] font-bold transition flex items-center gap-1"
                              title="Inspect on live map & timeline"
                            >
                              <ExternalLink className="w-3 h-3" />
                              <span>Map</span>
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-slate-950/60 p-4 rounded-lg border border-slate-800 text-center text-slate-500 text-[11px]">
                  No CCTV recognition sightings recorded yet.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MissingPersonsView
