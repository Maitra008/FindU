/**
 * Modal dialog for registering a new missing person.
 * Collects case metadata and 1–10 reference photographs, generates ArcFace embeddings on backend.
 */

import React, { useRef, useState } from 'react'
import {
  AlertCircle,
  Calendar,
  Camera,
  CheckCircle2,
  FileText,
  MapPin,
  Upload,
  User,
  UserPlus,
  X,
} from 'lucide-react'
import { registerMissingPerson } from '../api/client'
import type { PersonRecord } from '../types'

interface RegisterPersonModalProps {
  isOpen: boolean
  onClose: () => void
  onRegistered: (person: PersonRecord) => void
}

interface PhotoPreview {
  file: File
  previewUrl: string
  id: string
}

export const RegisterPersonModal: React.FC<RegisterPersonModalProps> = ({
  isOpen,
  onClose,
  onRegistered,
}) => {
  const [name, setName] = useState('')
  const [caseId, setCaseId] = useState('')
  const [age, setAge] = useState<string>('')
  const [gender, setGender] = useState('Male')
  const [dateLastSeen, setDateLastSeen] = useState('')
  const [lastKnownLocation, setLastKnownLocation] = useState('')
  const [notes, setNotes] = useState('')
  const [photos, setPhotos] = useState<PhotoPreview[]>([])

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successPerson, setSuccessPerson] = useState<PersonRecord | null>(null)

  const fileInputRef = useRef<HTMLInputElement | null>(null)

  if (!isOpen) return null

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return
    const newFiles = Array.from(e.target.files)
    addFiles(newFiles)
  }

  const addFiles = (files: File[]) => {
    const validFiles = files.filter((f) => f.type.startsWith('image/'))
    if (validFiles.length === 0) {
      setError('Please select valid image files (.jpg, .png, .webp).')
      return
    }

    const previews: PhotoPreview[] = validFiles.map((file) => ({
      file,
      previewUrl: URL.createObjectURL(file),
      id: `${file.name}-${file.size}-${Math.random()}`,
    }))

    setPhotos((prev) => [...prev, ...previews].slice(0, 10))
    setError(null)
  }

  const handleRemovePhoto = (id: string) => {
    setPhotos((prev) => {
      const filtered = prev.filter((p) => p.id !== id)
      const removed = prev.find((p) => p.id === id)
      if (removed) URL.revokeObjectURL(removed.previewUrl)
      return filtered
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) {
      setError('Full name is required.')
      return
    }

    if (photos.length === 0) {
      setError('At least 1 reference photo is required for facial recognition matching (3-10 recommended).')
      return
    }

    setIsSubmitting(true)
    setError(null)

    const formData = new FormData()
    formData.append('name', name.trim())
    if (caseId.trim()) formData.append('case_id', caseId.trim())
    if (age.trim()) formData.append('age', age.trim())
    if (gender) formData.append('gender', gender)
    if (dateLastSeen.trim()) formData.append('date_last_seen', dateLastSeen.trim())
    if (lastKnownLocation.trim()) formData.append('last_known_location', lastKnownLocation.trim())
    if (notes.trim()) formData.append('notes', notes.trim())

    photos.forEach((p) => {
      formData.append('photos', p.file)
    })

    try {
      const created = await registerMissingPerson(formData)
      setSuccessPerson(created)
      onRegistered(created)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleResetAndClose = () => {
    photos.forEach((p) => URL.revokeObjectURL(p.previewUrl))
    setPhotos([])
    setName('')
    setCaseId('')
    setAge('')
    setDateLastSeen('')
    setLastKnownLocation('')
    setNotes('')
    setError(null)
    setSuccessPerson(null)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
      <div
        className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-2xl w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
              <UserPlus className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
                Register Missing Person
              </h2>
              <p className="text-slate-400 text-xs">
                Extract ArcFace embeddings and add identity to FAISS recognition index
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleResetAndClose}
            className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        {successPerson ? (
          <div className="p-8 text-center space-y-4">
            <div className="w-14 h-14 bg-emerald-950 border border-emerald-600 rounded-full flex items-center justify-center mx-auto text-emerald-400 shadow-lg shadow-emerald-950">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <div className="space-y-1">
              <h3 className="text-base font-bold text-slate-100">
                Identity Successfully Registered!
              </h3>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                Facial embeddings generated via SCRFD + ArcFace and synced across all live camera
                workers.
              </p>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 text-xs font-mono max-w-md mx-auto text-left space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-500">Name:</span>
                <span className="text-slate-200 font-bold">{successPerson.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Person ID:</span>
                <span className="text-sky-400">{successPerson.person_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Case ID:</span>
                <span className="text-slate-300">{successPerson.case_id || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Index Status:</span>
                <span className="text-emerald-400 font-semibold">ACTIVE & SEARCHABLE</span>
              </div>
            </div>

            <button
              type="button"
              onClick={handleResetAndClose}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-semibold transition"
            >
              Done & View Cases
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="p-6 space-y-5">
            {error && (
              <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-red-300 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-red-400 mt-0.5" />
                <div>
                  <span className="font-bold">Error: </span>
                  <span>{error}</span>
                </div>
              </div>
            )}

            {/* Identity & Case Fields */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div>
                <label className="block text-slate-400 font-medium mb-1 uppercase tracking-wider text-[11px]">
                  Full Name <span className="text-red-400">*</span>
                </label>
                <div className="relative">
                  <User className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-500" />
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Sarah Connor"
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1 uppercase tracking-wider text-[11px]">
                  Case Reference ID
                </label>
                <div className="relative">
                  <FileText className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-500" />
                  <input
                    type="text"
                    value={caseId}
                    onChange={(e) => setCaseId(e.target.value)}
                    placeholder="Auto-generated if blank (e.g. CASE-2026-004)"
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 font-mono text-xs"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1 uppercase tracking-wider text-[11px]">
                  Age
                </label>
                <input
                  type="number"
                  min="0"
                  max="120"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  placeholder="e.g. 29"
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1 uppercase tracking-wider text-[11px]">
                  Gender
                </label>
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-blue-500"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other / Non-binary</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1 uppercase tracking-wider text-[11px]">
                  Date Last Seen
                </label>
                <div className="relative">
                  <Calendar className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-500" />
                  <input
                    type="date"
                    value={dateLastSeen}
                    onChange={(e) => setDateLastSeen(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-slate-100 focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1 uppercase tracking-wider text-[11px]">
                  Last Known Location
                </label>
                <div className="relative">
                  <MapPin className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-500" />
                  <input
                    type="text"
                    value={lastKnownLocation}
                    onChange={(e) => setLastKnownLocation(e.target.value)}
                    placeholder="e.g. East Wing Concourse"
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="block text-slate-400 font-medium mb-1 uppercase tracking-wider text-[11px]">
                Case Notes & Physical Description
              </label>
              <textarea
                rows={2}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Distinctive clothing, scars, medical conditions, or circumstances..."
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 text-xs"
              />
            </div>

            {/* Reference Photos Upload Section */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="block text-slate-400 font-medium uppercase tracking-wider text-[11px]">
                  Reference Photographs <span className="text-red-400">*</span> (3–10 Recommended)
                </label>
                <span className="text-slate-500 text-[11px] font-mono">
                  {photos.length}/10 selected
                </span>
              </div>

              {/* Upload Dropzone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-700 hover:border-blue-500/80 bg-slate-950/60 rounded-xl p-4 text-center cursor-pointer transition group"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="flex flex-col items-center gap-1.5 text-slate-400 group-hover:text-slate-200 transition">
                  <div className="w-9 h-9 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center text-blue-400">
                    <Upload className="w-4 h-4" />
                  </div>
                  <p className="text-xs font-medium">
                    Click or drag & drop reference photos of the subject
                  </p>
                  <p className="text-[11px] text-slate-500 font-mono">
                    Clear frontal portraits provide highest match precision (ArcFace 512-d)
                  </p>
                </div>
              </div>

              {/* Photos Preview Grid */}
              {photos.length > 0 && (
                <div className="grid grid-cols-5 sm:grid-cols-6 gap-2 pt-2">
                  {photos.map((p, idx) => (
                    <div
                      key={p.id}
                      className="relative group rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square"
                    >
                      <img
                        src={p.previewUrl}
                        alt={`Photo ${idx + 1}`}
                        className="w-full h-full object-cover"
                      />
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleRemovePhoto(p.id)
                        }}
                        className="absolute top-1 right-1 bg-slate-950/80 hover:bg-red-600 text-slate-300 hover:text-white rounded-full p-0.5 transition shadow"
                        title="Remove photo"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                      <div className="absolute bottom-0 inset-x-0 bg-slate-950/80 text-[9px] font-mono text-slate-300 text-center py-0.5">
                        #{idx + 1}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Modal Actions */}
            <div className="pt-2 flex items-center justify-end gap-3 border-t border-slate-800">
              <button
                type="button"
                onClick={handleResetAndClose}
                disabled={isSubmitting}
                className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting || !name.trim() || photos.length === 0}
                className="px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-xs font-semibold transition flex items-center gap-1.5 shadow-lg shadow-blue-950/60"
              >
                {isSubmitting ? (
                  <>
                    <Camera className="w-3.5 h-3.5 animate-spin" />
                    <span>Extracting Embeddings…</span>
                  </>
                ) : (
                  <>
                    <UserPlus className="w-3.5 h-3.5" />
                    <span>Register Missing Person</span>
                  </>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}

export default RegisterPersonModal

