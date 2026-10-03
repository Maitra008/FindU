/**
 * Reusable Media Preview & Lightbox Component for FindU.
 * Supports reference photos, CCTV snapshots, and alert evidence crops.
 */

import React, { useState } from 'react'
import { Camera, Download, Eye, Image as ImageIcon, Maximize2, X } from 'lucide-react'
import { getMediaUrl } from '../api/client'

interface MediaPreviewProps {
  src: string | null | undefined
  alt?: string
  title?: string
  className?: string
  thumbnailClassName?: string
  sourceLabel?: string
  timestamp?: string | number
  showMetadata?: boolean
  allowFullscreen?: boolean
}

export const MediaPreview: React.FC<MediaPreviewProps> = ({
  src,
  alt = 'Media preview',
  title,
  className = '',
  thumbnailClassName = 'w-full h-full object-cover',
  sourceLabel,
  timestamp,
  showMetadata = false,
  allowFullscreen = true,
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [hasError, setHasError] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  const mediaUrl = getMediaUrl(src)

  if (!src || hasError) {
    return (
      <div
        className={`bg-slate-950/80 border border-slate-800 rounded flex flex-col items-center justify-center text-slate-600 p-2 text-center select-none ${className}`}
      >
        <ImageIcon className="w-5 h-5 mb-1 opacity-50" />
        <span className="text-[10px] font-mono opacity-70">No Image</span>
      </div>
    )
  }

  return (
    <>
      {/* Thumbnail Container */}
      <div
        className={`relative group rounded overflow-hidden border border-slate-800 bg-slate-950 cursor-pointer ${className}`}
        onClick={() => allowFullscreen && setIsOpen(true)}
      >
        {isLoading && (
          <div className="absolute inset-0 bg-slate-900 animate-pulse flex items-center justify-center">
            <ImageIcon className="w-4 h-4 text-slate-700 animate-spin" />
          </div>
        )}

        <img
          src={mediaUrl}
          alt={alt}
          onLoad={() => setIsLoading(false)}
          onError={() => {
            setIsLoading(false)
            setHasError(true)
          }}
          className={`${thumbnailClassName} transition-transform duration-300 group-hover:scale-105`}
        />

        {/* Hover Overlay */}
        {allowFullscreen && (
          <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-1.5 text-slate-200">
            <Eye className="w-4 h-4 text-sky-400" />
            <span className="text-[11px] font-mono font-medium">Zoom</span>
          </div>
        )}

        {/* Metadata Badges */}
        {showMetadata && (sourceLabel || timestamp) && (
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-slate-950/90 via-slate-950/60 to-transparent p-1.5 flex items-center justify-between text-[10px] font-mono text-slate-300">
            {sourceLabel && (
              <span className="flex items-center gap-1">
                <Camera className="w-3 h-3 text-slate-400" />
                <span>{sourceLabel}</span>
              </span>
            )}
            {timestamp !== undefined && <span>{timestamp}</span>}
          </div>
        )}
      </div>

      {/* Fullscreen Lightbox Modal */}
      {isOpen && (
        <div
          className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-200"
          onClick={() => setIsOpen(false)}
        >
          <div
            className="relative max-w-4xl max-h-[90vh] bg-slate-900 border border-slate-700 rounded-xl overflow-hidden shadow-2xl flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Lightbox Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-950">
              <div className="flex items-center gap-2">
                <Maximize2 className="w-4 h-4 text-sky-400" />
                <span className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
                  {title || alt || 'Media Preview'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={mediaUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  download
                  className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition"
                  title="Open Original / Download"
                >
                  <Download className="w-4 h-4" />
                </a>
                <button
                  type="button"
                  onClick={() => setIsOpen(false)}
                  className="p-1 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded transition"
                  title="Close"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Lightbox Image */}
            <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-slate-950/50 min-h-[300px]">
              <img
                src={mediaUrl}
                alt={alt}
                className="max-h-[75vh] max-w-full object-contain rounded border border-slate-800 shadow-lg"
              />
            </div>

            {/* Lightbox Footer with Metadata */}
            {(sourceLabel || timestamp || src) && (
              <div className="px-4 py-2 bg-slate-950 border-t border-slate-800 text-xs font-mono text-slate-400 flex items-center justify-between">
                <span>Source: {sourceLabel || src}</span>
                {timestamp !== undefined && <span>Timestamp: {timestamp}</span>}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}

export default MediaPreview

