import React, { useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet'
import L from 'leaflet'
import type { AlertRecord, Camera, WorkerMetrics } from '../types'
import { CAMERA_COORDINATES, DEFAULT_ZOOM, MAP_CENTER } from '../config/mapConfig'

// Helper component to smoothly re-center map on selected camera/alert
const MapController: React.FC<{ targetCoords: [number, number] | null }> = ({ targetCoords }) => {
  const map = useMap()
  useEffect(() => {
    if (targetCoords) {
      map.flyTo(targetCoords, map.getZoom(), { duration: 1.0 })
    }
  }, [targetCoords, map])
  return null
}

interface MapViewProps {
  cameras?: Camera[]
  workers: WorkerMetrics[]
  selectedAlert: AlertRecord | null
  movementPath: string[] // List of camera_ids in chronological order e.g. ['C2', 'C3', 'C1']
  onSelectCamera: (cameraId: string) => void
}

export const MapView: React.FC<MapViewProps> = ({
  workers,
  selectedAlert,
  movementPath,
  onSelectCamera,
}) => {
  // Determine target coordinates for auto-pan
  const selectedCameraGeo = selectedAlert ? CAMERA_COORDINATES[selectedAlert.camera_id] : null
  const targetCoords: [number, number] | null = selectedCameraGeo
    ? [selectedCameraGeo.lat, selectedCameraGeo.lng]
    : null

  // Build movement polyline points
  const polylineCoords: [number, number][] = movementPath
    .map((cid) => CAMERA_COORDINATES[cid])
    .filter(Boolean)
    .map((geo) => [geo.lat, geo.lng])

  // Custom marker icon generator
  const createCameraIcon = (cid: string, isOnline: boolean, isSelected: boolean) => {
    let bgClass = isOnline ? 'bg-emerald-500' : 'bg-slate-600'
    let ringClass = isSelected
      ? 'ring-4 ring-sky-400 ring-offset-2 ring-offset-slate-900 scale-125 z-50'
      : 'ring-1 ring-slate-900'

    return L.divIcon({
      className: 'custom-map-icon',
      html: `
        <div class="flex items-center justify-center w-8 h-8 rounded-full shadow-lg ${bgClass} ${ringClass} text-slate-950 font-mono font-bold text-xs transition duration-300">
          ${cid}
        </div>
      `,
      iconSize: [32, 32],
      iconAnchor: [16, 16],
    })
  }

  return (
    <div className="relative w-full h-full flex-1 bg-slate-950">
      <MapContainer
        center={MAP_CENTER}
        zoom={DEFAULT_ZOOM}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        <MapController targetCoords={targetCoords} />

        {/* Cross-Camera Movement Polyline Path */}
        {polylineCoords.length > 1 && (
          <Polyline
            positions={polylineCoords}
            pathOptions={{
              color: '#38bdf8',
              weight: 4,
              dashArray: '8, 8',
              opacity: 0.85,
            }}
          />
        )}

        {/* Camera Markers */}
        {Object.entries(CAMERA_COORDINATES).map(([cid, geo]) => {
          const worker = workers.find((w) => w.camera_id === cid)
          const isOnline = worker ? worker.status === 'RUNNING' : false
          const isSelected = selectedAlert?.camera_id === cid
          const icon = createCameraIcon(cid, isOnline, isSelected)

          return (
            <Marker
              key={cid}
              position={[geo.lat, geo.lng]}
              icon={icon}
              eventHandlers={{
                click: () => onSelectCamera(cid),
              }}
            >
              <Popup>
                <div className="p-1 text-xs">
                  <div className="font-bold text-sm text-sky-400 mb-0.5">
                    {cid}: {geo.name}
                  </div>
                  <div className="text-slate-300 mb-1">{geo.zone}</div>
                  <div className="flex items-center space-x-2 font-mono text-[11px]">
                    <span className="text-slate-400">Status:</span>
                    <span className={isOnline ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                      {isOnline ? 'ONLINE' : 'OFFLINE'}
                    </span>
                  </div>
                  {worker && isOnline && (
                    <div className="text-slate-400 font-mono text-[11px] mt-0.5">
                      FPS: {worker.fps} | Faces: {worker.faces_detected}
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          )
        })}
      </MapContainer>

      {/* Map Overlay Legend */}
      <div className="absolute top-3 right-3 z-[1000] bg-slate-900/90 border border-slate-800 p-2.5 rounded shadow-lg text-xs font-mono text-slate-300 select-none backdrop-blur-sm">
        <div className="font-bold text-slate-200 mb-1.5 uppercase text-[10px] tracking-wider">
          Facility CCTV Map
        </div>
        <div className="flex items-center space-x-2 mb-1">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
          <span>Active Camera</span>
        </div>
        <div className="flex items-center space-x-2 mb-1">
          <span className="w-2.5 h-2.5 rounded-full bg-slate-600" />
          <span>Offline Camera</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-0.5 bg-sky-400" />
          <span>Cross-Camera Track</span>
        </div>
      </div>
    </div>
  )
}
