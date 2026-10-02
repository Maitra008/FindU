/**
 * Spatial mapping configuration for CCTV cameras across the facility campus.
 */

export interface CameraGeo {
  lat: number
  lng: number
  name: string
  zone: string
}

export const CAMERA_COORDINATES: Record<string, CameraGeo> = {
  C1: {
    lat: 37.7749,
    lng: -122.4194,
    name: 'North Entrance',
    zone: 'Building A - Gate 1',
  },
  C2: {
    lat: 37.7735,
    lng: -122.4168,
    name: 'Parking Lot East',
    zone: 'East Wing Parking',
  },
  C3: {
    lat: 37.7742,
    lng: -122.4182,
    name: 'Main Lobby',
    zone: 'Central Reception',
  },
  C4: {
    lat: 37.7731,
    lng: -122.4190,
    name: 'South Corridor',
    zone: 'Ground Floor South',
  },
}

export const MAP_CENTER: [number, number] = [37.7740, -122.4182]
export const DEFAULT_ZOOM = 17
