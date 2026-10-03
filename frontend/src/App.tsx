import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  clearStoredAuth,
  fetchAlerts,
  fetchCameras,
  fetchHealth,
  fetchTracks,
  fetchWorkerStatuses,
  getStoredToken,
  getStoredUserInfo,
  getWebSocketUrl,
  seedDemoMovement,
  updateAlertStatus,
} from './api/client'
import { AlertDetails } from './components/AlertDetails'
import { AlertFeed } from './components/AlertFeed'
import { Header } from './components/Header'
import { LoginScreen } from './components/LoginScreen'
import { MapView } from './components/MapView'
import { TrackTimeline } from './components/TrackTimeline'
import type {
  AlertRecord,
  AlertStatus,
  Camera,
  TrackRecord,
  WebSocketAlertMessage,
  WorkerMetrics,
} from './types'

export const App: React.FC = () => {
  const [currentUser, setCurrentUser] = useState<string | null>(() => getStoredUserInfo()?.username || null)
  const [currentRole, setCurrentRole] = useState<string | null>(() => getStoredUserInfo()?.role || null)
  const [backendConnected, setBackendConnected] = useState<boolean>(false)
  const [wsConnected, setWsConnected] = useState<boolean>(false)
  const [alerts, setAlerts] = useState<AlertRecord[]>([])
  const [selectedAlert, setSelectedAlert] = useState<AlertRecord | null>(null)
  const [tracks, setTracks] = useState<TrackRecord[]>([])
  const [cameras, setCameras] = useState<Camera[]>([])
  const [workers, setWorkers] = useState<WorkerMetrics[]>([])
  const [isLoadingAlerts, setIsLoadingAlerts] = useState<boolean>(true)
  const [isUpdatingStatus, setIsUpdatingStatus] = useState<boolean>(false)
  const [isSeedingDemo, setIsSeedingDemo] = useState<boolean>(false)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)

  // 1. Initial Data Fetch & Health Checks (only when authenticated)
  const loadInitialData = useCallback(async () => {
    try {
      const health = await fetchHealth()
      setBackendConnected(health.status === 'ok')

      const [cams, alts, trks, wrks] = await Promise.all([
        fetchCameras(false).catch(() => []),
        fetchAlerts().catch(() => []),
        fetchTracks().catch(() => []),
        fetchWorkerStatuses().catch(() => []),
      ])

      setCameras(cams)
      setAlerts(alts)
      setTracks(trks)
      setWorkers(wrks)

      // Default select the latest alert if available
      if (alts.length > 0 && !selectedAlert) {
        setSelectedAlert(alts[0])
      }
    } catch (e) {
      console.warn('Backend unavailable:', e)
      setBackendConnected(false)
    } finally {
      setIsLoadingAlerts(false)
    }
  }, [selectedAlert])

  // 2. Telemetry Polling (Every 3 seconds when authenticated)
  useEffect(() => {
    if (!currentUser || !getStoredToken()) return

    loadInitialData()
    const interval = setInterval(async () => {
      try {
        const wrks = await fetchWorkerStatuses()
        setWorkers(wrks)
        const trks = await fetchTracks()
        setTracks(trks)
      } catch {
        // Handled silently
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [currentUser, loadInitialData])

  // 3. WebSocket Real-Time Connection (only when authenticated)
  useEffect(() => {
    if (!currentUser || !getStoredToken()) return

    let isMounted = true

    const connectWs = () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }

      const prevSocket = wsRef.current
      if (prevSocket) {
        wsRef.current = null
        prevSocket.onopen = null
        prevSocket.onmessage = null
        prevSocket.onclose = null
        prevSocket.onerror = null
        try {
          prevSocket.close()
        } catch {}
      }

      const wsUrl = getWebSocketUrl()
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        if (!isMounted || wsRef.current !== ws) return
        setWsConnected(true)
      }

      ws.onmessage = (event) => {
        if (!isMounted || wsRef.current !== ws) return
        try {
          const msg: WebSocketAlertMessage = JSON.parse(event.data)
          if (msg.event === 'alert.created' && msg.alert) {
            const newAlert = msg.alert
            setAlerts((prev) => {
              const exists = prev.some((a) => a.id === newAlert.id || a.alert_id === newAlert.alert_id)
              if (exists) {
                return prev.map((a) =>
                  a.id === newAlert.id || a.alert_id === newAlert.alert_id ? newAlert : a
                )
              }
              return [newAlert, ...prev]
            })

            setSelectedAlert(newAlert)
          } else if (msg.event === 'alert.updated' && msg.alert) {
            const updatedAlert = msg.alert
            setAlerts((prev) =>
              prev.map((a) =>
                a.id === updatedAlert.id || a.alert_id === updatedAlert.alert_id ? updatedAlert : a
              )
            )
            setSelectedAlert((current) =>
              current && (current.id === updatedAlert.id || current.alert_id === updatedAlert.alert_id)
                ? updatedAlert
                : current
            )
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onclose = () => {
        if (!isMounted || wsRef.current !== ws) return
        setWsConnected(false)
        reconnectTimeoutRef.current = window.setTimeout(connectWs, 3000)
      }

      ws.onerror = () => {
        if (!isMounted || wsRef.current !== ws) return
        setWsConnected(false)
        try {
          ws.close()
        } catch {}
      }
    }

    connectWs()

    return () => {
      isMounted = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
      const s = wsRef.current
      if (s) {
        wsRef.current = null
        s.onopen = null
        s.onmessage = null
        s.onclose = null
        s.onerror = null
        try {
          s.close()
        } catch {}
      }
    }
  }, [currentUser])

  // 4. Update Alert Status (Confirm / Reject)
  const handleUpdateStatus = async (alertId: string | number, newStatus: AlertStatus) => {
    setIsUpdatingStatus(true)
    try {
      const updated = await updateAlertStatus(alertId, newStatus)
      setAlerts((prev) => prev.map((a) => (a.id === updated.id ? updated : a)))
      if (selectedAlert?.id === updated.id) {
        setSelectedAlert(updated)
      }
    } catch (err) {
      console.error('Failed to update alert status:', err)
      alert(`Error updating status: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsUpdatingStatus(false)
    }
  }

  // 5. Trigger Demo Scenario
  const handleSeedDemo = async () => {
    setIsSeedingDemo(true)
    try {
      const createdAlerts = await seedDemoMovement()
      await loadInitialData()
      if (createdAlerts.length > 0) {
        setSelectedAlert(createdAlerts[0])
      }
    } catch (err) {
      console.error('Failed to seed demo scenario:', err)
    } finally {
      setIsSeedingDemo(false)
    }
  }

  const handleLogout = () => {
    clearStoredAuth()
    setCurrentUser(null)
    setCurrentRole(null)
    setWsConnected(false)
    if (wsRef.current) {
      try {
        wsRef.current.close()
      } catch {}
      wsRef.current = null
    }
  }

  // If unauthenticated, show Login Screen
  if (!currentUser || !getStoredToken()) {
    return (
      <LoginScreen
        onAuthenticated={(username, role) => {
          setCurrentUser(username)
          setCurrentRole(role)
        }}
      />
    )
  }

  // Derive relevant tracks for selected person
  const selectedPersonId = selectedAlert?.person_id
  const personTracks = selectedPersonId
    ? tracks.filter((t) => t.person_id === selectedPersonId)
    : selectedAlert
    ? tracks.filter((t) => t.track_id === selectedAlert.track_id)
    : []

  // Derive movement path across cameras for selected person
  const movementPath = personTracks
    .sort((a, b) => a.first_seen - b.first_seen)
    .map((t) => t.camera_id)

  // Selected track record
  const selectedTrack = selectedAlert
    ? tracks.find(
        (t) => t.camera_id === selectedAlert.camera_id && t.track_id === selectedAlert.track_id
      ) || null
    : null

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Header */}
      <Header
        backendConnected={backendConnected}
        wsConnected={wsConnected}
        workers={workers}
        onRefresh={loadInitialData}
        onSeedDemo={handleSeedDemo}
        isSeeding={isSeedingDemo}
        currentUsername={currentUser}
        currentRole={currentRole}
        onLogout={handleLogout}
      />

      {/* Main Operational Body */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Left: Real-time Alert Feed */}
        <AlertFeed
          alerts={alerts}
          selectedAlert={selectedAlert}
          onSelectAlert={(a) => setSelectedAlert(a)}
          isLoading={isLoadingAlerts}
        />

        {/* Center: Leaflet Interactive Map & Timeline */}
        <div className="flex-1 flex flex-col h-full overflow-hidden">
          <MapView
            cameras={cameras}
            workers={workers}
            selectedAlert={selectedAlert}
            movementPath={movementPath}
            onSelectCamera={(cid) => {
              const matchingAlert = alerts.find((a) => a.camera_id === cid)
              if (matchingAlert) setSelectedAlert(matchingAlert)
            }}
          />

          {/* Bottom: Cross-Camera Movement Timeline */}
          <TrackTimeline
            personName={selectedAlert?.person_name || 'No selection'}
            personId={selectedAlert?.person_id || null}
            tracks={personTracks}
            alerts={alerts}
            selectedTrackId={selectedAlert?.track_id || null}
            onSelectTrackNode={(trk, alt) => {
              if (alt) {
                setSelectedAlert(alt)
              } else {
                // Find or construct temporary focal alert for this camera
                const fallbackAlt = alerts.find((a) => a.camera_id === trk.camera_id)
                if (fallbackAlt) setSelectedAlert(fallbackAlt)
              }
            }}
          />
        </div>

        {/* Right: Alert Details & Human Verification Panel */}
        <AlertDetails
          alert={selectedAlert}
          track={selectedTrack}
          onUpdateStatus={handleUpdateStatus}
          isUpdating={isUpdatingStatus}
        />
      </div>
    </div>
  )
}

export default App

