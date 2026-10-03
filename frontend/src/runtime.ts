/**
 * Development runtime diagnostic and build tracking.
 * Exposes non-sensitive build and connection metadata for live verification.
 */

export const FRONTEND_BUILD_ID = 'build-0df9745'
export const FRONTEND_MODE = import.meta.env.MODE || 'development'
export const FRONTEND_API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export interface RuntimeDiagnostic {
  buildId: string
  mode: string
  apiBase: string
  origin: string
  timestamp: string
}

export function getRuntimeDiagnostic(): RuntimeDiagnostic {
  return {
    buildId: FRONTEND_BUILD_ID,
    mode: FRONTEND_MODE,
    apiBase: FRONTEND_API_BASE,
    origin: typeof window !== 'undefined' ? window.location.origin : '',
    timestamp: new Date().toISOString(),
  }
}

// Log diagnostic banner on boot for immediate verification in DevTools console
if (typeof window !== 'undefined') {
  console.log(
    `%c[FindU Runtime] Build: ${FRONTEND_BUILD_ID} | Mode: ${FRONTEND_MODE} | API: ${FRONTEND_API_BASE}`,
    'color: #38bdf8; font-weight: bold; background: #0f172a; padding: 4px 8px; border-radius: 4px;'
  )
  ;(window as any).__FINDU_RUNTIME__ = getRuntimeDiagnostic()
}

