/**
 * Login screen for FindU operator portal.
 * Minimal form — username + password → JWT token.
 */

import React, { useState } from 'react'
import { login, setStoredToken, setStoredUserInfo } from '../api/client'

interface LoginScreenProps {
  onAuthenticated: (username: string, role: string) => void
}

export const LoginScreen: React.FC<LoginScreenProps> = ({ onAuthenticated }) => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) return

    setIsLoading(true)
    setError(null)

    try {
      const result = await login(username.trim(), password)
      setStoredToken(result.access_token)
      setStoredUserInfo(result.username, result.role)
      onAuthenticated(result.username, result.role)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-screen bg-slate-950 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Branding */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-100 tracking-tight">FindU</h1>
          <p className="text-slate-400 text-sm mt-1">Missing Person Recognition System</p>
          <p className="text-slate-500 text-xs mt-1">Operator Portal — Authorised Access Only</p>
        </div>

        {/* Login Card */}
        <form
          onSubmit={handleSubmit}
          className="bg-slate-900 border border-slate-700 rounded-lg p-6 space-y-4"
        >
          <div>
            <label htmlFor="username" className="block text-xs font-medium text-slate-400 mb-1 uppercase tracking-wide">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={isLoading}
              placeholder="officer01"
              className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-100 text-sm placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-xs font-medium text-slate-400 mb-1 uppercase tracking-wide">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isLoading}
              placeholder="••••••••"
              className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-100 text-sm placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
            />
          </div>

          {error && (
            <div className="bg-red-900/40 border border-red-700 rounded px-3 py-2 text-red-300 text-xs">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading || !username.trim() || !password}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded px-4 py-2 text-sm font-medium transition-colors"
          >
            {isLoading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <p className="text-center text-slate-600 text-xs mt-4">
          Unauthorised access is prohibited and subject to prosecution.
        </p>
      </div>
    </div>
  )
}

export default LoginScreen
