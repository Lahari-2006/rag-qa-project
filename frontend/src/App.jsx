import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

const API_BASE = (
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? 'http://localhost:8000' : '')
).replace(/\/+$/, '')
const ASK_URL = `${API_BASE}/ask`
const UPLOAD_URL = `${API_BASE}/upload`
const DOCUMENTS_URL = `${API_BASE}/documents`

function App() {
  const [authToken, setAuthToken] = useState(() => sessionStorage.getItem('rag-demo-auth') || '')
  const [authRequired, setAuthRequired] = useState(false)
  const [authError, setAuthError] = useState('')
  const [username, setUsername] = useState('demo')
  const [password, setPassword] = useState('')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [documents, setDocuments] = useState([])
  const [deletingDoc, setDeletingDoc] = useState(null)
  const bottomRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleUnauthorized = useCallback(() => {
    sessionStorage.removeItem('rag-demo-auth')
    setAuthToken('')
    setAuthRequired(true)
  }, [])

  const apiFetch = useCallback(async (url, options = {}) => {
    const headers = new Headers(options.headers)
    if (authToken) headers.set('Authorization', `Basic ${authToken}`)
    const res = await fetch(url, { ...options, headers })
    if (res.status === 401) handleUnauthorized()
    return res
  }, [authToken, handleUnauthorized])

  const refreshDocuments = useCallback(async () => {
    try {
      const res = await apiFetch(DOCUMENTS_URL)
      if (!res.ok) return
      const data = await res.json()
      setDocuments(data.documents || [])
    } catch {
      // silently ignore - document list is a convenience, not critical path
    }
  }, [apiFetch])

  useEffect(() => {
    refreshDocuments()
  }, [refreshDocuments])

  async function handleLogin(e) {
    e.preventDefault()
    setAuthError('')
    const encoded = new TextEncoder().encode(`${username}:${password}`)
    const token = btoa(Array.from(encoded, byte => String.fromCharCode(byte)).join(''))

    try {
      const res = await fetch(DOCUMENTS_URL, {
        headers: { Authorization: `Basic ${token}` }
      })
      if (!res.ok) {
        setAuthError(res.status === 401 ? 'Incorrect username or password.' : `Server responded ${res.status}`)
        return
      }
      sessionStorage.setItem('rag-demo-auth', token)
      setAuthToken(token)
      setDocuments((await res.json()).documents || [])
      setAuthRequired(false)
      setPassword('')
    } catch (err) {
      setAuthError(`Could not connect to the API: ${err.message}`)
    }
  }

  async function handleAsk(e) {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    setMessages(prev => [...prev, { role: 'user', text: question }])
    setInput('')
    setLoading(true)

    try {
      const res = await apiFetch(ASK_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || `Server responded ${res.status}`)
      }

      setMessages(prev => [
        ...prev,
        { role: 'assistant', text: data.answer, sources: data.sources }
      ])
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: 'error', text: err.message }
      ])
    } finally {
      setLoading(false)
    }
  }

  async function handleFileSelected(e) {
    const file = e.target.files[0]
    if (!file) return

    setUploadStatus({ text: `Indexing ${file.name}…`, kind: 'loading' })

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await apiFetch(UPLOAD_URL, { method: 'POST', body: formData })
      const data = await res.json()

      if (!res.ok) throw new Error(data.detail || `Server responded ${res.status}`)

      setUploadStatus({
        text: `${data.filename} — ${data.message}`,
        kind: data.chunks_added > 0 ? 'success' : 'neutral'
      })
      refreshDocuments()
    } catch (err) {
      setUploadStatus({ text: `Upload failed: ${err.message}`, kind: 'error' })
    } finally {
      e.target.value = ''
    }
  }

  async function handleDelete(filename) {
    setDeletingDoc(filename)
    try {
      const res = await apiFetch(`${DOCUMENTS_URL}/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.detail || `Server responded ${res.status}`)

      setUploadStatus({
        text: `Removed ${filename} (${data.chunks_removed} chunks)`,
        kind: 'neutral'
      })
      refreshDocuments()
    } catch (err) {
      setUploadStatus({ text: `Delete failed: ${err.message}`, kind: 'error' })
    } finally {
      setDeletingDoc(null)
    }
  }

  if (authRequired) {
    return (
      <main className="app login-page">
        <form className="login-panel" onSubmit={handleLogin}>
          <h1>Resume demo</h1>
          <p className="subtitle">Enter the demo credentials to continue.</p>
          <label>
            Username
            <input
              autoComplete="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </label>
          {authError && <p className="login-error">{authError}</p>}
          <button type="submit">Continue</button>
        </form>
      </main>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-top">
          <div>
            <h1>Document Q&amp;A</h1>
            <p className="subtitle">Ask questions grounded in your uploaded documents</p>
          </div>
          <div className="upload-area">
            <button
              type="button"
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              + Add PDF
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              onChange={handleFileSelected}
              hidden
            />
          </div>
        </div>

        {documents.length > 0 && (
          <div className="doc-list">
            {documents.map(doc => (
              <div key={doc} className="doc-chip">
                <span className="doc-chip-name">{doc}</span>
                <button
                  type="button"
                  className="doc-chip-delete"
                  onClick={() => handleDelete(doc)}
                  disabled={deletingDoc === doc}
                  title={`Remove ${doc}`}
                >
                  {deletingDoc === doc ? '…' : '×'}
                </button>
              </div>
            ))}
          </div>
        )}

        {uploadStatus && (
          <div className={`upload-status upload-status-${uploadStatus.kind}`}>
            {uploadStatus.text}
          </div>
        )}
      </header>

      <div className="thread">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Upload a PDF, then ask a question to begin. Answers are grounded in your documents — every response cites the source passages it drew from.</p>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`message message-${m.role}`}>
            <div className="message-label">
              {m.role === 'user' ? 'You' : m.role === 'assistant' ? 'Answer' : 'Error'}
            </div>
            <div className="message-text">{m.text}</div>
            {m.sources && m.sources.length > 0 && (
              <div className="sources">
                {m.sources.map((s, j) => (
                  <span key={j} className="source-tag">{s}</span>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message message-assistant">
            <div className="message-label">Answer</div>
            <div className="message-text loading-text">Retrieving and reasoning…</div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form className="input-bar" onSubmit={handleAsk}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask something about your documents…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Ask
        </button>
      </form>
    </div>
  )
}

export default App
