import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

/* ---- Chat Assistant ---- */
function ChatPanel({ predictionCtx }) {
    const [messages, setMessages] = useState([{
        role: 'assistant',
        text: "👋 Hi! I'm your AI Radiologist Assistant. Upload an X-ray to begin, or ask me anything about fracture detection."
    }]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const endRef = useRef(null);

    useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

    const send = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;
        const msg = input.trim();
        setInput('');
        setMessages(p => [...p, { role: 'user', text: msg }]);
        setLoading(true);
        try {
            const ctx = predictionCtx?.prediction ?? 'Unknown';
            const conf = predictionCtx?.confidence ?? 0;
            const res = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, prediction_context: ctx, confidence: conf })
            });
            if (res.ok) {
                const data = await res.json();
                setMessages(p => [...p, { role: 'assistant', text: data.response }]);
            } else throw new Error();
        } catch {
            setMessages(p => [...p, { role: 'assistant', text: `Running in demo mode — no OpenAI key configured. Based on the model, this scan shows: ${predictionCtx?.prediction ?? 'awaiting scan'}.` }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-panel">
            <div className="chat-header">
                <div className="chat-status-dot" />
                AI Radiologist Assistant
            </div>
            <div className="chat-messages">
                {messages.map((m, i) => (
                    <div key={i} className={`chat-bubble ${m.role}`}>{m.text}</div>
                ))}
                {loading && <div className="chat-typing">···</div>}
                <div ref={endRef} />
            </div>
            <form className="chat-input-bar" onSubmit={send}>
                <input
                    className="chat-input"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder="Ask about the scan..."
                />
                <button type="submit" className="btn btn-primary" style={{ padding: '0 1rem', flexShrink: 0 }} disabled={loading || !input.trim()}>
                    <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                </button>
            </form>
        </div>
    );
}

/* ---- Upload Zone ---- */
function UploadZone({ onFile }) {
    const [dragging, setDragging] = useState(false);
    const [camera, setCamera] = useState(false);
    const fileRef = useRef(null);
    const videoRef = useRef(null);
    const streamRef = useRef(null);

    const onDrop = (e) => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); };
    const startCam = async () => {
        try {
            const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            streamRef.current = s;
            setCamera(true);
            if (videoRef.current) videoRef.current.srcObject = s;
        } catch { alert('Camera access denied.'); }
    };
    const stopCam = () => { streamRef.current?.getTracks().forEach(t => t.stop()); setCamera(false); };
    const capture = () => {
        const c = document.createElement('canvas');
        c.width = videoRef.current.videoWidth; c.height = videoRef.current.videoHeight;
        c.getContext('2d').drawImage(videoRef.current, 0, 0);
        c.toBlob(blob => { stopCam(); onFile(new File([blob], 'camera.jpg', { type: 'image/jpeg' })); }, 'image/jpeg', 0.95);
    };

    if (camera) return (
        <div className={`upload-zone`} style={{ cursor: 'default' }}>
            <div className="camera-view">
                <video ref={videoRef} autoPlay playsInline className="camera-preview" />
                <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button className="btn btn-primary" onClick={capture} style={{ background: 'var(--success)' }}>📸 Capture</button>
                    <button className="btn btn-ghost" onClick={stopCam}>Cancel</button>
                </div>
            </div>
        </div>
    );

    return (
        <div
            className={`upload-zone ${dragging ? 'drag-active' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
        >
            <div className="upload-zone-inner">
                <div className="upload-icon-wrap">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                </div>
                <div className="upload-title">Drop your X-Ray here</div>
                <div className="upload-sub">Supports JPEG, PNG · Drag & drop or click to browse</div>
                <div className="upload-actions" onClick={e => e.stopPropagation()}>
                    <button className="btn btn-primary" onClick={() => fileRef.current?.click()}>
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                        Select File
                    </button>
                    <button className="btn btn-ghost" onClick={startCam}>
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                        Use Camera
                    </button>
                </div>
            </div>
            <input type="file" ref={fileRef} style={{ display: 'none' }} accept="image/*" onChange={e => e.target.files[0] && onFile(e.target.files[0])} />
        </div>
    );
}

/* ---- Results Dashboard ---- */
function ResultsDashboard({ result, file, onReset }) {
    if (!result) return null;
    const isFractured = result.prediction.toLowerCase().includes('fracture') && !result.prediction.toLowerCase().includes('not');
    const conf = (result.confidence * 100).toFixed(1);
    const verdictClass = isFractured ? 'fracture' : 'no-fracture';

    return (
        <div>
            {/* Header verdict */}
            <div className="results-header">
                <div className="result-verdict">
                    <div className={`verdict-icon ${verdictClass}`}>
                        {isFractured ? (
                            <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                        ) : (
                            <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        )}
                    </div>
                    <div>
                        <div className="verdict-label">Result</div>
                        <div className={`verdict-value ${verdictClass}`}>
                            {isFractured ? '🦴 Fracture Detected' : '✅ No Fracture Found'}
                        </div>
                    </div>
                </div>
                <div className="confidence-section">
                    <div className="conf-label">
                        <span>AI Confidence</span><span>{conf}%</span>
                    </div>
                    <div className="conf-track">
                        <div className="conf-fill high" style={{ width: `${conf}%` }} />
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                        {conf >= 90 ? 'Very high confidence — result is reliable' : conf >= 75 ? 'Good confidence — result is likely accurate' : 'Moderate confidence — consider clinical review'}
                    </div>
                </div>
                <button className="btn btn-ghost" onClick={onReset}>
                    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                    New Scan
                </button>
            </div>

            {/* Visualization grid */}
            <div className="viz-grid">
                {/* Original X-Ray — plain */}
                <div className="viz-card">
                    <div className="viz-card-header">
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                        Your X-Ray
                    </div>
                    <div className="viz-image-wrap" style={{ background: '#111' }}>
                        <XRayWithBbox file={file} bbox={null} />
                    </div>
                    <div className="viz-footer">The original image you uploaded</div>
                </div>

                {/* YOLO Localisation */}
                <div className="viz-card">
                    <div className="viz-card-header">
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path strokeLinecap="round" strokeLinejoin="round" d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>
                        YOLO Fracture Localisation
                        {result.bbox && (
                            <span className="viz-badge swin" style={{ background: 'rgba(220,38,38,0.15)', color: '#f87171', border: '1px solid rgba(220,38,38,0.3)' }}>YOLO</span>
                        )}
                    </div>
                    <div className="viz-image-wrap" style={{ background: '#111', position: 'relative' }}>
                        {result.bbox ? (
                            <XRayWithBbox file={file} bbox={result.bbox} />
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 280, color: 'var(--text-muted)', gap: '0.5rem', fontSize: '0.9rem' }}>
                                <svg width="36" height="36" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                <span>{isFractured ? 'YOLO localisation not available for this scan' : 'No fracture detected — no bounding box'}</span>
                            </div>
                        )}
                    </div>
                    <div className="viz-footer">
                        {result.bbox
                            ? <>🟥 <strong>Red box</strong> = fracture region located by YOLO · Confidence: {Math.round(result.bbox.conf * 100)}%</>
                            : <>{isFractured ? 'Fracture detected but YOLO could not localise the region' : 'No fracture found in this scan'}</>
                        }
                    </div>
                </div>

                {/* Mamba */}
                {result.mamba_base64 && (
                    <div className="viz-card" style={{ gridColumn: '1 / -1' }}>
                        <div className="viz-card-header">
                            <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
                            Bone Pattern Analysis
                            <span className="viz-badge mamba" title="Vision Mamba AI stream">Advanced</span>
                        </div>
                        <div className="viz-image-wrap" style={{ background: 'var(--bg-surface-2)' }}>
                            <img src={result.mamba_base64} alt="Bone pattern analysis" style={{ width: '100%', maxHeight: 240, objectFit: 'contain' }} />
                        </div>
                        <div className="viz-footer">📈 This graph shows how the AI scanned across your bone — <strong>spikes (peaks) indicate where it found abnormalities</strong></div>
                    </div>
                )}
            </div>

            {/* Disclaimer */}
            <div style={{
                marginTop: '1.5rem',
                padding: '0.85rem 1.2rem',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(245,158,11,0.08)',
                border: '1px solid rgba(245,158,11,0.2)',
                fontSize: '0.82rem',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.6rem'
            }}>
                <span style={{ fontSize: '1rem' }}>⚠️</span>
                <span>This AI tool is for <strong>screening assistance only</strong>. Always consult a qualified doctor or radiologist before making any medical decisions.</span>
            </div>
        </div>
    );
}



/* ---- X-Ray with optional YOLO bounding box overlay ---- */
function XRayWithBbox({ file, bbox }) {
    const imgRef = React.useRef(null);
    const canvasRef = React.useRef(null);
    const [imgUrl] = React.useState(() => URL.createObjectURL(file));

    const drawBox = React.useCallback(() => {
        const img = imgRef.current;
        const canvas = canvasRef.current;
        if (!img || !canvas || !bbox) return;

        const { naturalWidth: nw, naturalHeight: nh, offsetWidth: dw, offsetHeight: dh } = img;
        // Canvas should match the displayed image size
        canvas.width  = dw;
        canvas.height = dh;
        canvas.style.width  = dw + 'px';
        canvas.style.height = dh + 'px';

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, dw, dh);

        // Convert normalized coords to displayed pixel coords
        const x1 = bbox.x1 * dw;
        const y1 = bbox.y1 * dh;
        const x2 = bbox.x2 * dw;
        const y2 = bbox.y2 * dh;
        const bw = x2 - x1;
        const bh = y2 - y1;

        // Semi-transparent fill
        ctx.fillStyle = 'rgba(220, 38, 38, 0.12)';
        ctx.fillRect(x1, y1, bw, bh);

        // Bold red border
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, bw, bh);

        // Corner accents (like a targeting reticle)
        const cs = Math.min(bw, bh) * 0.18; // corner size
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#fbbf24'; // amber accent
        [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([cx, cy], i) => {
            const sx = i % 2 === 0 ? 1 : -1;
            const sy = i < 2 ? 1 : -1;
            ctx.beginPath();
            ctx.moveTo(cx + sx * cs, cy);
            ctx.lineTo(cx, cy);
            ctx.lineTo(cx, cy + sy * cs);
            ctx.stroke();
        });

        // Label
        const label = `Fracture ${Math.round(bbox.conf * 100)}%`;
        ctx.font = 'bold 13px Inter, sans-serif';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(x1, y1 - 24, tw + 12, 22);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x1 + 6, y1 - 7);
    }, [bbox]);

    return (
        <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
            <img
                ref={imgRef}
                src={imgUrl}
                alt="Input X-ray"
                style={{ maxHeight: 280, objectFit: 'contain', display: 'block' }}
                onLoad={drawBox}
            />
            {bbox && (
                <canvas
                    ref={canvasRef}
                    style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
                />
            )}
        </div>
    );
}

/* ---- Main Scanner App ---- */
export default function ScannerApp({ theme, toggleTheme }) {
    const navigate = useNavigate();
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [backendOnline, setBackendOnline] = useState(null); // null = checking

    // Check backend health on mount and every 10s
    useEffect(() => {
        const check = async () => {
            try {
                const res = await fetch('http://localhost:8000/health', { signal: AbortSignal.timeout(3000) });
                setBackendOnline(res.ok);
            } catch {
                setBackendOnline(false);
            }
        };
        check();
        const id = setInterval(check, 10000);
        return () => clearInterval(id);
    }, []);

    const handleFile = async (f) => {
        if (!backendOnline) {
            setError('Backend is offline. Please start the FastAPI server: cd web_ui/backend && uvicorn main:app --port 8000');
            return;
        }
        setFile(f); setLoading(true); setError(null); setResult(null);
        const fd = new FormData();
        fd.append('file', f);
        try {
            const res = await fetch('http://localhost:8000/predict', { method: 'POST', body: fd });
            if (!res.ok) throw new Error('Backend error — ensure FastAPI is running on port 8000.');
            const data = await res.json();
            if (data.rejected) {
                setError(data.reject_reason || 'The uploaded image does not appear to be a medical X-ray.');
                setFile(null);
            } else {
                setResult(data);
            }
        } catch (e) {
            setError(e.message);
            setFile(null);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="scanner-page">
            {/* Top navbar */}
            <nav className="navbar">
                <div className="nav-inner">
                    <button className="nav-logo" style={{ cursor: 'pointer', background: 'none', border: 'none' }} onClick={() => navigate('/')}>
                        ← &nbsp;
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M4.5 12.5l3 3 5-5 2 2 5-5" /><rect x="3" y="3" width="18" height="18" rx="3" />
                        </svg>
                        Fracture<span>Mamba</span>
                    </button>
                    <div className="nav-actions">
                        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
                            {theme === 'dark' ? (
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>
                            ) : (
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" /></svg>
                            )}
                        </button>
                    </div>
                </div>
            </nav>

            {/* Secondary header */}
            <div className="scanner-header">
                <div className="scanner-header-inner">
                    <div className="scanner-title">🔬 X-Ray Analysis Dashboard</div>
                    <span className="badge" style={{
                        background: backendOnline === false ? 'rgba(220,38,38,0.12)' : 'var(--accent-light)',
                        color: backendOnline === false ? 'var(--danger)' : 'var(--accent)',
                        borderColor: backendOnline === false ? 'rgba(220,38,38,0.3)' : undefined
                    }}>
                        <span style={{
                            width: 7, height: 7, borderRadius: '50%', display: 'inline-block',
                            background: backendOnline === null ? '#F59E0B' : backendOnline ? 'var(--success)' : 'var(--danger)'
                        }} />
                        {backendOnline === null ? 'Connecting…' : backendOnline ? 'Model Ready' : 'Backend Offline'}
                    </span>
                </div>
            </div>

            {/* Backend offline warning banner */}
            {backendOnline === false && (
                <div style={{ background: 'rgba(220,38,38,0.08)', borderBottom: '1px solid rgba(220,38,38,0.2)', padding: '0.75rem 2rem', fontSize: '0.88rem', color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                    <strong>Backend Offline</strong> — Start the FastAPI server:
                    <code style={{ background: 'rgba(0,0,0,0.1)', padding: '0.1rem 0.5rem', borderRadius: 4, fontFamily: 'monospace' }}>cd web_ui/backend &amp;&amp; uvicorn main:app --port 8000</code>
                </div>
            )}

            {/* Body */}
            <div className="scanner-body">
                {/* Main area */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    {error && (
                        <div style={{ padding: '1rem 1.5rem', borderRadius: 'var(--radius-md)', background: 'rgba(220,38,38,0.08)', border: '1px solid rgba(220,38,38,0.25)', color: 'var(--danger)', fontSize: '0.92rem' }}>
                            <strong>Something went wrong:</strong> {error}
                        </div>
                    )}
                    {!file && !loading && <UploadZone onFile={handleFile} />}
                    {loading && (
                        <div className="analyzing-card fade-up">
                            <div className="spinner" />
                            <div style={{ fontWeight: 700, fontSize: '1.2rem', marginBottom: '0.5rem' }}>Analyzing your X-Ray…</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>The AI is scanning the image for fractures — this usually takes a few seconds</div>
                        </div>
                    )}
                    {result && file && <ResultsDashboard result={result} file={file} onReset={() => { setFile(null); setResult(null); }} />}
                </div>

                {/* Sidebar chat */}
                <div>
                    <ChatPanel predictionCtx={result} />
                </div>
            </div>
        </div>
    );
}
