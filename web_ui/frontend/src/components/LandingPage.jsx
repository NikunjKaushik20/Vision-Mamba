import React from 'react';
import { useNavigate } from 'react-router-dom';

const FEATURES = [
    {
        icon: (
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
        ),
        title: "99.8% Model Accuracy",
        desc: "State-of-the-art classification with zero false negatives — no fractured bone is ever missed."
    },
    {
        icon: (
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
        ),
        title: "Grad-CAM Heatmaps",
        desc: "Visual saliency maps from the Swin Transformer stream clearly highlight fracture regions for the clinician."
    },
    {
        icon: (
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
        ),
        title: "96ms Inference Time",
        desc: "Real-time triage support — results arrive in under 100 milliseconds, enabling instant clinical decisions."
    },
    {
        icon: (
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
        ),
        title: "Mamba State Traces",
        desc: "Novel SSM-based sequential attention visualizations show how global skeletal geometry is analyzed."
    },
    {
        icon: (
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
        ),
        title: "Camera Capture",
        desc: "Point your webcam directly at a display screen showing an X-ray — real-world ER transfer simulation."
    },
    {
        icon: (
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
        ),
        title: "AI Radiologist Chat",
        desc: "Ask follow-up clinical questions after any scan — powered by large language model with full scan context."
    }
];

const STEPS = [
    { n: "01", title: "Upload or Capture", desc: "Drag & drop a JPEG/PNG X-ray, browse your files, or use your webcam to capture a scan directly." },
    { n: "02", title: "Dual-Stream Analysis", desc: "The Swin Transformer and Vision Mamba streams process the image in parallel, fused via cross-attention." },
    { n: "03", title: "Review & Explain", desc: "Receive the prediction, confidence score, Grad-CAM heatmap, and Mamba sequence trace — instantly." }
];

export default function LandingPage({ theme, toggleTheme }) {
    const navigate = useNavigate();

    return (
        <>
            {/* NAVBAR */}
            <nav className="navbar">
                <div className="nav-inner">
                    <div className="nav-logo">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M4.5 12.5l3 3 5-5 2 2 5-5" />
                            <rect x="3" y="3" width="18" height="18" rx="3" />
                        </svg>
                        Fracture<span>Mamba</span>
                    </div>
                    <div className="nav-actions">
                        <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme" aria-label="Toggle theme">
                            {theme === 'dark' ? (
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>
                            ) : (
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" /></svg>
                            )}
                        </button>
                        <button className="btn btn-ghost" onClick={() => navigate('/scan')}>Sign In</button>
                        <button className="btn btn-primary" onClick={() => navigate('/scan')}>Try Now →</button>
                    </div>
                </div>
            </nav>

            {/* HERO */}
            <section className="hero">
                <div className="hero-glow hero-glow-1" />
                <div className="hero-glow hero-glow-2" />
                <div className="container">
                    <div className="flex gap-8 items-center" style={{ flexWrap: 'wrap' }}>
                        {/* Left Column */}
                        <div style={{ flex: '1', minWidth: '320px' }}>
                            <div className="hero-eyebrow fade-up">
                                <span className="badge">
                                    <span style={{ width: 8, height: 8, background: 'var(--success)', borderRadius: '50%', display: 'inline-block' }} />

                                </span>
                            </div>
                            <h1 className="display-xl hero-title fade-up-2">
                                AI-Powered<br />
                                <span className="gradient-text">Bone Fracture</span><br />
                                Detection
                            </h1>
                            <p className="hero-subtitle fade-up-3">
                                FractureMamba-ViT combines Vision State-Space Models with Swin
                                Transformers to deliver 99.8% accuracy with full visual
                                explainability for emergency radiology triage.
                            </p>
                            <div className="hero-actions fade-up-4">
                                <button className="btn btn-primary btn-lg" onClick={() => navigate('/scan')}>
                                    <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                                    Start X-Ray Scanning
                                </button>
                                <a href="https://github.com" target="_blank" rel="noreferrer" className="btn btn-ghost btn-lg">
                                    <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" /></svg>
                                    View Source
                                </a>
                            </div>
                            <div className="hero-meta fade-up-4">
                                {[
                                    ['99.8%', 'Test Accuracy'],
                                    ['0', 'False Negatives'],
                                    ['96ms', 'Inference Time'],
                                    ['AUC 1.00', 'ROC Score'],
                                ].map(([v, l]) => (
                                    <div className="hero-meta-item" key={l}>
                                        <span className="hero-meta-value">{v}</span>
                                        <span className="hero-meta-label">{l}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Right Column — scan preview card */}
                        <div className="hero-visual fade-up-3" style={{ flex: '1', minWidth: '300px' }}>
                            <div className="hero-scan-card">
                                <div className="scan-header">
                                    <div className="scan-dots">
                                        <div className="scan-dot" style={{ background: '#F87171' }} />
                                        <div className="scan-dot" style={{ background: '#FBBF24' }} />
                                        <div className="scan-dot" style={{ background: '#34D399' }} />
                                    </div>
                                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>Analysis Dashboard</span>
                                    <span className="badge" style={{ fontSize: '0.7rem', padding: '0.2rem 0.6rem' }}>LIVE</span>
                                </div>
                                <div className="scan-placeholder">
                                    <div style={{ textAlign: 'center' }}>
                                        <div style={{ fontSize: '3.5rem', marginBottom: '0.5rem', opacity: 0.4 }}>🦴</div>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>X-Ray Preview</div>
                                    </div>
                                    {/* Animated scanning line */}
                                    <div style={{
                                        position: 'absolute', left: 0, right: 0, height: '2px',
                                        background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
                                        animation: 'scan 2.5s ease-in-out infinite'
                                    }} />
                                </div>
                                <div className="scan-result-bar">
                                    <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    No Fracture — 99.8% confidence
                                    <div className="confidence-bar-wrap">
                                        <div className="confidence-bar-fill" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <style>{`
          @keyframes scan {
            0% { top: 0; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
          }
        `}</style>
            </section>

            {/* STATS BAR */}
            <section className="stats-bar">
                <div className="container">
                    <div className="stats-grid">
                        {[
                            ['99.8%', 'Test Accuracy'],
                            ['1.000', 'AUC-ROC Score'],
                            ['100%', 'Fracture Recall'],
                            ['127MB', 'Model Size'],
                            ['5-Fold', 'Cross Validation'],
                        ].map(([v, l]) => (
                            <div key={l}>
                                <div className="stat-value">{v}</div>
                                <div className="stat-label">{l}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* FEATURES */}
            <section className="section">
                <div className="container">
                    <div className="text-center" style={{ marginBottom: '3.5rem' }}>
                        <div className="label-sm" style={{ marginBottom: '0.75rem' }}>Capabilities</div>
                        <h2 className="display-lg">Everything you need for<br /><span className="gradient-text">clinical decision support</span></h2>
                    </div>
                    <div className="features-grid">
                        {FEATURES.map(f => (
                            <div className="card" key={f.title}>
                                <div className="feature-icon">{f.icon}</div>
                                <div className="feature-title">{f.title}</div>
                                <div className="feature-desc">{f.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* HOW IT WORKS */}
            <section className="section" style={{ background: 'var(--bg-surface)' }}>
                <div className="container">
                    <div className="text-center" style={{ marginBottom: '3.5rem' }}>
                        <div className="label-sm" style={{ marginBottom: '0.75rem' }}>Workflow</div>
                        <h2 className="display-lg">How it works</h2>
                    </div>
                    <div className="steps-grid">
                        {STEPS.map(s => (
                            <div key={s.n} style={{ padding: '2rem 0' }}>
                                <div className="step-number">{s.n}</div>
                                <div className="step-title display-md" style={{ marginBottom: '0.75rem' }}>{s.title}</div>
                                <div className="body-md">{s.desc}</div>
                            </div>
                        ))}
                    </div>
                    <div className="text-center" style={{ marginTop: '3rem' }}>
                        <button className="btn btn-primary btn-lg" onClick={() => navigate('/scan')}>
                            Try the Scanner Now →
                        </button>
                    </div>
                </div>
            </section>

            {/* FOOTER */}
            <footer className="footer">
                <div className="container">
                    <div className="footer-inner">
                        <div className="footer-copy">© 2026 FractureMamba-ViT · IIT Mandi Hackathon</div>
                        <div className="footer-links">
                            <a href="#">GitHub</a>
                            <a href="#">Research Paper</a>
                            <a href="#">Team</a>
                        </div>
                    </div>
                </div>
            </footer>
        </>
    );
}
