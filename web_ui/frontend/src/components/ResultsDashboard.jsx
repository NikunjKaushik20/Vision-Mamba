import React, { useState } from 'react';

const ResultsDashboard = ({ result, originalImage, onReset }) => {
    const [activeTab, setActiveTab] = useState('gradcam'); // 'gradcam' | 'mamba'

    // Safety checks
    if (!result || !originalImage) return null;

    const isFractured = result.prediction.toLowerCase().includes('fracture')
        && !result.prediction.toLowerCase().includes('not');

    const statusColor = isFractured ? 'var(--accent-red)' : 'var(--accent-green)';
    const statusIcon = isFractured ? (
        <svg style={{ width: '24px', color: 'var(--accent-red)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
    ) : (
        <svg style={{ width: '24px', color: 'var(--accent-green)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
    );

    return (
        <div className="fade-in">
            {/* Top Banner */}
            <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1.8rem', marginBottom: '0.5rem' }}>
                        {statusIcon}
                        Model Prediction: <span style={{ color: statusColor }}>{result.prediction.toUpperCase()}</span>
                    </h2>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
                        Confidence Level:{' '}
                        <strong style={{ color: 'white' }}>{(result.confidence * 100).toFixed(2)}%</strong>
                    </div>
                </div>
                <button className="btn-primary" onClick={onReset} style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
                    <svg style={{ width: '20px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                    New Scan
                </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                {/* Original Image */}
                <div className="glass-panel" style={{ padding: '1rem' }}>
                    <h3 style={{ marginBottom: '1rem', textAlign: 'center' }}>Input X-Ray</h3>
                    <div style={{ background: 'black', borderRadius: '8px', overflow: 'hidden', display: 'flex', justifyContent: 'center' }}>
                        <img
                            src={URL.createObjectURL(originalImage)}
                            alt="Original"
                            style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }}
                        />
                    </div>
                </div>

                {/* Visualizations Tabbed Panel */}
                <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
                        <button
                            onClick={() => setActiveTab('gradcam')}
                            style={{
                                background: 'none', border: 'none', color: activeTab === 'gradcam' ? 'var(--accent-blue)' : 'var(--text-secondary)',
                                fontSize: '1.1rem', fontWeight: activeTab === 'gradcam' ? '600' : '400', cursor: 'pointer',
                                borderBottom: activeTab === 'gradcam' ? '2px solid var(--accent-blue)' : '2px solid transparent',
                                paddingBottom: '0.2rem'
                            }}
                        >
                            Swin Grad-CAM
                        </button>
                        <button
                            onClick={() => setActiveTab('mamba')}
                            style={{
                                background: 'none', border: 'none', color: activeTab === 'mamba' ? 'var(--accent-purple)' : 'var(--text-secondary)',
                                fontSize: '1.1rem', fontWeight: activeTab === 'mamba' ? '600' : '400', cursor: 'pointer',
                                borderBottom: activeTab === 'mamba' ? '2px solid var(--accent-purple)' : '2px solid transparent',
                                paddingBottom: '0.2rem'
                            }}
                        >
                            Mamba Attention
                        </button>
                    </div>

                    <div style={{ flex: 1, background: 'black', borderRadius: '8px', overflow: 'hidden', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '300px' }}>
                        {activeTab === 'gradcam' ? (
                            <img
                                src={result.gradcam_base64}
                                alt="Grad-CAM Saliency Map"
                                style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }}
                            />
                        ) : (
                            result.mamba_base64 ? (
                                <div style={{ width: '100%', height: '100%', padding: '1rem', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <img
                                        src={result.mamba_base64}
                                        alt="Mamba Sequence State"
                                        style={{ width: '100%', borderRadius: '4px' }}
                                    />
                                    <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '1rem', fontSize: '0.9rem' }}>
                                        Visualizes the state evolution across sequence patches in the Vision Mamba stream. Peak norms indicate areas of high network attention.
                                    </p>
                                </div>
                            ) : (
                                <div style={{ color: 'var(--text-secondary)' }}>Mamba visualization not available.</div>
                            )
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ResultsDashboard;
