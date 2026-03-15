import React, { useState, useEffect, useRef } from 'react';

// Reusable component for drawing the YOLO bounding box
function XRayWithBbox({ imageUrl, bbox, confidence }) {
    const imgRef = useRef(null);
    const canvasRef = useRef(null);
    const wrapperRef = useRef(null);

    const drawBox = () => {
        const img = imgRef.current;
        const canvas = canvasRef.current;
        const wrapper = wrapperRef.current;
        if (!img || !canvas || !bbox) return;

        // Use the image's actual rendered dimensions
        const rect = img.getBoundingClientRect();
        const dw = rect.width;
        const dh = rect.height;

        // Size the canvas to match the image exactly
        canvas.width  = dw;
        canvas.height = dh;
        canvas.style.width  = dw + 'px';
        canvas.style.height = dh + 'px';

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, dw, dh);

        const x1 = bbox.x1 * dw;
        const y1 = bbox.y1 * dh;
        const x2 = bbox.x2 * dw;
        const y2 = bbox.y2 * dh;
        const bw = x2 - x1;
        const bh = y2 - y1;

        ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
        ctx.fillRect(x1, y1, bw, bh);

        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, bw, bh);

        // Label
        const label = `FRACTURE ${(confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 10px Inter, sans-serif';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(x1, y1 > 20 ? y1 - 20 : 0, tw + 8, 20);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x1 + 4, (y1 > 20 ? y1 - 20 : 0) + 14);
    };

    // Re-draw on window resize or when props change
    useEffect(() => {
        drawBox();
        window.addEventListener('resize', drawBox);
        return () => window.removeEventListener('resize', drawBox);
    }, [bbox, imageUrl]);

    return (
        // Outer div centers the image; inner wrapper is sized to the image for accurate canvas overlay
        <div className="w-full h-full flex items-center justify-center">
            <div ref={wrapperRef} style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
                <img
                    ref={imgRef}
                    src={imageUrl}
                    alt="YOLO Analysis"
                    className="max-w-full max-h-full object-contain block"
                    onLoad={drawBox}
                />
                {bbox && (
                    <canvas
                        ref={canvasRef}
                        style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
                    />
                )}
            </div>
        </div>
    );
}

const AnalysisWorkspace = ({ analysisState, currentImage, result, error }) => {
  const [showFullReport, setShowFullReport] = useState(false);

  if (analysisState === 'idle') {
    return (
      <section className="col-span-12 lg:col-span-6 bg-white dark:bg-brandDark/50 p-6 overflow-y-auto flex items-center justify-center">
        <div className="text-center text-slate-500 max-w-sm">
          <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
          </svg>
          <p>Upload an X-ray image or select from recent scans to begin analysis.</p>
        </div>
      </section>
    );
  }

  if (analysisState === 'error') {
    return (
      <section className="col-span-12 lg:col-span-6 bg-white dark:bg-brandDark/50 p-6 overflow-y-auto flex items-center justify-center">
        <div className="text-center text-red-500 max-w-md bg-red-500/10 p-6 rounded-custom border border-red-500/30">
          <svg className="w-12 h-12 mx-auto mb-4 opacity-80" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
          </svg>
          <h3 className="text-lg font-bold mb-2">Analysis Failed</h3>
          <p className="text-sm text-red-400">{error || "An unknown error occurred during analysis."}</p>
        </div>
      </section>
    );
  }

  if (analysisState === 'analyzing') {
    return (
      <section className="col-span-12 lg:col-span-6 bg-white dark:bg-brandDark/50 p-6 overflow-y-auto flex flex-col items-center justify-center">
        <div className="w-24 h-24 mb-6 relative">
          <div className="absolute inset-0 border-4 border-brandBorder rounded-full border-t-primary animate-spin"></div>
        </div>
        <h3 className="text-lg font-medium text-white mb-2">Analyzing Scan...</h3>
        <p className="text-sm text-slate-400">Running Vision-Mamba spatial sequence processing</p>
      </section>
    );
  }

  if (!result) return null;

  // Use ensemble verdict from backend (ViT + YOLO combined decision)
  const isFracture = result.ensemble_verdict === 'fractured';
  const confPercent = (result.confidence * 100).toFixed(1);
  const locStatus = result.localization_status; // 'success' | 'failed' | 'not_applicable'
  // Detect if YOLO overrode ViT (ViT originally said not fractured)
  const wasOverridden = isFracture && result.yolo_image_base64 && result.probabilities && 
    Object.entries(result.probabilities).some(([k, v]) => k.toLowerCase().includes('not') && v > 0.5);

  return (
    <section className="col-span-12 lg:col-span-6 bg-white dark:bg-brandDark/50 p-6 overflow-y-auto transition-colors">
      {/* Verdict Banner */}
      <div className={`border rounded-custom p-4 mb-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${
        isFracture 
          ? 'bg-red-500/10 border-red-500/50 glow-danger' 
          : 'bg-green-500/10 border-green-500/50'
      }`}>
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white shrink-0 ${
            isFracture ? 'bg-red-500' : 'bg-green-500'
          }`}>
            {isFracture ? (
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
              </svg>
            ) : (
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
              </svg>
            )}
          </div>
          <div>
            <h2 className={`text-lg font-bold ${isFracture ? 'text-red-500' : 'text-green-500'}`}>
              {isFracture ? '⚠️ Fracture Detected' : '✅ No Fracture Detected'}
            </h2>
            <p className={`text-xs ${isFracture ? 'text-red-400/80' : 'text-green-400/80'}`}>
              Classification: {result.prediction}
              {wasOverridden && ' (YOLO override)'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {wasOverridden && (
            <span className="text-[9px] bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded border border-amber-500/30 uppercase font-bold whitespace-nowrap">
              YOLO Override
            </span>
          )}
          <button 
            onClick={() => setShowFullReport(!showFullReport)}
            className={`text-xs font-bold uppercase hover:underline whitespace-nowrap ${
              isFracture ? 'text-red-500' : 'text-green-500'
            }`}
          >
            {showFullReport ? 'Hide Detail' : 'Report Detail'}
          </button>
        </div>
      </div>

      {/* Fracture Type — Highlighted outside verdict box */}
      {isFracture && result.fracture_type && (
        <div className="mb-6 flex items-center gap-3 px-4 py-3 rounded-custom bg-orange-500/10 border border-orange-500/30" style={{ boxShadow: '0 0 12px rgba(249, 115, 22, 0.15)' }}>
          <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold shrink-0">Fracture Type</span>
          <span className="text-base font-black text-orange-400 uppercase tracking-wide">
            {result.fracture_type}
          </span>
          {result.fracture_type_confidence && (
            <span className="text-xs text-orange-400/60 font-medium ml-auto shrink-0">
              {(result.fracture_type_confidence * 100).toFixed(1)}% conf
            </span>
          )}
        </div>
      )}
      
      {showFullReport && (
        <div className="mb-6 p-4 rounded-custom bg-brandSurface border border-brandBorder text-sm text-slate-300">
          <h4 className="font-bold text-white mb-2 border-b border-brandBorder pb-2">Clinical Findings</h4>
          {isFracture ? (
            <p>
              {wasOverridden 
                ? `The YOLO object detection model identified a fracture region that the ViT classifier initially missed. The ensemble system has overridden the classification to "Fractured" with ${confPercent}% confidence. Please review the YOLO localization below.`
                : `The Vision-Mamba model has identified structural abnormalities consistent with a fracture. AI confidence is ${confPercent}%.`
              }
              {result.fracture_type && (
                <span className="block mt-1">
                  <strong>Fracture type identified:</strong> {result.fracture_type}
                  {result.fracture_type_confidence && ` (${(result.fracture_type_confidence * 100).toFixed(1)}% confidence)`}
                  {result.openai_override
                    ? '. Classification assisted by OpenAI GPT-4o-mini vision model.'
                    : '. This classification represents the most likely fracture morphology based on visual pattern analysis.'
                  }
                  {result.openai_override && result.openai_reasoning && (
                    <span className="block mt-0.5 text-purple-400 italic">GPT reasoning: "{result.openai_reasoning}"</span>
                  )}
                </span>
              )}
              {locStatus === 'failed' && ' Note: YOLO localization could not identify a specific fracture region — saliency-based heatmap is shown instead.'}
            </p>
          ) : (
            <p>The model did not identify any definitive fracture patterns in the provided scan. AI confidence is {confPercent}%. Clinical correlation is recommended if symptoms persist.</p>
          )}
          <p className="mt-2 text-xs italic text-slate-500">Note: This is an AI-generated screening report and does not replace a professional radiologist's diagnosis.</p>
        </div>
      )}

      {/* AI Confidence Bar */}
      <div className="mb-8">
        <div className="flex justify-between items-end mb-2">
          <span className="text-sm font-semibold text-slate-400">AI Confidence Score</span>
          <span className={`text-2xl font-black italic ${isFracture ? 'text-red-500' : 'text-primary'}`}>
            {confPercent}%
          </span>
        </div>
        <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-1000 ${isFracture ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'bg-primary glow-primary'}`} 
            style={{ width: `${confPercent}%` }}
          ></div>
        </div>
      </div>

      {/* Advanced Visualizations */}
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          {/* Tier 1: Original */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">1. Original X-Ray</h3>
            <div className="aspect-square bg-black rounded-custom border border-brandBorder overflow-hidden relative group flex items-center justify-center p-2">
              <img src={currentImage} alt="Original Scan" className="max-w-full max-h-full object-contain" />
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center cursor-pointer">
                <a href={currentImage} target="_blank" rel="noreferrer" className="text-[10px] bg-white text-black px-2 py-1 rounded">Open Full Size</a>
              </div>
            </div>
          </div>

          {/* Tier 2: YOLO Localisation — driven by ensemble logic */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex justify-between">
              <span>2. Localisation</span>
              {locStatus === 'success' && <span className="text-[9px] bg-red-500/20 text-red-500 px-1.5 py-0.5 rounded border border-red-500/30">YOLO DETECTION</span>}
              {locStatus === 'failed' && <span className="text-[9px] bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded border border-amber-500/30">LOCALIZATION FAILED</span>}
            </h3>
            <div className="aspect-square bg-black rounded-custom border border-brandBorder overflow-hidden relative p-2 flex items-center justify-center">
              {locStatus === 'success' && result.yolo_image_base64 ? (
                  <img src={result.yolo_image_base64} alt="YOLO Fracture Detection" className="max-w-full max-h-full object-contain" />
              ) : locStatus === 'failed' ? (
                  /* ViT says fracture but YOLO couldn't localize — show GradCAM as fallback */
                  result.gradcam_base64 ? (
                    <div className="relative w-full h-full flex items-center justify-center">
                      <img src={result.gradcam_base64} alt="Saliency Heatmap" className="max-w-full max-h-full object-contain" />
                      <div className="absolute bottom-2 left-2 right-2 bg-amber-500/10 border border-amber-500/30 rounded px-2 py-1.5 text-center">
                        <p className="text-[10px] text-amber-400 font-bold uppercase">YOLO Localization Failed</p>
                        <p className="text-[9px] text-amber-400/70">Saliency heatmap shown as fallback</p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-amber-400">
                      <svg className="w-10 h-10 mx-auto mb-2 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                      <p className="text-sm font-medium">Localization Failed</p>
                      <p className="text-xs mt-1 text-amber-400/70">Fracture detected but YOLO could not localize the region</p>
                    </div>
                  )
              ) : locStatus === 'not_applicable' ? (
                  /* Both agree no fracture — don't show YOLO image */
                  <div className="text-center text-slate-500">
                    <svg className="w-10 h-10 mx-auto mb-2 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M5 13l4 4L19 7"></path></svg>
                    <p className="text-sm">No fracture region to localize</p>
                    <p className="text-xs mt-1">Both ViT and YOLO agree: no fracture detected</p>
                  </div>
              ) : (
                  <div className="text-center text-slate-500">
                    <p className="text-sm">No localisation data</p>
                    <p className="text-xs mt-1">YOLO model not available</p>
                  </div>
              )}
            </div>
          </div>
        </div>

        {/* Tier 3: Bone Pattern Analysis (Mamba) */}
        <div className="space-y-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">3. Bone Pattern Analysis (Mamba States)</h3>
          <div className="bg-brandDark rounded-custom border border-brandBorder relative flex items-center justify-center overflow-hidden min-h-[12rem]">
            {result.mamba_base64 ? (
                <img src={result.mamba_base64} alt="Mamba Sequence State" className="w-full object-contain p-2" />
            ) : (
                <div className="text-slate-500 text-sm py-12">Mamba state visualization not available for this scan</div>
            )}
            
            {result.mamba_base64 && (
              <div className="absolute bottom-2 right-4 flex gap-4 bg-brandDark/80 px-2 py-1 rounded backdrop-blur border border-white/5">
                <div className="flex items-center gap-1">
                  <span className="text-[10px] text-slate-400">Peaks indicate anomalous pattern attention</span>
                </div>
              </div>
            )}
          </div>
        </div>

      </div>
    </section>
  );
};

export default AnalysisWorkspace;
