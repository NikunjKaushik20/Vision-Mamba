import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import AnalysisWorkspace from './AnalysisWorkspace';
import ChatAssistant from './ChatAssistant';

const Dashboard = ({ theme, toggleTheme }) => {
  const navigate = useNavigate();
  // 'idle', 'analyzing', 'complete', 'error'
  const [analysisState, setAnalysisState] = useState('idle');
  const [currentImage, setCurrentImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null); // null = checking

  const [recentScans, setRecentScans] = useState([]);

  // Check backend health on mount and every 10s
  useEffect(() => {
      const checkHealth = async () => {
          try {
              const res = await fetch('http://localhost:8000/health', { signal: AbortSignal.timeout(3000) });
              setBackendOnline(res.ok);
          } catch {
              setBackendOnline(false);
          }
      };
      
      checkHealth();
      const id = setInterval(checkHealth, 10000);
      return () => clearInterval(id);
  }, []);

  const handleFileUpload = async (file) => {
    if (!backendOnline) {
      setError('Backend is offline. Please ensure FastAPI is running on port 8000.');
      return;
    }

    const imageUrl = URL.createObjectURL(file);
    setCurrentImage(imageUrl);
    await triggerAnalysisSequence(file, imageUrl, file.name);
  };

  const handleCameraCapture = async (file) => {
    if (!backendOnline) {
      setError('Backend is offline. Please ensure FastAPI is running on port 8000.');
      return;
    }

    const imageUrl = URL.createObjectURL(file);
    setCurrentImage(imageUrl);
    await triggerAnalysisSequence(file, imageUrl, 'camera_capture.jpg');
  };

  const triggerAnalysisSequence = async (file, imageUrl, filename) => {
    setAnalysisState('analyzing');
    setError(null);
    setPredictionResult(null);
    
    // Add to recent scans temporarily
    const tempScanId = Math.random().toString(36).substring(7);
    const newScan = {
      id: tempScanId,
      name: filename,
      time: 'Just now',
      thumb: imageUrl,
      fullImage: imageUrl,
      file: file // Keep reference to file for history selection if needed
    };
    
    setRecentScans([newScan, ...recentScans]);

    // Actual API Call
    const fd = new FormData();
    fd.append('file', file);
    
    try {
        const res = await fetch('http://localhost:8000/predict', { method: 'POST', body: fd });
        if (!res.ok) throw new Error('Backend error — prediction failed.');
        const data = await res.json();
        
        if (data.rejected) {
            setError(data.reject_reason || 'The uploaded image does not appear to be a medical X-ray.');
            setAnalysisState('error');
            return;
        }

        setPredictionResult(data);
        setAnalysisState('complete');
        
        // Update the scan object with the result
        setRecentScans(prev => prev.map(s => 
          s.id === tempScanId ? { ...s, result: data } : s
        ));
    } catch (e) {
        setError(e.message);
        setAnalysisState('error');
    }
  };

  const handleSelectScan = (scan) => {
    setCurrentImage(scan.fullImage);
    if (scan.result) {
      setPredictionResult(scan.result);
      setAnalysisState('complete');
      setError(null);
    } else if (scan.file) {
      // Re-run if we have the file but no result
      triggerAnalysisSequence(scan.file, scan.fullImage, scan.name);
    }
  };

  return (
    <div className="bg-slate-100 dark:bg-brandDark text-slate-800 dark:text-slate-200 font-sans min-h-screen flex flex-col h-screen overflow-hidden transition-colors duration-300">
      {/* Header */}
      <header className="border-b border-slate-200 dark:border-brandBorder bg-white dark:bg-brandSurface px-6 py-4 flex items-center justify-between shrink-0 transition-colors">
        <div className="flex items-center gap-4">
          <div 
            className="flex items-center gap-2 cursor-pointer transition-opacity hover:opacity-80"
            onClick={() => navigate('/')}
          >
            <div className="w-8 h-8 bg-primary rounded-custom flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
              </svg>
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white hidden sm:block">FractureMamba-ViT</h1>
          </div>
          {/* Backend Health Status */}
          <div className={`ml-2 sm:ml-6 flex items-center gap-2 px-3 py-1 border rounded-full ${
            backendOnline === null ? 'bg-amber-500/10 border-amber-500/30' :
            backendOnline ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              backendOnline === null ? 'bg-amber-500' :
              backendOnline ? 'bg-green-500 animate-pulse-green' : 'bg-red-500'
            }`}></span>
            <span className={`text-xs font-medium uppercase tracking-wider hidden sm:inline ${
              backendOnline === null ? 'text-amber-400' :
              backendOnline ? 'text-green-400' : 'text-red-400'
            }`}>
              {backendOnline === null ? 'Connecting...' : backendOnline ? 'Model Ready' : 'Backend Offline'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* User Profile */}
          <div className="flex items-center gap-3 pl-6 border-l border-slate-200 dark:border-brandBorder">
            <div className="text-right hidden sm:block">
              <p className="text-sm font-medium text-slate-900 dark:text-white">Guest User</p>
              <p className="text-[10px] text-slate-500 uppercase">Viewer</p>
            </div>
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-400 to-orange-600 dark:from-primary dark:to-orange-600 border border-white/20 flex items-center justify-center overflow-hidden">
              <div className="text-white text-md font-bold">GU</div>
            </div>
          </div>
          {/* Theme Toggle */}
          <button 
            onClick={toggleTheme}
            className="p-2 text-slate-400 hover:text-primary transition-colors" 
            title="Toggle Theme"
          >
            {theme === 'dark' ? (
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>
              </svg>
            ) : (
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
              </svg>
            )}
          </button>
        </div>
      </header>

      {/* Backend offline warning banner */}
      {backendOnline === false && (
          <div className="bg-red-500/10 border-b border-red-500/20 px-6 py-2 text-sm text-red-500 flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
              <span><strong>Backend Offline:</strong> Start the FastAPI server by running <code className="bg-black/20 px-1 rounded">cd web_ui/backend &amp;&amp; uvicorn main:app --port 8000</code></span>
          </div>
      )}

      {/* Main Workspace */}
      <main className="flex-1 grid grid-cols-12 gap-0 overflow-y-auto lg:overflow-hidden h-full relative">
        <Sidebar 
          onUpload={handleFileUpload}
          onCameraCapture={handleCameraCapture}
          recentScans={recentScans}
          onSelectScan={handleSelectScan}
        />
        
        <AnalysisWorkspace 
          analysisState={analysisState}
          currentImage={currentImage}
          result={predictionResult}
          error={error}
        />
        
        <ChatAssistant 
          analysisState={analysisState}
          predictionCtx={predictionResult}
        />
      </main>

      {/* Footer Status */}
      <footer className="bg-white dark:bg-brandSurface border-t border-slate-200 dark:border-brandBorder px-6 py-2 flex items-center justify-between text-[10px] text-slate-500 uppercase font-medium shrink-0 transition-colors">
        <div className="flex gap-6">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">System Load:</span>
            <span className="text-green-500">12%</span>
          </div>
          <div className="flex items-center gap-1.5 hidden sm:flex">
            <span className="text-slate-400">Latency:</span>
            <span className="text-primary">45ms</span>
          </div>
        </div>
        <div className="flex gap-4">
          <span>Ver: 0.8.4-alpha</span>
          <span className="text-slate-400 hidden sm:inline">© 2026 FractureMamba Labs</span>
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;
