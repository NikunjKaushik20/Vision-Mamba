import React, { useRef, useState } from 'react';

const Sidebar = ({ onUpload, onCameraCapture, recentScans, onSelectScan }) => {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [usingCamera, setUsingCamera] = useState(false);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  const startCamera = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        streamRef.current = stream;
        setUsingCamera(true);
        if (videoRef.current) videoRef.current.srcObject = stream;
    } catch { 
        alert('Camera access denied or unavailable.'); 
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
    }
    setUsingCamera(false);
  };

  const captureCamera = () => {
    if (videoRef.current) {
        const c = document.createElement('canvas');
        c.width = videoRef.current.videoWidth; 
        c.height = videoRef.current.videoHeight;
        const ctx = c.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0);
        
        c.toBlob(blob => { 
            stopCamera(); 
            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            onCameraCapture(file);
        }, 'image/jpeg', 0.95);
    }
  };

  return (
    <section className="col-span-12 lg:col-span-3 border-r border-slate-200 dark:border-brandBorder p-6 overflow-y-auto bg-white dark:bg-transparent transition-colors">
      <div className="space-y-6">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-400 mb-4">Input Source</h2>
          
          <div 
            className="border-2 border-dashed border-slate-300 dark:border-brandBorder hover:border-primary/50 transition-colors bg-slate-50 dark:bg-brandSurface/50 rounded-custom p-8 text-center flex flex-col items-center justify-center gap-4 cursor-pointer group"
            onClick={handleUploadClick}
          >
            <div className="w-16 h-16 rounded-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
              <svg className="h-8 w-8 text-slate-400 group-hover:text-primary transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Drag & Drop X-ray Image</p>
              <p className="text-xs text-slate-500 mt-1">DICOM, PNG, JPG supported</p>
            </div>
            <button className="mt-2 w-full py-2 bg-primary hover:bg-primary/90 text-white font-medium rounded-custom transition-all">
              Upload File
            </button>
            <input 
              type="file" 
              className="hidden" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept=".jpg,.jpeg,.png,.dicom" 
            />
          </div>
        </div>

        <div className="pt-4 border-t border-slate-200 dark:border-brandBorder">
            {!usingCamera ? (
              <button 
                onClick={startCamera}
                className="w-full py-3 flex items-center justify-center gap-2 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-custom transition-colors"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
                </svg>
                <span>Live Camera Capture</span>
              </button>
            ) : (
              <div className="flex flex-col gap-2">
                <div className="relative rounded-custom overflow-hidden bg-black aspect-video">
                  <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
                </div>
                <div className="flex gap-2">
                   <button onClick={captureCamera} className="flex-1 py-2 bg-green-600 hover:bg-green-500 text-white rounded-custom transition-colors text-sm font-medium">Capture</button>
                   <button onClick={stopCamera} className="w-10 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-custom transition-colors flex items-center justify-center">
                     <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                   </button>
                </div>
              </div>
            )}
        </div>

        <div className="mt-8">
          <h3 className="text-xs font-semibold text-slate-500 uppercase mb-3">Recent Scans</h3>
          <div className="space-y-3">
            {recentScans.map((scan) => (
              <div 
                key={scan.id} 
                className="flex items-center gap-3 p-2 rounded-custom hover:bg-slate-100 dark:hover:bg-brandSurface cursor-pointer transition-colors"
                onClick={() => onSelectScan(scan)}
              >
                <div className="w-12 h-12 bg-slate-700 rounded-custom overflow-hidden">
                  <img src={scan.thumb} alt="Scan thumb" className="w-full h-full object-cover grayscale opacity-60" />
                </div>
                <div>
                  <p className="text-xs font-medium text-slate-700 dark:text-slate-300 w-32 truncate" title={scan.name}>{scan.name}</p>
                  <p className="text-[10px] text-slate-500">{scan.time}</p>
                </div>
              </div>
            ))}
            
            {recentScans.length === 0 && (
              <p className="text-xs text-slate-500 italic">No recent scans</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Sidebar;
