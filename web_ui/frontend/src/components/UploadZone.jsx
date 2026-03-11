import React, { useState, useRef } from 'react';

const UploadZone = ({ onFileSelect }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [useCamera, setUseCamera] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      setUseCamera(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      streamRef.current = stream;
    } catch (err) {
      alert("Camera access denied or unavailable.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setUseCamera(false);
  };

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        stopCamera();
        onFileSelect(file);
      }, 'image/jpeg', 0.95);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '2rem' }}>
      {!useCamera ? (
        <div 
          className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <div className="upload-text">Drag and drop X-Ray scan here</div>
          <div className="upload-subtext">or click to browse files (JPEG, PNG, DICOM converted)</div>
          
          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }} onClick={e => e.stopPropagation()}>
            <button className="btn-primary" onClick={() => fileInputRef.current?.click()}>
               Select File
            </button>
            <button className="btn-primary" style={{ background: 'transparent', border: '1px solid var(--accent-blue)' }} onClick={startCamera}>
              <svg style={{width:'20px'}} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
              Use Camera
            </button>
          </div>
          
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleChange} 
            accept="image/*" 
            style={{ display: 'none' }} 
          />
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            className="camera-preview fade-in"
          />
          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
            <button className="btn-primary" onClick={captureImage} style={{background: 'var(--accent-green)'}}>
              Capture X-Ray
            </button>
            <button className="btn-primary" onClick={stopCamera} style={{background: 'var(--accent-red)'}}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadZone;
