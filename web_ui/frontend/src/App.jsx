import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import ScannerApp from './components/ScannerApp';

function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('fm-theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('fm-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light');

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="/scan" element={<ScannerApp theme={theme} toggleTheme={toggleTheme} />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
