import React from 'react';
import { useNavigate } from 'react-router-dom';

const LandingPage = ({ theme, toggleTheme }) => {
  const navigate = useNavigate();

  return (
    <div className="relative flex min-h-screen w-full flex-col overflow-x-hidden font-display transition-colors duration-300">
      {/* Top Navigation */}
      <header className="sticky top-0 z-50 w-full border-b border-landing-primary/10 bg-background-light/80 dark:bg-background-dark/80 backdrop-blur-md">
        <div className="container mx-auto flex h-16 items-center justify-between px-6 lg:px-12">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-landing-primary text-white">
              <span className="material-symbols-outlined">bolt</span>
            </div>
            <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">FractureMamba-ViT</h2>
          </div>

          <nav className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm font-medium hover:text-landing-primary transition-colors">Features</a>
            <a href="#how-it-works" className="text-sm font-medium hover:text-landing-primary transition-colors">Process</a>
            <a href="#" className="text-sm font-medium hover:text-landing-primary transition-colors">Docs</a>
            <button
              onClick={toggleTheme}
              className="text-sm font-medium hover:text-landing-primary transition-colors flex items-center gap-1"
            >
              {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
            </button>
          </nav>

          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/scan')}
              className="hidden sm:flex items-center gap-2 rounded-xl border border-landing-primary/20 bg-landing-primary/5 px-4 py-2 text-sm font-semibold text-landing-primary hover:bg-landing-primary/10 transition-all"
            >
              Sign In
            </button>
            <button
              onClick={() => navigate('/scan')}
              className="flex items-center justify-center rounded-xl bg-landing-primary px-5 py-2.5 text-sm font-bold text-white shadow-lg shadow-landing-primary/20 hover:scale-105 active:scale-95 transition-all"
            >
              Get Started
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative overflow-hidden py-20 lg:py-32">
          <div className="container mx-auto px-6 lg:px-12">
            <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-2">
              <div className="flex flex-col gap-8">
                <div className="inline-flex w-fit items-center gap-2 rounded-full border border-landing-primary/20 bg-landing-primary/10 px-3 py-1 text-xs font-bold uppercase tracking-wider text-landing-primary">
                  <span className="material-symbols-outlined text-sm">rocket_launch</span>
                  Next-Gen Medical AI
                </div>

                <h1 className="text-5xl font-black leading-[1.1] tracking-tight text-slate-900 dark:text-white lg:text-7xl">
                  AI-Powered Fracture Detection at the <span className="text-landing-primary">Speed of Sight</span>
                </h1>

                <p className="max-w-xl text-lg leading-relaxed text-slate-600 dark:text-slate-400">
                  Revolutionizing diagnostics with Mamba-based architecture and a high-performance React/FastAPI stack for real-time clinical insights and unprecedented accuracy.
                </p>

                <div className="flex flex-wrap gap-4">
                  <button
                    onClick={() => navigate('/scan')}
                    className="rounded-xl bg-landing-primary px-8 py-4 text-lg font-bold text-white shadow-xl shadow-landing-primary/30 hover:bg-landing-primary/90 transition-all"
                  >
                    Start Analysis Now
                  </button>

                </div>

                <div className="flex items-center gap-6 pt-4 text-slate-500">
                  <div className="flex flex-col">
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">99.8%</span>
                    <span className="text-xs uppercase tracking-widest">Accuracy</span>
                  </div>
                  <div className="h-8 w-px bg-slate-200 dark:bg-slate-800"></div>
                  <div className="flex flex-col">
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">&lt;200ms</span>
                    <span className="text-xs uppercase tracking-widest">Latency</span>
                  </div>
                </div>
              </div>

              {/* Hero Image / Mockup */}
              <div className="relative">
                <div className="absolute -inset-4 rounded-3xl bg-gradient-to-tr from-landing-primary/20 to-accent-blue/20 blur-3xl"></div>
                <div className="relative overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl">
                  <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuB08HCA1btowUZG42xFyE3r_tGlvj8eifBn3OS4VQKseALSGynDhNu3dHJiZrm-iNpKaq1OLXczGhLpApAY9VDrqzbf7rlodQoJEBszhsAW4KjGoWn-0zVj7sxAlkAfqBZfSie-_xzo2Y3S2DfoyngYFsJoN67mjhswtQatFvaUE4gUWVIyllEhx6frIFaWpwZkjKKiRpgLG1sho0qnZ4Yh6eITl7qvk7ETqL1blJbwQT6JjcxAWS3lABjoesILewIOLEGrp0L2eLER" alt="High tech medical x-ray analysis on digital screen" className="h-full w-full object-cover" />

                  <div className="absolute bottom-4 left-4 right-4 rounded-xl glass-panel p-4 text-white">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-500/20 text-green-400">
                          <span className="material-symbols-outlined text-sm">check_circle</span>
                        </div>
                        <span className="text-sm font-medium">Fracture Detected: Radius Bone</span>
                      </div>
                      <span className="text-xs font-bold text-landing-primary">CONFIDENCE: 98.4%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section id="features" className="py-24 bg-slate-50 dark:bg-black/20">
          <div className="container mx-auto px-6 lg:px-12">
            <div className="mb-16 text-center">
              <h2 className="text-3xl font-bold text-slate-900 dark:text-white lg:text-4xl">Advanced Diagnostic Capabilities</h2>
              <p className="mt-4 text-slate-600 dark:text-slate-400">Precision engineering meets clinical expertise.</p>
            </div>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
              <div className="group flex flex-col gap-4 rounded-2xl border border-slate-200 dark:border-landing-primary/10 bg-white dark:bg-[#2d1e16] p-8 transition-all hover:-translate-y-2 hover:border-landing-primary/50">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-landing-primary/10 text-landing-primary">
                  <span className="material-symbols-outlined">timer</span>
                </div>
                <h3 className="text-xl font-bold dark:text-white">Instant Analysis</h3>
                <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">Color-coded verdicts and confidence scores delivered in milliseconds through optimized CUDA kernels.</p>
              </div>

              <div className="group flex flex-col gap-4 rounded-2xl border border-slate-200 dark:border-landing-primary/10 bg-white dark:bg-[#2d1e16] p-8 transition-all hover:-translate-y-2 hover:border-landing-primary/50">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent-blue/10 text-accent-blue">
                  <span className="material-symbols-outlined">layers</span>
                </div>
                <h3 className="text-xl font-bold dark:text-white">3-Tier Visualization</h3>
                <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">Comprehensive view including Original X-ray, YOLO object detection, and Mamba Spikes attention mapping.</p>
              </div>

              <div className="group flex flex-col gap-4 rounded-2xl border border-slate-200 dark:border-landing-primary/10 bg-white dark:bg-[#2d1e16] p-8 transition-all hover:-translate-y-2 hover:border-landing-primary/50">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-purple-500/10 text-purple-500">
                  <span className="material-symbols-outlined">smart_toy</span>
                </div>
                <h3 className="text-xl font-bold dark:text-white">Radiologist Assistant</h3>
                <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">Context-aware LLM integration providing suggested findings and immediate clinical documentation support.</p>
              </div>

              <div className="group flex flex-col gap-4 rounded-2xl border border-slate-200 dark:border-landing-primary/10 bg-white dark:bg-[#2d1e16] p-8 transition-all hover:-translate-y-2 hover:border-landing-primary/50">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-green-500/10 text-green-500">
                  <span className="material-symbols-outlined">cloud_upload</span>
                </div>
                <h3 className="text-xl font-bold dark:text-white">Seamless Integration</h3>
                <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">Direct DICOM uploads or live medical camera stream capture via high-performance WebSocket connections.</p>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section id="how-it-works" className="py-24">
          <div className="container mx-auto px-6 lg:px-12">
            <div className="flex flex-col items-center gap-16 lg:flex-row">
              <div className="lg:w-1/2">
                <h2 className="text-4xl font-bold dark:text-white">From Scan to Solution in Seconds</h2>
                <p className="mt-4 text-slate-600 dark:text-slate-400">Our pipeline is optimized for clinical workflows, removing friction from emergency diagnostics.</p>

                <div className="mt-12 space-y-12">
                  <div className="flex gap-6">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-landing-primary text-white font-bold">1</div>
                    <div>
                      <h4 className="text-xl font-bold dark:text-white">Upload X-Ray Data</h4>
                      <p className="mt-2 text-slate-600 dark:text-slate-400">Drag and drop standard medical imaging formats (DICOM, JPEG, PNG) into our secure, encrypted gateway.</p>
                    </div>
                  </div>

                  <div className="flex gap-6">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-landing-primary text-white font-bold">2</div>
                    <div>
                      <h4 className="text-xl font-bold dark:text-white">Mamba-ViT Processing</h4>
                      <p className="mt-2 text-slate-600 dark:text-slate-400">Our state-space model architecture analyzes spatial sequences with linear scaling efficiency.</p>
                    </div>
                  </div>

                  <div className="flex gap-6">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-landing-primary text-white font-bold">3</div>
                    <div>
                      <h4 className="text-xl font-bold dark:text-white">Detailed Diagnostic Report</h4>
                      <p className="mt-2 text-slate-600 dark:text-slate-400">Instant visualization of potential fractures with bounding boxes and heatmaps for verification.</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:w-1/2">
                <div className="relative overflow-hidden rounded-3xl bg-slate-200 dark:bg-slate-800 p-2">
                  <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuAJHMn6bA8pWbUjBCqKYbrpwGnmw7E4o4X-NNaIawjw5KfCfbZG_-k7ifcJLKdaXpERNr7pOHoMzAolXYy1OcHLG6bqIxbgVOR_0SewqzxtbbfthcInvAlbzv1MP6uXi58AOGhoQ6wTKu8-rFJ23NnbXX1H8OFP_iutgcP8pMaClgMjUC08yw8FA9-5cyJA45lWRQs3mJGn-OGWYW2akfmPGvyiD8N9B5iYcMXd7fJzAyhHpK0i0WhKvqZYgfiicwsujRx2WmeS3lHT" alt="Laboratory scientist" className="rounded-2xl object-cover" />
                  <div className="absolute inset-0 bg-gradient-to-t from-background-dark/80 to-transparent"></div>

                  <div className="absolute bottom-8 left-8">
                    <div className="flex items-center gap-2 text-white">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-green-500"></span>
                      <span className="text-xs font-bold uppercase tracking-widest">Processing Node #4 Active</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* System Status Banner */}
        <section className="border-y border-slate-200 dark:border-landing-primary/10 py-12">
          <div className="container mx-auto px-6 lg:px-12">
            <div className="flex flex-wrap items-center justify-between gap-8">
              <div className="flex items-center gap-4">
                <span className="material-symbols-outlined text-4xl text-landing-primary">query_stats</span>
                <div>
                  <h3 className="text-lg font-bold dark:text-white">Real-Time Backend Monitoring</h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">System architecture health and model performance tracking.</p>
                </div>
              </div>

              <div className="flex gap-8">
                <div className="text-center">
                  <div className="text-xl font-bold text-green-500">Operational</div>
                  <div className="text-[10px] uppercase tracking-widest text-slate-500">API Gateway</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-green-500">99.99%</div>
                  <div className="text-[10px] uppercase tracking-widest text-slate-500">Uptime</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-landing-primary">FastAPI</div>
                  <div className="text-[10px] uppercase tracking-widest text-slate-500">Engine</div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-slate-50 dark:bg-[#1a120e] pt-20 pb-10 border-t border-slate-200 dark:border-landing-primary/5">
        <div className="container mx-auto px-6 lg:px-12">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-16">
            <div className="flex flex-col gap-6">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-landing-primary text-white">
                  <span className="material-symbols-outlined text-sm">bolt</span>
                </div>
                <h2 className="text-lg font-bold tracking-tight dark:text-white">FractureMamba</h2>
              </div>
              <p className="text-sm leading-relaxed text-slate-500">
                Advancing orthopedic diagnostic accuracy through state-of-the-art vision state-space models.
              </p>
              <div className="flex gap-4">
                <a href="#" className="h-8 w-8 flex items-center justify-center rounded-full bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-landing-primary hover:text-white transition-colors">
                  <span className="material-symbols-outlined text-lg">terminal</span>
                </a>
                <a href="#" className="h-8 w-8 flex items-center justify-center rounded-full bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-landing-primary hover:text-white transition-colors">
                  <span className="material-symbols-outlined text-lg">code</span>
                </a>
              </div>
            </div>

            <div>
              <h4 className="font-bold mb-6 dark:text-white">Documentation</h4>
              <ul className="space-y-4 text-sm text-slate-500">
                <li><a href="#" className="hover:text-landing-primary">Getting Started</a></li>
                <li><a href="#" className="hover:text-landing-primary">API Reference</a></li>
                <li><a href="#" className="hover:text-landing-primary">Mamba Architecture</a></li>
                <li><a href="#" className="hover:text-landing-primary">Model Training</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-bold mb-6 dark:text-white">Resources</h4>
              <ul className="space-y-4 text-sm text-slate-500">
                <li><a href="#" className="hover:text-landing-primary">Research Paper</a></li>
                <li><a href="#" className="hover:text-landing-primary">GitHub Repository</a></li>
                <li><a href="#" className="hover:text-landing-primary">Clinical Trials</a></li>
                <li><a href="#" className="hover:text-landing-primary">Data Ethics</a></li>
              </ul>
            </div>


          </div>

          <div className="pt-8 border-t border-slate-200 dark:border-landing-primary/5 flex flex-col sm:flex-row justify-between items-center gap-4 text-xs text-slate-500">
            <p>© 2026 FractureMamba-ViT. All rights reserved.</p>
            <div className="flex gap-6">
              <a href="#" className="hover:text-landing-primary">Privacy Policy</a>
              <a href="#" className="hover:text-landing-primary">Terms of Service</a>
              <a href="#" className="hover:text-landing-primary">Cookie Settings</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
