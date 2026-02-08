'use client';

import { useState, useEffect } from 'react';
import AnimatedSphere from './AnimatedSphere';

// --- Icons ---
const PlayIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
    <path d="M8 5v14l11-7z" />
  </svg>
);

const ArrowRightIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
  </svg>
);

const UploadCloudIcon = () => (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

const ChatBubbleIcon = () => (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
  </svg>
);

const LightBulbIcon = () => (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const CheckCircleIcon = () => (
  <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

// --- Components ---

export default function LandingPage({ onGetStarted }: { onGetStarted: () => void }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-[#050505] text-white overflow-hidden font-sans selection:bg-cyan-500/30">

      {/* Background Gradients */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px] animate-pulse" style={{ animationDuration: '8s' }} />
        <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-cyan-600/10 rounded-full blur-[100px] animate-pulse" style={{ animationDuration: '10s', animationDelay: '1s' }} />
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-[0.03]"></div>
      </div>

      {/* --- HERO SECTION --- */}
      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 container mx-auto px-6">
        <div className="flex flex-col lg:flex-row items-center gap-16">

          <div className="flex-1 space-y-8 z-10 animate-fade-in-up">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
              </span>
              v2.0 Now Available with Voice Control
            </div>

            <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1]">
              Your Autonomous <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-emerald-400 animate-gradient-x">
                Data Analyst
              </span>
            </h1>

            <p className="text-xl text-gray-400 max-w-xl leading-relaxed">
              Stop fighting with spreadsheets and Python scripts.
              Just upload your data and <strong>talk</strong>. IntelliML builds models,
              finds insights, and visualizes trends—automatically.
            </p>

            <div className="flex flex-wrap items-center gap-4 pt-4">
              <button
                onClick={onGetStarted}
                className="group relative px-8 py-4 bg-white text-black rounded-lg font-bold text-lg hover:bg-gray-100 transition-all hover:scale-105 shadow-[0_0_20px_rgba(255,255,255,0.3)]"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/10 to-blue-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out"></div>
                <span className="flex items-center gap-2">
                  Start Analysis Free
                  <ArrowRightIcon />
                </span>
              </button>

              <a
                href="https://www.youtube.com"
                target="_blank"
                rel="noopener noreferrer"
                className="px-8 py-4 flex items-center gap-2 rounded-lg font-semibold text-gray-300 hover:text-white border border-white/10 hover:border-white/30 hover:bg-white/5 transition-all"
              >
                <PlayIcon />
                Watch Demo
              </a>
            </div>

            <div className="pt-8 flex items-center gap-6 text-sm text-gray-500 font-medium">
              <span className="flex items-center gap-2"><CheckCircleIcon /> No Coding Required</span>
              <span className="flex items-center gap-2"><CheckCircleIcon /> Instant Results</span>
              <span className="flex items-center gap-2"><CheckCircleIcon /> Secure & Private</span>
            </div>
          </div>

          <div className="flex-1 w-full relative h-[500px] flex items-center justify-center lg:justify-end">
            {/* Abstract Visualization of "AI Thinking" */}
            <div className="relative w-full max-w-lg aspect-square">
              <div className="absolute inset-0 bg-blue-500/20 blur-[100px] rounded-full animate-pulse"></div>
              <div className="relative z-10 w-full h-full scale-125">
                <AnimatedSphere />
              </div>

              {/* Floating Cards simulating analysis */}
              <div className="absolute top-10 left-0 bg-slate-900/80 backdrop-blur-md border border-white/10 p-4 rounded-xl shadow-2xl shadow-black/50 animate-float" style={{ animationDelay: '0s' }}>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded bg-blue-500/20 flex items-center justify-center text-blue-400"><ChatBubbleIcon /></div>
                  <div className="text-sm font-semibold">"Predict churn for Q3"</div>
                </div>
                <div className="h-1 w-24 bg-blue-500/30 rounded-full overflow-hidden">
                  <div className="h-full w-2/3 bg-blue-400 rounded-full animate-progress"></div>
                </div>
              </div>

              <div className="absolute bottom-20 right-0 bg-slate-900/80 backdrop-blur-md border border-white/10 p-4 rounded-xl shadow-2xl shadow-black/50 animate-float" style={{ animationDelay: '2s' }}>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded bg-emerald-500/20 flex items-center justify-center text-emerald-400"><CheckCircleIcon /></div>
                  <div>
                    <div className="text-xs text-gray-400">Model Accuracy</div>
                    <div className="text-xl font-bold text-white">94.8%</div>
                  </div>
                </div>
              </div>

            </div>
          </div>
        </div>
      </section>

      {/* --- HOW IT WORKS --- */}
      <section className="py-24 bg-slate-900/30 border-y border-white/5 backdrop-blur-sm">
        <div className="container mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Data Science, Democratized.</h2>
            <p className="text-gray-400">
              You don't need a PhD to build machine learning models.
              IntelliML handles the complex math while you focus on the strategy.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <StepCard
              number="01"
              icon={<UploadCloudIcon />}
              title="Upload Data"
              desc="Drag & drop your CSV or Excel files. We automatically clean, format, and detect types."
            />
            <StepCard
              number="02"
              icon={<ChatBubbleIcon />}
              title="Ask Questions"
              desc="Use your voice or text. 'Analyze sales trends' or 'Train a model to predict price'."
            />
            <StepCard
              number="03"
              icon={<LightBulbIcon />}
              title="Get Insights"
              desc="Receive interactive charts, key metrics, and actionable AI-driven recommendations instantly."
            />
          </div>
        </div>
      </section>

      {/* --- USE CASES --- */}
      <section className="py-24 container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Built for <span className="text-blue-400">Every Team</span>
            </h2>
            <p className="text-gray-400 mb-8 text-lg">
              Whether you're in finance, marketing, or operations, IntelliML adapts to your data.
            </p>

            <div className="space-y-4">
              <UseCaseRow title="Marketing Teams" desc="Predict customer churn and optimize campaign ROI." />
              <UseCaseRow title="Financial Analysts" desc="Forecast revenue and detect anomalies in transaction data." />
              <UseCaseRow title="Product Managers" desc="Analyze user feedback sentiment and prioritize features." />
              <UseCaseRow title="Researchers" desc="Process experimental data and identify correlations quickly." />
            </div>

            <div className="mt-8">
              <a href="#" className="text-cyan-400 hover:text-cyan-300 font-medium flex items-center gap-2">
                View all use cases <ArrowRightIcon />
              </a>
            </div>
          </div>

          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-tr from-blue-600/20 to-purple-600/20 rounded-2xl blur-2xl"></div>
            <div className="relative bg-slate-950 border border-white/10 rounded-2xl p-8 shadow-2xl">
              <div className="flex items-center justify-between mb-8 border-b border-white/10 pb-4">
                <div>
                  <div className="text-xs text-gray-500 uppercase tracking-wider">Project</div>
                  <div className="font-semibold text-white">Q3 Sales Forecast</div>
                </div>
                <div className="px-3 py-1 bg-emerald-500/10 text-emerald-400 text-xs rounded-full border border-emerald-500/20">Analysis Complete</div>
              </div>

              <div className="space-y-4">
                <div className="h-32 bg-slate-900 rounded-lg border border-white/5 relative overflow-hidden group">
                  <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-sm group-hover:text-cyan-400 transition-colors">
                    Interactive Visualization
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-1 h-20 bg-slate-900 rounded-lg border border-white/5 p-3">
                    <div className="text-xs text-gray-500">R2 Score</div>
                    <div className="text-2xl font-bold text-white mt-1">0.892</div>
                  </div>
                  <div className="flex-1 h-20 bg-slate-900 rounded-lg border border-white/5 p-3">
                    <div className="text-xs text-gray-500">MSE</div>
                    <div className="text-2xl font-bold text-white mt-1">124.5</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --- FOOTER --- */}
      <footer className="border-t border-white/10 bg-black pt-16 pb-8">
        <div className="container mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            <div className="col-span-1 md:col-span-2">
              <h3 className="text-xl font-bold text-white mb-4">IntelliML</h3>
              <p className="text-gray-500 max-w-xs">
                Empowering teams to make data-driven decisions with the power of autonomous AI.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">Platform</h4>
              <ul className="space-y-2 text-gray-500 text-sm">
                <li><a href="#" className="hover:text-blue-400 transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">Security</a></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">Enterprise</a></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">Pricing</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">Resources</h4>
              <ul className="space-y-2 text-gray-500 text-sm">
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Documentation</a></li>
                <li><a href="#" className="hover:text-cyan-400 transition-colors">API Reference</a></li>
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Community</a></li>
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Help Center</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-white/5 pt-8 flex flex-col md:flex-row items-center justify-between text-xs text-gray-600">
            <div>&copy; 2026 IntelliML Inc. All rights reserved.</div>
            <div className="flex gap-6 mt-4 md:mt-0">
              <a href="#" className="hover:text-white">Privacy</a>
              <a href="#" className="hover:text-white">Terms</a>
              <a href="#" className="hover:text-white">Twitter</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

function StepCard({ number, icon, title, desc }: any) {
  return (
    <div className="relative group p-8 rounded-2xl bg-gradient-to-b from-white/5 to-white/[0.02] border border-white/10 hover:border-blue-500/30 transition-all hover:translate-y-[-5px]">
      <div className="absolute top-4 right-6 text-4xl font-bold text-white/5 group-hover:text-blue-500/10 transition-colors font-mono">
        {number}
      </div>
      <div className="w-14 h-14 rounded-xl bg-blue-600/20 flex items-center justify-center text-blue-400 mb-6 group-hover:bg-blue-600 group-hover:text-white transition-all">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-white mb-3">{title}</h3>
      <p className="text-gray-400 leading-relaxed text-sm">
        {desc}
      </p>
    </div>
  );
}

function UseCaseRow({ title, desc }: any) {
  return (
    <div className="flex items-start gap-4 p-4 rounded-xl hover:bg-white/5 transition-colors cursor-default">
      <div className="mt-1 w-2 h-2 rounded-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]"></div>
      <div>
        <h4 className="text-white font-medium">{title}</h4>
        <p className="text-sm text-gray-500">{desc}</p>
      </div>
    </div>
  );
}