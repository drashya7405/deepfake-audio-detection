import React from 'react';
import { Activity, ShieldCheck, Github } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-[#1c2233] bg-[#0b0d14] py-6 px-4 sm:px-8 mt-12 text-xs text-slate-400">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        
        <div className="flex items-center space-x-2.5">
          <div className="w-6 h-6 rounded bg-[#141926] border border-[#222a3d] flex items-center justify-center text-blue-400">
            <Activity className="w-3.5 h-3.5" />
          </div>
          <span className="text-slate-300 font-medium">Deepfake Audio Detector</span>
          <span className="text-slate-400">·</span>
          <span>AI Forensics & Authenticity Ensemble</span>
        </div>

        <div className="flex items-center space-x-4 text-[11px] text-slate-400">
          <span>Trained on 11,778 Balanced Samples</span>
          <span>•</span>
          <span>FastAPI + TensorFlow + Librosa</span>
        </div>

      </div>
    </footer>
  );
}

