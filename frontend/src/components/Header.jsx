import React from 'react';
import { Activity, Cpu, Layers, HelpCircle } from 'lucide-react';

export default function Header({ onOpenArchitecture, onScrollToHowItWorks }) {
  return (
    <header className="border-b border-[#1c2233] bg-[#0b0d14]/90 backdrop-blur-md sticky top-0 z-30 px-4 sm:px-8 py-3.5">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        
        {/* Brand & Identity */}
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-[#141926] border border-[#222a3d] flex items-center justify-center text-blue-400 shrink-0">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-tight">
              Deepfake Audio Detector
            </h1>
            <p className="text-xs text-slate-400">
              AI-powered audio authenticity analysis
            </p>
          </div>
        </div>

        {/* Navigation Actions */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-[#121622] border border-[#1e2538] text-[11px] text-slate-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>3 Models Active</span>
          </div>

          <button
            onClick={onScrollToHowItWorks}
            className="flex items-center space-x-1 px-3 py-1.5 rounded-md hover:bg-[#151a27] text-slate-300 hover:text-white text-xs font-medium transition-colors"
          >
            <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
            <span className="hidden sm:inline">How It Works</span>
          </button>

          <button
            onClick={onOpenArchitecture}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-[#141926] hover:bg-[#1a2133] border border-[#222a3d] text-xs font-medium text-slate-200 transition-colors"
          >
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>Architecture</span>
          </button>
        </div>

      </div>
    </header>
  );
}

