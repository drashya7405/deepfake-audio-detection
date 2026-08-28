import React from 'react';
import { Waves, Cpu, CheckCircle2, ShieldCheck } from 'lucide-react';

export default function Hero() {
  return (
    <section className="text-center space-y-4 max-w-3xl mx-auto pt-2 pb-1">
      
      {/* Capability Indicator Badge */}
      <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#121622] border border-[#1e2538] text-xs text-slate-300">
        <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
        <span className="font-medium text-slate-300">Audio Forensics & Synthetic Speech Detection</span>
      </div>

      {/* Main Headline */}
      <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-white">
        Detect AI-generated and manipulated speech using a 3-model ensemble.
      </h2>

      {/* Subtitle */}
      <p className="text-sm text-slate-400 max-w-2xl mx-auto leading-relaxed">
        Extract 26 acoustic spectral & cepstral features from any audio recording and classify speech authenticity through combined CNN-RNN, Deep CNN, and Transformer models.
      </p>

      {/* Capability Metadata Pills */}
      <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
        <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-[#0f131d] border border-[#1b2234] text-[11px] font-mono text-slate-300">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
          <span>26 Acoustic Features</span>
        </div>

        <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-[#0f131d] border border-[#1b2234] text-[11px] font-mono text-slate-300">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
          <span>3 Deep Learning Models</span>
        </div>

        <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-[#0f131d] border border-[#1b2234] text-[11px] font-mono text-slate-300">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          <span>Majority Ensemble Consensus</span>
        </div>
      </div>

    </section>
  );
}

