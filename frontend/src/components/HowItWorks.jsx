import React from 'react';
import { Upload, Activity, Cpu, ShieldCheck, ArrowRight, ArrowDown } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      num: "01",
      title: "Upload Audio",
      desc: "Ingest any voice recording (MP3, WAV, M4A, OGG, FLAC) and decode into a standardized mono PCM waveform.",
      icon: Upload
    },
    {
      num: "02",
      title: "Extract Acoustic Features",
      desc: "Compute 26 handcrafted acoustic dimensions: Chroma STFT, RMS Energy, Spectral Centroid, Bandwidth, Rolloff, ZCR, and 20 MFCCs.",
      icon: Activity
    },
    {
      num: "03",
      title: "Run 3 Deep Learning Models",
      desc: "Pass normalized features into CNN-RNN Hybrid, Deep 1D CNN, and CNN-Transformer networks trained on balanced audio data.",
      icon: Cpu
    },
    {
      num: "04",
      title: "Ensemble Decision",
      desc: "Apply majority voting consensus (≥2/3 agreement) to produce a robust final authenticity verdict.",
      icon: ShieldCheck
    }
  ];

  return (
    <section id="how-it-works" className="space-y-4 pt-6">
      <div className="text-center space-y-1">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Detection Pipeline</h3>
        <p className="text-lg font-bold text-white">How the Multi-Model Ensemble Works</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 relative">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div
              key={idx}
              className="bg-[#0f121a] border border-[#1c2233] rounded-xl p-5 space-y-3 relative group hover:border-[#2b354d] transition-colors flex flex-col justify-between"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                  {step.num}
                </span>
                <div className="w-8 h-8 rounded-lg bg-[#141926] border border-[#222a3d] flex items-center justify-center text-slate-300">
                  <Icon className="w-4 h-4" />
                </div>
              </div>

              <div className="space-y-1.5">
                <h4 className="text-xs font-semibold text-white">{step.title}</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">{step.desc}</p>
              </div>

              {idx < steps.length - 1 && (
                <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 z-10 text-slate-600">
                  <ArrowRight className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

