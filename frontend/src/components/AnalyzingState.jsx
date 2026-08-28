import React, { useState, useEffect } from 'react';
import { Loader2, CheckCircle2, Circle, Activity } from 'lucide-react';

export default function AnalyzingState({ audioName }) {
  const [stepIndex, setStepIndex] = useState(0);

  const steps = [
    { label: "Audio file loaded & decoded", desc: "Verifying sample rate and mono channel stream" },
    { label: "Extracting 26 acoustic features", desc: "Calculating Chroma, RMS, Spectral Centroid, Bandwidth, Rolloff, ZCR, MFCC 1-20" },
    { label: "Running 3 deep learning models", desc: "Evaluating CNN-RNN, Deep CNN, and Transformer with Positional Encoding" },
    { label: "Calculating ensemble decision", desc: "Aggregating individual probabilities & computing majority consensus" },
  ];

  useEffect(() => {
    const timer1 = setTimeout(() => setStepIndex(1), 500);
    const timer2 = setTimeout(() => setStepIndex(2), 1200);
    const timer3 = setTimeout(() => setStepIndex(3), 2000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, []);

  return (
    <div className="bg-[#0f121a] border border-[#1c2233] rounded-xl p-8 shadow-sm space-y-6 max-w-2xl mx-auto my-6">
      
      {/* Header */}
      <div className="flex items-center space-x-3 pb-4 border-b border-[#1c2233]">
        <div className="w-10 h-10 rounded-lg bg-[#141926] border border-[#222a3d] flex items-center justify-center text-blue-400">
          <Activity className="w-5 h-5 animate-pulse" />
        </div>
        <div>
          <h3 className="text-base font-semibold text-white">Analyzing Audio</h3>
          <p className="text-xs font-mono text-slate-400 truncate max-w-md">{audioName || "Audio Stream"}</p>
        </div>
      </div>

      {/* Indeterminate Scanning Line */}
      <div className="space-y-2">
        <div className="h-1.5 w-full bg-[#161c2b] rounded-full overflow-hidden relative">
          <div className="absolute top-0 bottom-0 left-0 w-1/3 bg-blue-500 rounded-full animate-scan"></div>
        </div>
        <p className="text-[11px] font-mono text-slate-400 text-right">Inference in progress...</p>
      </div>

      {/* Forensic Pipeline Steps */}
      <div className="space-y-3 pt-2">
        {steps.map((step, idx) => {
          const isDone = idx < stepIndex;
          const isCurrent = idx === stepIndex;

          return (
            <div
              key={idx}
              className={`flex items-start space-x-3 p-2.5 rounded-lg transition-colors ${
                isCurrent ? 'bg-[#141926] border border-[#222a3d]' : ''
              }`}
            >
              <div className="mt-0.5 shrink-0">
                {isDone ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : isCurrent ? (
                  <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                ) : (
                  <Circle className="w-4 h-4 text-slate-600" />
                )}
              </div>

              <div>
                <p className={`text-xs font-medium ${
                  isDone ? 'text-slate-300' : isCurrent ? 'text-white font-semibold' : 'text-slate-500'
                }`}>
                  {step.label}
                </p>
                <p className="text-[11px] text-slate-400">{step.desc}</p>
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}

