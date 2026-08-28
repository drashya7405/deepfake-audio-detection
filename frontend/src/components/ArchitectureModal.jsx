import React from 'react';
import { X, Layers, Cpu, Database, CheckCircle2, Shield } from 'lucide-react';

export default function ArchitectureModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0f121a] border border-[#222a3d] rounded-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 space-y-6 shadow-2xl">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1c2233] pb-4">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#161c2b] border border-[#222a3d] flex items-center justify-center text-blue-400">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">System Architecture & Forensics Pipeline</h3>
              <p className="text-[11px] text-slate-400">Deep Learning & Soft Computing Deepfake Audio Detector</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-[#161c2b] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-5 text-xs text-slate-300 leading-relaxed">
          
          {/* Visual Pipeline ASCII Box */}
          <div className="space-y-1.5">
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider text-blue-400">
              Forensic Detection Pipeline
            </h4>
            <pre className="bg-[#0a0d14] border border-[#181e2e] p-4 rounded-lg font-mono text-[11px] text-slate-300 overflow-x-auto leading-relaxed">
{`Audio Input (MP3 / WAV / M4A)
     ↓
Acoustic Feature Extraction (Librosa)
     ↓
26 Features: [Chroma, RMS, Centroid, Bandwidth, Rolloff, ZCR, MFCC 1-20]
     ↓
StandardScaler Normalization → Reshape (1, 26, 1)
     ↓
┌─────────────────┬──────────────────┬──────────────────┐
│  Drashya Model  │   Devesh Model   │   Swayam Model   │
│  (CNN + RNN)    │   (Deep 1D CNN)  │  (Transformer)   │
└────────┬────────┴────────┬─────────┴────────┬─────────┘
         │                 │                  │
         ▼                 ▼                  ▼
    Vote 1 (P1)       Vote 2 (P2)        Vote 3 (P3)
         │                 │                  │
         └─────────────────┼──────────────────┘
                           ↓
                Majority Voting Engine (≥ 2/3)
                           ↓
               REAL or AI GENERATED SYNTHETIC`}
            </pre>
          </div>

          {/* Model Family Breakdown */}
          <div className="space-y-2.5">
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider text-blue-400">
              The 3 Deep Learning Architectures
            </h4>

            <div className="space-y-2">
              <div className="p-3 rounded-lg bg-[#0a0d14] border border-[#181e2e] space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">1. Drashya Model (CNN-RNN Hybrid)</span>
                  <span className="font-mono text-[10px] text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">98.59% Accuracy</span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  Combines 1D Convolutional feature extractors with Recurrent (LSTM/GRU) layers to model both local acoustic patterns and temporal transitions across the ordered feature vector.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-[#0a0d14] border border-[#181e2e] space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">2. Devesh Model (Deep 1D CNN Architecture)</span>
                  <span className="font-mono text-[10px] text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">95.80% Accuracy</span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  Utilizes deep stacked 1D convolutional layers with Batch Normalization and Dropout regularization to isolate high-order acoustic spectral artifacts left by speech synthesis engines.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-[#0a0d14] border border-[#181e2e] space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">3. Swayam Model (CNN-Transformer with Positional Encoding)</span>
                  <span className="font-mono text-[10px] text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">97.40% Accuracy</span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  Applies custom Sinusoidal Positional Encoding and Multi-Head Self-Attention to capture cross-feature interactions across all 26 acoustic descriptors simultaneously.
                </p>
              </div>
            </div>
          </div>

          {/* Dataset & Majority Voting */}
          <div className="p-3.5 rounded-lg bg-[#141926] border border-[#222a3d] space-y-2">
            <h5 className="font-semibold text-white text-xs">Dataset & Consensus Principle</h5>
            <p className="text-slate-300 text-[11px]">
              Trained on a balanced dataset of 11,778 samples (5,889 Real vs 5,889 Fake balanced via SMOTE). Majority voting guarantees robustness by eliminating single-model bias and false positives.
            </p>
          </div>

        </div>

        {/* Footer */}
        <div className="pt-3 border-t border-[#1c2233] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-md bg-[#161c2b] hover:bg-[#1f273b] border border-[#242e47] text-xs font-medium text-slate-200 transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
}

