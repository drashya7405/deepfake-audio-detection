import React from 'react';
import { FileAudio, Play, ShieldAlert, ShieldCheck, ArrowRight } from 'lucide-react';

export default function SampleAudios({ samples, selectedSampleId, onSelectSample, onAnalyzeSample }) {
  if (!samples || samples.length === 0) return null;

  const sampleDescriptions = {
    sample_real_1: { title: "Sample 01 · Natural Speech", desc: "Authentic human voice recording (Hindi vocal excerpt)", duration: "00:30" },
    sample_fake_1: { title: "Sample 02 · Voice Clone 01", desc: "AI speech synthesized audio sample", duration: "00:04" },
    sample_fake_2: { title: "Sample 03 · AI Synthesis (WAV)", desc: "Deepfake cloned voice sample in uncompressed PCM format", duration: "00:04" },
    sample_fake_3: { title: "Sample 04 · Voice Clone 02", desc: "AI speech generation with acoustic artifacts", duration: "00:02" }
  };

  return (
    <section className="space-y-3 pt-2">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-white">Try a Sample</h3>
          <p className="text-[11px] text-slate-400">Benchmark instant detection on pre-packaged audio clips</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {samples.map((s) => {
          const isSelected = selectedSampleId === s.id;
          const meta = sampleDescriptions[s.id] || { title: s.name, desc: s.type, duration: "Audio" };
          const isFake = s.tag.includes('Synthetic');

          return (
            <div
              key={s.id}
              onClick={() => onSelectSample(s)}
              className={`group p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-3 ${
                isSelected
                  ? 'bg-[#141926] border-blue-500/80 shadow-sm'
                  : 'bg-[#0f121a] hover:bg-[#131722] border-[#1c2233] hover:border-[#28324a]'
              }`}
            >
              {/* Card Top */}
              <div className="flex items-start justify-between">
                <div className="w-8 h-8 rounded-lg bg-[#161c2b] border border-[#222a3d] flex items-center justify-center text-blue-400 group-hover:text-blue-300">
                  <FileAudio className="w-4 h-4" />
                </div>

                <span className={`text-[10px] font-mono font-medium px-2 py-0.5 rounded border ${
                  isFake
                    ? 'bg-rose-950/40 text-rose-300 border-rose-800/40'
                    : 'bg-emerald-950/40 text-emerald-300 border-emerald-800/40'
                }`}>
                  {isFake ? 'Synthetic' : 'Human'}
                </span>
              </div>

              {/* Title & Desc */}
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-white group-hover:text-blue-300 transition-colors">
                  {meta.title}
                </h4>
                <p className="text-[11px] text-slate-400 line-clamp-2 leading-tight">
                  {meta.desc}
                </p>
              </div>

              {/* Card Bottom Meta & Button */}
              <div className="pt-2 border-t border-[#1a2030] flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span>{meta.duration} · {s.size_kb} KB</span>
                
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onAnalyzeSample(s);
                  }}
                  className="inline-flex items-center space-x-1 text-[11px] font-semibold text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <span>Analyze</span>
                  <ArrowRight className="w-3 h-3" />
                </button>
              </div>

            </div>
          );
        })}
      </div>
    </section>
  );
}

