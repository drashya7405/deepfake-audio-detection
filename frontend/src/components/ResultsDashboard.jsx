import React, { useState, useRef, useEffect } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  RotateCcw,
  Play,
  Pause,
  Layers,
  BarChart3,
  Download,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Cpu,
  FileAudio,
  Activity,
  ArrowLeft
} from 'lucide-react';

export default function ResultsDashboard({
  result,
  audioUrl,
  audioName,
  onReset
}) {
  const [activeFeatureTab, setActiveFeatureTab] = useState('spectral'); // 'spectral' | 'mfcc' | 'json'
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);

  const isFake = result.final_decision === 'FAKE';
  const confidencePct = result.majority_vote?.ensemble_confidence_pct ?? 0;
  const avgRealProb = result.majority_vote?.avg_real_probability ?? 0.5;
  const realPct = (avgRealProb * 100).toFixed(1);
  const fakePct = ((1 - avgRealProb) * 100).toFixed(1);

  // Audio Playback Controls
  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      if (!isNaN(audioRef.current.duration) && audioRef.current.duration > 0) {
        setDuration(audioRef.current.duration);
      }
    }
  };

  const handleSeek = (e) => {
    const time = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const formatTime = (secs) => {
    if (isNaN(secs) || secs === 0) return "00:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // Canvas waveform visualizer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const width = canvas.width;
      const height = canvas.height;
      const barCount = 48;
      const barWidth = width / barCount - 2;

      for (let i = 0; i < barCount; i++) {
        const progress = duration > 0 ? currentTime / duration : 0;
        const currentBarIndex = Math.floor(progress * barCount);
        const isPassed = i <= currentBarIndex;

        const baseHeight = (Math.sin(i * 0.4) * 0.35 + Math.cos(i * 0.7) * 0.25 + 0.4);
        const dynamicFactor = isPlaying ? Math.abs(Math.sin(currentTime * 5 + i * 0.4)) * 0.25 : 0;
        const barHeight = Math.max(3, (baseHeight + dynamicFactor) * (height - 8));

        const x = i * (barWidth + 2);
        const y = (height - barHeight) / 2;

        ctx.fillStyle = isPassed 
          ? (isFake ? '#f43f5e' : '#10b981')
          : '#1e2638';
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 1.5);
        ctx.fill();
      }

      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isPlaying, currentTime, duration, isFake]);

  const handleDownloadJSON = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensic_report_${audioName || 'audio'}_${Date.now()}.json`;
    a.click();
  };

  return (
    <div className="space-y-6">

      {/* Top Header Navigation */}
      <div className="flex items-center justify-between pb-2">
        <div className="flex items-center space-x-2 truncate">
          <span className="text-xs uppercase font-mono tracking-wider text-slate-400">Analysis Result</span>
          <span className="text-slate-400">·</span>
          <span className="text-xs font-semibold text-white truncate max-w-xs sm:max-w-md">{audioName}</span>
        </div>

        <button
          onClick={onReset}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-[#141926] hover:bg-[#1a2133] border border-[#222a3d] text-xs font-medium text-slate-200 transition-colors shrink-0"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Analyze Another File</span>
        </button>
      </div>

      {/* 1. Large High-Impact Result Card */}
      <div className={`p-6 sm:p-7 rounded-xl border ${
        isFake
          ? 'bg-[#150f14] border-rose-900/50'
          : 'bg-[#0f1715] border-emerald-900/50'
      }`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          
          {/* Left: Verdict Status */}
          <div className="flex items-start sm:items-center space-x-4">
            <div className={`w-14 h-14 rounded-xl flex items-center justify-center shrink-0 border ${
              isFake
                ? 'bg-rose-950/60 border-rose-800/60 text-rose-400'
                : 'bg-emerald-950/60 border-emerald-800/60 text-emerald-400'
            }`}>
              {isFake ? <ShieldAlert className="w-7 h-7" /> : <ShieldCheck className="w-7 h-7" />}
            </div>

            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className={`text-[11px] font-mono uppercase font-bold px-2 py-0.5 rounded ${
                  isFake ? 'bg-rose-500 text-white' : 'bg-emerald-500 text-white'
                }`}>
                  {isFake ? 'AI GENERATED' : 'LIKELY HUMAN'}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  {confidencePct >= 90 ? 'High Confidence' : 'Moderate Confidence'}
                </span>
              </div>

              <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
                {isFake ? 'Synthetic Deepfake Audio' : 'Authentic Human Speech'}
              </h2>

              <p className="text-xs text-slate-400 leading-relaxed max-w-xl">
                {isFake
                  ? 'Acoustic spectral distribution and cepstral timbre patterns indicate artificial voice generation or voice cloning.'
                  : 'Harmonic resonance and micro-variations in spectral energy are consistent with natural unmanipulated human speech.'}
              </p>
            </div>
          </div>

          {/* Right: Confidence Score */}
          <div className="bg-[#0b0e14]/80 border border-[#1e2538] rounded-lg p-4 text-left md:text-right shrink-0 min-w-[170px]">
            <p className="text-[11px] font-mono uppercase text-slate-400">Ensemble Confidence</p>
            <p className={`text-3xl font-extrabold font-mono mt-0.5 ${
              isFake ? 'text-rose-400' : 'text-emerald-400'
            }`}>
              {confidencePct}%
            </p>
            <p className="text-[10px] text-slate-400 font-mono mt-1">
              Inference: {result.processing_time_ms} ms
            </p>
          </div>
        </div>

        {/* Probability Visualization Comparison Bar */}
        <div className="mt-6 pt-5 border-t border-[#1e2538] space-y-2">
          <div className="flex justify-between text-xs font-mono">
            <span className={isFake ? 'text-rose-400 font-semibold' : 'text-slate-400'}>
              AI Generated: {fakePct}%
            </span>
            <span className={!isFake ? 'text-emerald-400 font-semibold' : 'text-slate-400'}>
              Human: {realPct}%
            </span>
          </div>

          <div className="h-2 w-full bg-[#121622] rounded-full overflow-hidden flex">
            <div
              className="bg-rose-500 h-full transition-all duration-700"
              style={{ width: `${fakePct}%` }}
              title={`AI Generated: ${fakePct}%`}
            />
            <div
              className="bg-emerald-500 h-full transition-all duration-700"
              style={{ width: `${realPct}%` }}
              title={`Human: ${realPct}%`}
            />
          </div>
        </div>

      </div>

      {/* 2. Embedded Audio Replay Deck */}
      {audioUrl && (
        <div className="bg-[#0f121a] border border-[#1c2233] rounded-xl p-4 sm:p-5 space-y-3 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2.5 truncate">
              <FileAudio className="w-4 h-4 text-blue-400 shrink-0" />
              <span className="text-xs font-medium text-slate-200 truncate">{audioName}</span>
              <span className="text-[11px] font-mono text-slate-400 hidden sm:inline">
                ({result.audio_info?.duration_seconds}s @ {result.audio_info?.sample_rate} Hz)
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  if (audioRef.current) {
                    audioRef.current.currentTime = 0;
                    setCurrentTime(0);
                  }
                }}
                className="p-1.5 rounded bg-[#141926] hover:bg-[#1a2133] border border-[#20283d] text-slate-400 hover:text-slate-200"
                title="Restart"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>

              <button
                onClick={togglePlay}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                <span>{isPlaying ? 'Pause' : 'Play'}</span>
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="flex-1 h-8 rounded bg-[#0a0d14] border border-[#181e2e] px-2 flex items-center overflow-hidden">
              <canvas ref={canvasRef} width={600} height={32} className="w-full h-full" />
            </div>

            <div className="text-right shrink-0 text-xs font-mono text-slate-400">
              {formatTime(currentTime)} <span className="text-slate-600">/</span> {formatTime(duration || result.audio_info?.duration_seconds)}
            </div>
          </div>

          <input
            type="range"
            min="0"
            max={duration || result.audio_info?.duration_seconds || 100}
            step="0.01"
            value={currentTime}
            onChange={handleSeek}
            className="w-full"
          />

          <audio
            ref={audioRef}
            src={audioUrl}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleTimeUpdate}
            onEnded={() => setIsPlaying(false)}
            className="hidden"
          />
        </div>
      )}

      {/* 3. Three-Model Ensemble Section */}
      <div className="bg-[#0f121a] border border-[#1c2233] rounded-xl p-5 sm:p-6 space-y-5 shadow-sm">
        
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[#1c2233]">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
              <Layers className="w-4 h-4 text-blue-400" />
              <span>Model Predictions</span>
            </h3>
            <p className="text-[11px] text-slate-400">Independent classification from each neural architecture</p>
          </div>

          <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-[#141926] border border-[#20283d] text-xs font-mono text-slate-300">
            <span className="font-semibold text-white">Ensemble Decision:</span>
            <span className={`font-bold ${isFake ? 'text-rose-400' : 'text-emerald-400'}`}>
              {result.majority_vote?.decision}
            </span>
            <span className="text-slate-400">({result.majority_vote?.agreement} models)</span>
          </div>
        </div>

        {/* 3 Model Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(result.models || {}).map(([key, model]) => {
            const mIsFake = model.prediction === 'FAKE';
            return (
              <div
                key={key}
                className={`p-4 rounded-lg border space-y-3 ${
                  mIsFake
                    ? 'bg-[#120e13] border-rose-900/40'
                    : 'bg-[#0d1412] border-emerald-900/40'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-slate-400">
                      Model {key === 'drashya' ? '01' : key === 'devesh' ? '02' : '03'}
                    </span>
                    <h4 className="text-xs font-semibold text-white mt-0.5">
                      {model.name.replace(/\(.*\)/, '').trim()}
                    </h4>
                  </div>

                  <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                    mIsFake
                      ? 'bg-rose-950 text-rose-300 border-rose-800/50'
                      : 'bg-emerald-950 text-emerald-300 border-emerald-800/50'
                  }`}>
                    {model.prediction}
                  </span>
                </div>

                <p className="text-[11px] text-slate-400 leading-normal min-h-[30px]">
                  {model.architecture}
                </p>

                {/* Confidence Bar */}
                <div className="space-y-1 pt-1 border-t border-[#1c2233]">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-slate-400">Confidence:</span>
                    <span className={`font-semibold ${mIsFake ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {model.confidence_pct}%
                    </span>
                  </div>

                  <div className="h-1.5 w-full bg-[#161c2b] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${mIsFake ? 'bg-rose-500' : 'bg-emerald-500'}`}
                      style={{ width: `${model.confidence_pct}%` }}
                    />
                  </div>
                </div>

                <div className="flex justify-between text-[10px] font-mono text-slate-400 pt-0.5">
                  <span>P(Real): {model.real_probability_pct}%</span>
                  <span>P(Fake): {model.fake_probability_pct}%</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Ensemble Consensus Explainer */}
        <div className="p-3 rounded-lg bg-[#0a0d14] border border-[#181e2e] flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-blue-400 shrink-0" />
            <span>Majority Voting Logic: &ge;2 out of 3 models determine the consensus verdict.</span>
          </div>
          <span className="font-mono text-[11px] text-slate-300">
            {result.majority_vote?.fake_votes} Fake vs {result.majority_vote?.real_votes} Real
          </span>
        </div>

      </div>

      {/* 4. Acoustic Feature Analysis (26 Features) */}
      <div className="bg-[#0f121a] border border-[#1c2233] rounded-xl p-5 sm:p-6 space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1c2233] pb-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-blue-400" />
              <span>Acoustic Feature Analysis</span>
            </h3>
            <p className="text-[11px] text-slate-400">26 handcrafted acoustic features extracted via Librosa</p>
          </div>

          <div className="flex rounded-md bg-[#0a0d14] p-1 border border-[#1c2233] text-xs">
            <button
              onClick={() => setActiveFeatureTab('spectral')}
              className={`px-3 py-1 rounded font-medium transition-colors ${
                activeFeatureTab === 'spectral' ? 'bg-[#1c2336] text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Spectral & Energy (6)
            </button>
            <button
              onClick={() => setActiveFeatureTab('mfcc')}
              className={`px-3 py-1 rounded font-medium transition-colors ${
                activeFeatureTab === 'mfcc' ? 'bg-[#1c2336] text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              MFCC 1-20 (20)
            </button>
            <button
              onClick={() => setActiveFeatureTab('json')}
              className={`px-3 py-1 rounded font-medium transition-colors ${
                activeFeatureTab === 'json' ? 'bg-[#1c2336] text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Report JSON
            </button>
          </div>
        </div>

        {/* Tab 1: Spectral & Energy */}
        {activeFeatureTab === 'spectral' && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[
              { key: 'chroma_stft', name: 'Chroma STFT', desc: '12-bin pitch class energy distribution across audio spectrum' },
              { key: 'rms', name: 'RMS Energy', desc: 'Root-mean-square amplitude representing signal power' },
              { key: 'spectral_centroid', name: 'Spectral Centroid', desc: 'Center frequency representing brightness of speech sound' },
              { key: 'spectral_bandwidth', name: 'Spectral Bandwidth', desc: 'Spread of frequencies around spectral centroid' },
              { key: 'rolloff', name: 'Spectral Rolloff', desc: 'Frequency below which 85% of total spectral energy lies' },
              { key: 'zero_crossing_rate', name: 'Zero Crossing Rate', desc: 'Frequency of waveform sign changes indicating noisiness' },
            ].map((item) => (
              <div key={item.key} className="bg-[#0a0d14] border border-[#181e2e] p-3.5 rounded-lg space-y-1">
                <p className="text-[10px] font-mono font-semibold text-blue-400 uppercase tracking-wider">{item.name}</p>
                <p className="text-base font-mono font-bold text-white">
                  {result.features?.[item.key] !== undefined ? result.features[item.key] : 'N/A'}
                </p>
                <p className="text-[10px] text-slate-400 leading-tight">{item.desc}</p>
              </div>
            ))}
          </div>
        )}

        {/* Tab 2: 20 MFCCs */}
        {activeFeatureTab === 'mfcc' && (
          <div className="space-y-3">
            <p className="text-[11px] text-slate-400">
              Mel-Frequency Cepstral Coefficients (MFCCs) capture the human vocal tract filter envelope and timbre characteristics.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-2.5">
              {Array.from({ length: 20 }, (_, i) => `mfcc${i + 1}`).map((k) => (
                <div key={k} className="bg-[#0a0d14] border border-[#181e2e] p-2.5 rounded-lg text-center">
                  <span className="text-[10px] font-mono text-slate-400 font-bold uppercase">{k}</span>
                  <p className="text-xs font-mono font-medium text-slate-200 mt-0.5">
                    {result.features?.[k] !== undefined ? result.features[k] : 0}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Raw JSON */}
        {activeFeatureTab === 'json' && (
          <div className="relative">
            <pre className="bg-[#0a0d14] border border-[#181e2e] p-4 rounded-lg text-[11px] font-mono text-slate-300 overflow-x-auto max-h-72">
              {JSON.stringify(result, null, 2)}
            </pre>
            <button
              onClick={handleDownloadJSON}
              className="absolute top-3 right-3 px-2.5 py-1 rounded bg-[#161c2b] hover:bg-[#1f273b] border border-[#242e47] text-[10px] font-medium text-slate-200 flex items-center space-x-1 transition-colors"
            >
              <Download className="w-3 h-3" />
              <span>Download JSON</span>
            </button>
          </div>
        )}
      </div>

    </div>
  );
}

