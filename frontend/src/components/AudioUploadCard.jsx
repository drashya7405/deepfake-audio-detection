import React, { useState, useRef, useEffect } from 'react';
import {
  Upload,
  Play,
  Pause,
  RotateCcw,
  FileAudio,
  RefreshCw,
  ArrowRight,
  Sliders,
  Volume2
} from 'lucide-react';

export default function AudioUploadCard({
  audioFile,
  audioName,
  audioUrl,
  onFileUpload,
  onClearFile,
  onAnalyze,
  isAnalyzing
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  // Audio Playback
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

  // Waveform Canvas Rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const width = canvas.width;
      const height = canvas.height;
      const barCount = 42;
      const barWidth = width / barCount - 2;

      for (let i = 0; i < barCount; i++) {
        // Compute amplitude pattern
        const progress = duration > 0 ? currentTime / duration : 0;
        const currentBarIndex = Math.floor(progress * barCount);
        const isPassed = i <= currentBarIndex;

        // Realistic acoustic waveform pattern
        const baseHeight = (Math.sin(i * 0.4) * 0.35 + Math.cos(i * 0.8) * 0.25 + 0.4);
        const dynamicFactor = isPlaying ? Math.abs(Math.sin(currentTime * 4 + i * 0.5)) * 0.3 : 0;
        const barHeight = Math.max(4, (baseHeight + dynamicFactor) * (height - 12));

        const x = i * (barWidth + 2);
        const y = (height - barHeight) / 2;

        ctx.fillStyle = isPassed ? '#60a5fa' : '#222a3d';
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
  }, [isPlaying, currentTime, duration]);

  // If no audio is loaded, show clean upload dropzone
  if (!audioUrl) {
    return (
      <div className="bg-[#0f121a] border border-[#1c2233] rounded-xl p-5 sm:p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-7 h-7 rounded-md bg-[#161c2b] border border-[#242e47] flex items-center justify-center text-blue-400">
              <FileAudio className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Analyze an Audio File</h3>
              <p className="text-[11px] text-slate-400">Upload voice recording or sound clip for forensic evaluation</p>
            </div>
          </div>
        </div>

        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById('audio-file-input').click()}
          className={`border-2 border-dashed rounded-lg p-7 sm:p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragOver
              ? 'border-blue-500 bg-blue-500/5'
              : 'border-[#222a3d] hover:border-[#354363] bg-[#0a0d14]/70 hover:bg-[#0d101a]'
          }`}
        >
          <input
            id="audio-file-input"
            type="file"
            accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.aac,.webm"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                onFileUpload(e.target.files[0]);
              }
            }}
          />

          <div className="w-11 h-11 mx-auto mb-3 rounded-lg bg-[#141926] border border-[#222a3d] flex items-center justify-center text-blue-400">
            <Upload className="w-5 h-5" />
          </div>

          <p className="text-sm font-medium text-slate-200">
            Drag & drop your audio here <span className="text-slate-500 font-normal">or</span>{' '}
            <span className="text-blue-400 font-semibold underline underline-offset-2">Browse Files</span>
          </p>

          <p className="text-[11px] font-mono text-slate-500 mt-2">
            MP3 · WAV · M4A · OGG · FLAC · AAC · WEBM
          </p>
        </div>
      </div>
    );
  }

  // File Selected Preview State
  return (
    <div className="bg-[#0f121a] border border-[#1c2233] rounded-xl p-5 sm:p-6 shadow-sm space-y-5">
      
      {/* File Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#1c2233]">
        <div className="flex items-center space-x-3 truncate">
          <div className="w-9 h-9 rounded-lg bg-[#141926] border border-[#222a3d] flex items-center justify-center text-blue-400 shrink-0">
            <FileAudio className="w-4 h-4" />
          </div>
          <div className="truncate">
            <p className="text-sm font-semibold text-white truncate">{audioName}</p>
            <div className="flex items-center space-x-2 text-[11px] font-mono text-slate-400 mt-0.5">
              <span>{audioFile ? `${(audioFile.size / 1024).toFixed(1)} KB` : 'Demo Sample'}</span>
              <span>•</span>
              <span>{duration > 0 ? formatTime(duration) : 'Loading...'}</span>
              <span>•</span>
              <span className="uppercase">{audioName.split('.').pop() || 'AUDIO'}</span>
            </div>
          </div>
        </div>

        <button
          onClick={onClearFile}
          className="text-xs text-slate-400 hover:text-slate-200 px-2.5 py-1 rounded bg-[#141926] hover:bg-[#1b2234] border border-[#222a3d] self-start sm:self-auto transition-colors"
        >
          Replace File
        </button>
      </div>

      {/* Interactive Waveform & Player Deck */}
      <div className="bg-[#0a0d14] border border-[#181e2e] rounded-lg p-4 space-y-3">
        
        {/* Top Play/Pause and Seek */}
        <div className="flex items-center space-x-3">
          <button
            onClick={togglePlay}
            className="w-9 h-9 rounded-md bg-blue-600 hover:bg-blue-500 text-white flex items-center justify-center shrink-0 transition-colors shadow-sm"
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
          </button>

          <button
            onClick={() => {
              if (audioRef.current) {
                audioRef.current.currentTime = 0;
                setCurrentTime(0);
              }
            }}
            className="p-2 rounded bg-[#141926] hover:bg-[#1a2133] border border-[#20283d] text-slate-400 hover:text-slate-200"
            title="Restart"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Canvas Waveform */}
          <div className="flex-1 h-9 rounded bg-[#0d101a] border border-[#1c2233] px-2 flex items-center overflow-hidden">
            <canvas ref={canvasRef} width={600} height={36} className="w-full h-full" />
          </div>

          <div className="text-right shrink-0">
            <span className="text-xs font-mono font-medium text-slate-300">
              {formatTime(currentTime)} <span className="text-slate-600">/</span> {formatTime(duration)}
            </span>
          </div>
        </div>

        {/* Scrub Bar */}
        <input
          type="range"
          min="0"
          max={duration || 100}
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

      {/* Action CTA Button */}
      <div className="flex items-center justify-end space-x-3 pt-1">
        <button
          onClick={onAnalyze}
          disabled={isAnalyzing}
          className="w-full sm:w-auto px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-sm disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed"
        >
          <span>Analyze Audio</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

    </div>
  );
}

