import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import AudioUploadCard from './components/AudioUploadCard';
import SampleAudios from './components/SampleAudios';
import AnalyzingState from './components/AnalyzingState';
import ResultsDashboard from './components/ResultsDashboard';
import HowItWorks from './components/HowItWorks';
import ArchitectureModal from './components/ArchitectureModal';
import Footer from './components/Footer';
import { AlertCircle, Loader2, RefreshCw, CheckCircle2, AlertTriangle } from 'lucide-react';

import {
  fetchSamplesList,
  getSampleAudioStreamUrl,
  submitPrediction,
  checkBackendHealth,
  checkBackendReady
} from './config/api';

export default function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioName, setAudioName] = useState('');
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [samples, setSamples] = useState([]);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);
  const [architectureOpen, setArchitectureOpen] = useState(false);

  // Backend readiness state: 'checking' | 'ready' | 'error'
  const [backendStatus, setBackendStatus] = useState('checking');
  const [retryTrigger, setRetryTrigger] = useState(0);

  // Polling management refs
  const pollTimerRef = useRef(null);
  const isMountedRef = useRef(true);

  // 1. Background Readiness Polling
  useEffect(() => {
    isMountedRef.current = true;
    setBackendStatus('checking');
    setError(null);

    const STARTUP_TIMEOUT_MS = 90000; // 90 seconds max wait for Render Free cold start
    const POLL_INTERVAL_MS = 3000;    // Poll every 3 seconds
    const startTime = Date.now();

    const checkStatus = async () => {
      if (!isMountedRef.current) return;

      try {
        // Probe health first, then readiness
        await checkBackendHealth();
        const readyData = await checkBackendReady();

        if (readyData && isMountedRef.current) {
          setBackendStatus('ready');
          // Load demo samples once backend is responsive
          fetchSamples();
          return; // Stop polling immediately on success
        }
      } catch (err) {
        // Backend still starting up or waking from sleep
        if (!isMountedRef.current) return;

        const elapsed = Date.now() - startTime;
        if (elapsed >= STARTUP_TIMEOUT_MS) {
          // Reached timeout threshold without successful connection
          setBackendStatus('error');
          return;
        }

        // Schedule next poll in 3 seconds
        pollTimerRef.current = setTimeout(checkStatus, POLL_INTERVAL_MS);
      }
    };

    // Begin first check immediately
    checkStatus();

    return () => {
      isMountedRef.current = false;
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
      }
    };
  }, [retryTrigger]);

  const fetchSamples = async () => {
    try {
      const data = await fetchSamplesList();
      if (isMountedRef.current) {
        setSamples(data.samples || []);
      }
    } catch (err) {
      console.warn("Could not fetch samples:", err.message);
    }
  };

  const handleRetryStartup = () => {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
    }
    setRetryTrigger((prev) => prev + 1);
  };

  // Upload Handlers
  const handleFileUpload = (file) => {
    if (!file) return;
    setError(null);
    setPredictionResult(null);
    setSelectedSampleId(null);
    setAudioFile(file);
    setAudioName(file.name);
    setAudioUrl(URL.createObjectURL(file));
  };

  const handleClearFile = () => {
    setAudioFile(null);
    setAudioUrl(null);
    setAudioName('');
    setSelectedSampleId(null);
    setError(null);
    setPredictionResult(null);
  };

  // Sample Selection Handlers
  const handleSelectSample = (sample) => {
    setError(null);
    setPredictionResult(null);
    setSelectedSampleId(sample.id);
    setAudioFile(null);
    setAudioName(sample.name);
    setAudioUrl(getSampleAudioStreamUrl(sample.id));
  };

  const handleAnalyzeSample = async (sample) => {
    if (backendStatus !== 'ready') {
      setError("Inference server is still starting up. Please wait until the detector is ready.");
      return;
    }
    handleSelectSample(sample);
    await triggerInference(null, sample.id, sample.name);
  };

  // Inference Execution
  const triggerInference = async (fileObj, sampleId, fileName) => {
    if (backendStatus !== 'ready') {
      setError("Inference server is not ready yet. Please wait a moment.");
      return;
    }
    if (isAnalyzing) return; // Prevent duplicate requests

    setIsAnalyzing(true);
    setError(null);
    setPredictionResult(null);

    try {
      const data = await submitPrediction(fileObj, sampleId);
      setPredictionResult(data);
    } catch (err) {
      console.error("Analysis error:", err);
      setError(err.message || "Failed to analyze audio. Please check the backend connection.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyze = async () => {
    if (backendStatus !== 'ready') {
      setError("Inference server is still starting up. Please wait until the detector is ready.");
      return;
    }
    if (!audioFile && !selectedSampleId) {
      setError("Please select or upload an audio file first.");
      return;
    }
    await triggerInference(audioFile, selectedSampleId, audioName);
  };

  const handleScrollToHowItWorks = () => {
    const el = document.getElementById('how-it-works');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-[#090a0f] text-slate-200 flex flex-col justify-between selection:bg-blue-600 selection:text-white">
      
      {/* 1. Header */}
      <Header
        onOpenArchitecture={() => setArchitectureOpen(true)}
        onScrollToHowItWorks={handleScrollToHowItWorks}
      />

      {/* 2. Main Content */}
      <main className="max-w-5xl w-full mx-auto px-4 sm:px-6 py-8 flex-1 space-y-8">
        
        {/* If no result is showing, display Hero & Upload interface */}
        {!predictionResult && !isAnalyzing && (
          <>
            <Hero />

            {/* Backend Startup Readiness Banner (Render Free Cold-Start Notice) */}
            {backendStatus === 'checking' && (
              <div className="max-w-3xl mx-auto p-4 rounded-xl bg-[#0f1422] border border-blue-900/40 shadow-sm flex items-center justify-between space-x-4 animate-in fade-in duration-300">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-lg bg-blue-950/60 border border-blue-800/50 flex items-center justify-center text-blue-400 shrink-0">
                    <Loader2 className="w-4 h-4 animate-spin" />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-white">Preparing AI Detector...</h4>
                    <p className="text-[11px] text-slate-400">
                      Waking up inference server on Render Free. This may take up to a minute.
                    </p>
                  </div>
                </div>
                <span className="text-[10px] font-mono text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded shrink-0">
                  Connecting...
                </span>
              </div>
            )}

            {/* Backend Startup Error / Timeout Banner */}
            {backendStatus === 'error' && (
              <div className="max-w-3xl mx-auto p-4 rounded-xl bg-amber-950/30 border border-amber-800/50 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 animate-in fade-in duration-300">
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 rounded-lg bg-amber-900/40 border border-amber-700/50 flex items-center justify-center text-amber-400 shrink-0 mt-0.5 sm:mt-0">
                    <AlertTriangle className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-amber-200">Inference server is taking longer than expected</h4>
                    <p className="text-[11px] text-slate-300">
                      The backend server could not be reached. Render Free may still be starting the container.
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleRetryStartup}
                  className="px-3.5 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-semibold flex items-center space-x-1.5 transition-colors shrink-0 shadow-sm"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Retry Connection</span>
                </button>
              </div>
            )}

            {/* Prediction Error Message */}
            {error && (
              <div className="p-3.5 rounded-lg bg-rose-950/40 border border-rose-900/60 flex items-start space-x-2.5 text-xs text-rose-300 max-w-3xl mx-auto">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Upload / Audio Card */}
            <div className="max-w-3xl mx-auto space-y-6">
              <AudioUploadCard
                audioFile={audioFile}
                audioName={audioName}
                audioUrl={audioUrl}
                onFileUpload={handleFileUpload}
                onClearFile={handleClearFile}
                onAnalyze={handleAnalyze}
                isAnalyzing={isAnalyzing || backendStatus === 'checking'}
              />

              {/* Sample Audios */}
              <SampleAudios
                samples={samples}
                selectedSampleId={selectedSampleId}
                onSelectSample={handleSelectSample}
                onAnalyzeSample={handleAnalyzeSample}
              />
            </div>

            {/* How It Works Pipeline */}
            <div className="pt-4 border-t border-[#161b28]">
              <HowItWorks />
            </div>
          </>
        )}

        {/* Loading / Analyzing State */}
        {isAnalyzing && (
          <AnalyzingState audioName={audioName} />
        )}

        {/* 3. Results Dashboard */}
        {predictionResult && !isAnalyzing && (
          <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-300">
            <ResultsDashboard
              result={predictionResult}
              audioUrl={audioUrl}
              audioName={audioName}
              onReset={handleClearFile}
            />

            {/* How It Works Pipeline under results */}
            <div className="pt-6 border-t border-[#161b28]">
              <HowItWorks />
            </div>
          </div>
        )}

      </main>

      {/* Architecture Modal */}
      <ArchitectureModal
        isOpen={architectureOpen}
        onClose={() => setArchitectureOpen(false)}
      />

      {/* 4. Footer */}
      <Footer />

    </div>
  );
}
