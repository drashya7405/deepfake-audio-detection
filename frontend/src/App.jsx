import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import AudioUploadCard from './components/AudioUploadCard';
import SampleAudios from './components/SampleAudios';
import AnalyzingState from './components/AnalyzingState';
import ResultsDashboard from './components/ResultsDashboard';
import HowItWorks from './components/HowItWorks';
import ArchitectureModal from './components/ArchitectureModal';
import Footer from './components/Footer';
import { AlertCircle } from 'lucide-react';

import { fetchSamplesList, getSampleAudioStreamUrl, submitPrediction } from './config/api';

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

  // Fetch available demo samples
  useEffect(() => {
    fetchSamples();
  }, []);

  const fetchSamples = async () => {
    try {
      const data = await fetchSamplesList();
      setSamples(data.samples || []);
    } catch (err) {
      console.warn("Could not fetch samples:", err.message);
    }
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
    handleSelectSample(sample);
    await triggerInference(null, sample.id, sample.name);
  };

  // Inference Execution
  const triggerInference = async (fileObj, sampleId, fileName) => {
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

            {/* Error Message */}
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
                isAnalyzing={isAnalyzing}
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
