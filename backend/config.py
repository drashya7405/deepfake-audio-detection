"""
Deepfake Audio Detection - Backend Configuration
Centralizes all environment variables, feature column definitions, and file paths.
Optimized for 512 MB memory-constrained hosting environments (Render Free).
"""

import os
import logging

# --- Path Configurations ---
# Project root directory (one level up from backend)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "BestModels")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FALLBACK_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
DATASET_PATH = os.path.join(PROJECT_ROOT, "DATASET-balanced.csv")
SAMPLE_AUDIOS_DIR = PROJECT_ROOT

# Low-memory environment flags for 512 MB hosts (Render Free)
# Disables oneDNN huge memory arena pre-allocations and forces single-thread CPU execution
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Ensure KERAS_HOME stays within the project to prevent sandbox/permission errors
KERAS_HOME = os.environ.get("KERAS_HOME", os.path.join(PROJECT_ROOT, ".keras"))
os.environ["KERAS_HOME"] = KERAS_HOME
os.makedirs(KERAS_HOME, exist_ok=True)

# --- Server & Environment Configurations ---
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

# CORS: Configurable frontend origin(s)
FRONTEND_URL_RAW = os.environ.get("FRONTEND_URL", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,https://deepfake-audio-detection-rust.vercel.app")
ALLOWED_ORIGINS = [url.strip() for url in FRONTEND_URL_RAW.split(",") if url.strip()]

# Logging level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Upload limits
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 25))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = float(os.environ.get("MAX_AUDIO_DURATION_SECONDS", 60.0))

# Rate limiting
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 15))

# --- Supported Audio Extensions ---
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".wma", ".opus"
}

# --- Centralized Exact 26 Feature Definitions ---
# The models strictly require these exact 26 acoustic features in this order
FEATURE_COLUMNS = [
    'chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
    'rolloff', 'zero_crossing_rate',
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8',
    'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
    'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
]

# --- Model Metadata ---
MODEL_CONFIGS = [
    {
        "id": "drashya",
        "name": "Drashya Model (CNN-RNN Hybrid)",
        "filename": "drashya_best_deepfake_audio_model.h5",
        "tflite_filename": "drashya_best_deepfake_audio_model.tflite",
        "architecture": "1D CNN + Recurrent Neural Network (LSTM/GRU)",
        "test_split_accuracy": "98.59% (on project test split)"
    },
    {
        "id": "devesh",
        "name": "Devesh Model (Deep CNN Architecture)",
        "filename": "devesh_best_deepfake_audio_model.h5",
        "tflite_filename": "devesh_best_deepfake_audio_model.tflite",
        "architecture": "Deep 1D CNN with Batch Normalization & Dropout",
        "test_split_accuracy": "95.80% (on project test split)"
    },
    {
        "id": "swayam",
        "name": "Swayam Model (CNN-Transformer)",
        "filename": "swayam_best_deepfake_audio_model.h5",
        "tflite_filename": "swayam_best_deepfake_audio_model.tflite",
        "architecture": "1D CNN + Multi-Head Self-Attention with Positional Encoding",
        "test_split_accuracy": "97.40% (on project test split)"
    }
]

# --- Pre-packaged Sample Audio Files ---
KNOWN_SAMPLES = [
    {
        "id": "sample_real_1",
        "name": "gehra_hua.mp3",
        "path": "gehra_hua.mp3",
        "tag": "Natural Speech",
        "type": "MP3 Audio"
    },
    {
        "id": "sample_fake_1",
        "name": "file1004.mp3",
        "path": "FAKE_AUDIOS/file1004.mp3",
        "tag": "AI Synthetic Voice",
        "type": "MP3 Audio"
    },
    {
        "id": "sample_fake_2",
        "name": "file1004.wav",
        "path": "FAKE_AUDIOS/file1004.wav",
        "tag": "AI Synthetic Voice",
        "type": "WAV Audio"
    },
    {
        "id": "sample_fake_3",
        "name": "file1019.wav",
        "path": "FAKE_AUDIOS/file1019.wav",
        "tag": "AI Synthetic Voice",
        "type": "WAV Audio"
    }
]
