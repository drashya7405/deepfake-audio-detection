# 🎙️ Deepfake Audio Detection System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF.svg?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade AI audio forensics and deepfake detection application. The system analyzes speech recordings (MP3, WAV, M4A, OGG, FLAC), extracts **26 acoustic features**, evaluates them across a **3-model deep learning ensemble** (CNN-RNN, Deep 1D CNN, CNN-Transformer), and generates an authenticity verdict through majority voting consensus.

---

## 🏗️ Production Architecture

```text
React/Vite Frontend (Vercel)
        │
        │ HTTPS REST API
        ▼
FastAPI Backend (Native Python Host)
        │
        ├── Rate Limiter & Validation (25 MB max, 60s max duration)
        │
        ├── Audio Decoding & Conversion (Librosa / SoundFile / PyDub)
        │
        ├── 26 Acoustic Feature Extraction
        │   ├── Chroma STFT (1)
        │   ├── RMS Energy (1)
        │   ├── Spectral Centroid (1)
        │   ├── Spectral Bandwidth (1)
        │   ├── Spectral Rolloff (1)
        │   ├── Zero Crossing Rate (1)
        │   └── MFCCs 1–20 (20)
        │
        ├── Feature Normalization (Pre-computed StandardScaler)
        │
        ├── 3-Model Deep Learning Evaluation
        │   ├── 1. Drashya Model (1D CNN + RNN Hybrid)
        │   ├── 2. Devesh Model (Deep 1D CNN)
        │   └── 3. Swayam Model (CNN-Transformer with Positional Encoding)
        │
        └── Ensemble Engine (Majority Voting Consensus ≥ 2/3)
                 │
                 ▼
          REAL / AI-GENERATED
```

---

## 📊 Evaluated Architectures & Test-Split Performance

The ensemble evaluates 3 complementary neural architectures trained on a balanced dataset of 11,778 audio samples (balanced via SMOTE):

| Model | Architecture Family | Evaluated Test-Split Accuracy* | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **Drashya Model** | CNN-RNN Hybrid | **98.59%** | Combines 1D convolutions with recurrent LSTM/GRU layers to capture spectral feature sequences. |
| **Devesh Model** | Deep 1D CNN | **95.80%** | Deep convolutional layers with Batch Normalization and Dropout to isolate synthesis artifacts. |
| **Swayam Model** | CNN-Transformer | **97.40%** | Multi-Head Self-Attention with Sinusoidal Positional Encoding for cross-feature interactions. |
| **Ensemble** | **Majority Voting (≥2/3)** | **100.0%** (on benchmark set) | Eliminates single-model bias and guarantees high-confidence detection. |

*\*Note: Accuracy figures represent performance on the project's evaluated test split and benchmark test files.*

---

## 📁 Repository Structure

```text
deepfake-audio-detection/
├── backend/                        # FastAPI Machine Learning Backend
│   ├── app.py                      # REST API endpoints & lifespan handler
│   ├── config.py                   # Centralized configuration & feature definitions
│   ├── requirements.txt            # Pinned Python dependencies
│   └── services/
│       ├── audio.py                # Audio conversion & 26-feature extraction
│       ├── preprocessing.py        # StandardScaler normalization & tensor reshaping
│       ├── inference.py            # Model loading & 3-model majority voting
│       └── rate_limiter.py         # In-memory sliding-window rate limiting
│
├── frontend/                       # Modern React + Vite Forensics Web App
│   ├── src/
│   │   ├── components/             # Modular UI components (Waveform, Results, etc.)
│   │   ├── config/api.js           # Centralized API network layer with timeouts
│   │   ├── App.jsx                 # Application state & routing
│   │   └── index.css               # Tailwind CSS & custom forensics theme
│   ├── package.json
│   └── .env.example
│
├── BestModels/                     # Pre-trained Keras HDF5 Models
│   ├── drashya_best_deepfake_audio_model.h5
│   ├── devesh_best_deepfake_audio_model.h5
│   └── swayam_best_deepfake_audio_model.h5
│
├── models/
│   └── scaler.pkl                  # Serialized StandardScaler artifact (26 dimensions)
│
├── tests/                          # Automated Pytest Suite
│   ├── test_backend_unit.py        # Unit tests for scaler, features, audio, rate limits
│   ├── test_api_endpoints.py       # REST API integration tests
│   └── test_ml_regression.py       # ML regression tests on known audio samples
│
├── app.py                          # Root production launcher (respects $PORT)
├── run.py                          # 1-command local development runner
├── requirements.txt                # Root Python dependencies
├── packages.txt                    # System apt packages (ffmpeg, libsndfile1)
└── .env.example                    # Environment variable template
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root or configure these variables in your hosting provider:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port for the FastAPI server (auto-injected on cloud hosts like Render/HF). |
| `HOST` | `0.0.0.0` | Host binding address. |
| `FRONTEND_URL` | `http://localhost:5173` | Allowed CORS origin(s), comma-separated. Set to your Vercel URL in production. |
| `VITE_API_URL` | `http://localhost:8000` | Frontend backend API URL (set in Vercel environment settings). |
| `MAX_UPLOAD_MB` | `25` | Maximum upload file size in megabytes. |
| `MAX_AUDIO_DURATION_SECONDS` | `60` | Maximum audio duration in seconds. |
| `RATE_LIMIT_PER_MINUTE` | `15` | Max prediction requests per minute per client IP. |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## 💻 Local Development Setup (Native Python — No Docker Required)

### 1. Prerequisites
- **Python 3.10 or 3.11**
- **Node.js 18+ & npm**
- **FFmpeg** (required for decoding audio files):
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu / Debian**: `sudo apt-get install -y ffmpeg libsndfile1`
  - **Windows**: Install via `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org).

### 2. Backend Setup
```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Start the FastAPI server
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`
- Readiness Check: `http://localhost:8000/api/ready`

### 3. Frontend Setup
```bash
# In a separate terminal:
cd frontend
npm install
npm run dev
```
- Web Application: `http://localhost:5173`

### 4. Or Run Both Together with 1 Command:
```bash
python run.py
```

---

## 🧪 Running Automated Tests

Run the complete test suite (unit tests, API tests, and ML regression checks):

```bash
# Activate virtual environment
source .venv/bin/activate

# Run pytest
pytest tests/ -v
```

---

## 🚀 Production Deployment (100% Free & Native Python)

### 1. Backend Deployment (Render or Hugging Face Spaces)

#### Option A: Render.com (Web Service)
1. Create a free account on [render.com](https://render.com).
2. Click **New +** $\rightarrow$ **Web Service** $\rightarrow$ Connect your GitHub repo.
3. Configure settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables:
   - `FRONTEND_URL`: `https://your-app-name.vercel.app`
5. Click **Create Web Service**. Copy your live backend URL (`https://your-service.onrender.com`).

#### Option B: Hugging Face Spaces (16 GB RAM Free Tier)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Space SDK**: `Gradio` *(runs standard Python natively)*.
3. Select **Space Hardware**: `CPU basic · 2 vCPU · 16 GB RAM · Free`.
4. Select **Visibility**: `Public`.
5. Push repository files to your Space (`git push`).
6. Copy your Space URL (`https://<username>-<space-name>.hf.space`).

---

### 2. Frontend Deployment (Vercel)

1. Go to [vercel.com](https://vercel.com) and click **Add New...** $\rightarrow$ **Project**.
2. Import your GitHub repository.
3. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. In **Environment Variables**, add:
   - **`VITE_API_URL`**: `https://your-service.onrender.com` (or your Hugging Face Space URL, with no trailing slash).
5. Click **Deploy**.

---

## 🛡️ Security & Hardening Features

- **Strict CORS**: Origins are dynamically restricted via `FRONTEND_URL`.
- **In-Memory Rate Limiting**: Protects inference endpoints from spam (15 req/min/IP).
- **File Validation**: Enforces extension checks, MIME types, duration ceilings, and 25 MB payload limits.
- **Safe Temp File Isolation**: Audio conversions use cryptographically random UUID filenames with strict `try ... finally` disk cleanup.
- **Readiness Probes**: `/api/ready` ensures models and scalers are loaded into RAM before serving inference traffic.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
