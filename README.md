# 🎙️ Deepfake Audio Detection System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF.svg?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade AI audio forensics and deepfake detection application. The system analyzes speech recordings (MP3, WAV, M4A, OGG, FLAC), extracts **26 acoustic features**, evaluates them across a **3-model deep learning ensemble** (CNN-RNN, Deep 1D CNN, CNN-Transformer), and generates an authenticity verdict through majority voting consensus.

---

## 🌐 Live Production Deployments

| Component | Provider / Platform | Production URL |
| :--- | :--- | :--- |
| **Frontend Application** | Vercel | [https://deepfake-audio-detection-rust.vercel.app](https://deepfake-audio-detection-rust.vercel.app) |
| **Backend API** | Render (Free Tier) | [https://deepfake-audio-api-faon.onrender.com](https://deepfake-audio-api-faon.onrender.com) |
| **Swagger Interactive Docs** | FastAPI Docs | [https://deepfake-audio-api-faon.onrender.com/docs](https://deepfake-audio-api-faon.onrender.com/docs) |
| **Liveness Health Check** | REST API | [https://deepfake-audio-api-faon.onrender.com/api/health](https://deepfake-audio-api-faon.onrender.com/api/health) |
| **Readiness Check** | REST API | [https://deepfake-audio-api-faon.onrender.com/api/ready](https://deepfake-audio-api-faon.onrender.com/api/ready) |

---

## 📑 Table of Contents

1. [Production Architecture](#-production-architecture)
2. [Evaluated Architectures & Test-Split Performance](#-evaluated-architectures--test-split-performance)
3. [Repository Structure](#-repository-structure)
4. [Environment Variables](#️-environment-variables)
5. [Local Development Setup](#-local-development-setup-native-python--no-docker-required)
6. [Running Automated Tests](#-running-automated-tests)
7. [Production Deployment Guide](#-production-deployment-guide-100-free--native-python)
8. [Production Deployment Journey, Issues & Solutions](#-production-deployment-journey-issues--solutions)
   - [Initial Render Python & TensorFlow Resolution Issue](#1-initial-render-python--tensorflow-resolution-issue)
   - [Backend Deployment Success & Port Binding](#2-backend-deployment-success--port-binding)
   - [Swagger / API Verification](#3-swagger--api-verification)
   - [Frontend Vercel Root Directory Build Mismatch](#4-frontend-vercel-root-directory-build-mismatch)
   - [API URL & Dynamic Environment Variables](#5-api-url--dynamic-environment-variables)
   - [Production CORS Configuration](#6-production-cors-configuration)
   - [The 45-Second Client Timeout Problem](#7-the-45-second-client-timeout-problem)
   - [Render Free Cold-Start Latency Analysis](#8-render-free-cold-start-latency-analysis)
   - [Why the Same File Times Out First and Succeeds Second](#9-why-the-same-file-times-out-first-and-succeeds-second)
   - [Health Probe vs. Readiness Probe (`/api/health` vs `/api/ready`)](#10-health-probe-vs-readiness-probe-apihealth-vs-apiready)
   - [Memory Constraints & Sequential Model Lifecycle](#11-memory-constraints--sequential-model-lifecycle)
   - [Frontend Startup Warm-Up & Readiness UX](#12-frontend-startup-warm-up--readiness-ux)
   - [Polling Safeguards, Timeout & Retry Handling](#13-polling-safeguards-timeout--retry-handling)
   - [Pre-Prediction Readiness Verification](#14-pre-prediction-readiness-verification)
   - [Lessons Learned & Root Cause Summary Table](#15-lessons-learned--root-cause-summary-table)
   - [Current End-to-End Production Request Flow](#16-current-end-to-end-production-request-flow)
   - [Free Tier Architecture Limitations](#17-free-tier-architecture-limitations)
   - [Production Troubleshooting Guide](#18-production-troubleshooting-guide)
   - [Deployment Verification Checklist](#19-deployment-verification-checklist)
9. [Security & Hardening Features](#-security--hardening-features)
10. [License](#-license)

---

## 🏗️ Production Architecture

```text
                                  USER BROWSER
                                       │
                                       ▼
                       Vercel React 18 / Vite 5 Frontend
                  [https://deepfake-audio-detection-rust.vercel.app]
                                       │
                                       │ HTTPS REST API
                                       ▼
                        Render FastAPI Backend (Linux)
                     [https://deepfake-audio-api-faon.onrender.com]
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ 1. Rate Limiting & Validation Check          │
                │ 2. Audio Decoding (Librosa / SoundFile)      │
                │ 3. 26 Acoustic Feature Extraction            │
                │ 4. Pre-computed StandardScaler Normalization │
                └──────────────────────┬──────────────────────┘
                                       │
                     Sequential 3-Model Inference Engine
                       (Render Free 512 MB RAM Safeguard)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   [ Model 1: Drashya ]       [ Model 2: Devesh ]        [ Model 3: Swayam ]
   - Architecture: CNN-RNN    - Architecture: Deep CNN   - Architecture: Transformer
   - Load .h5 from disk       - Load .h5 from disk       - Load .h5 from disk
   - Evaluate single sample   - Evaluate single sample   - Evaluate single sample
   - Extract Prob P(REAL)     - Extract Prob P(REAL)     - Extract Prob P(REAL)
   - Clear Session & GC       - Clear Session & GC       - Clear Session & GC
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                         Majority Voting Consensus Engine
                           - Decision Rule: ≥ 2 of 3 votes
                           - Soft-Probability Calibration
                                       │
                                       ▼
                              JSON Verdict Response
                          (REAL / FAKE + Model Breakdown)
                                       │
                                       ▼
                        React Forensics Results Dashboard
```

### 🧠 The Three Evaluated Models

1. **Drashya Model (1D CNN + RNN Hybrid)**:
   - Combines 1D convolutional feature extractors with recurrent LSTM/GRU layers to capture spectral feature sequences.
2. **Devesh Model (Deep 1D CNN)**:
   - Utilizes deep 1D convolutional layers with Batch Normalization and Dropout regularization to isolate synthesis artifacts.
3. **Swayam Model (CNN-Transformer with Positional Encoding)**:
   - Uses Multi-Head Self-Attention layers paired with sinusoidal positional encodings to capture cross-feature interdependencies.

> [!IMPORTANT]
> **Sequential Memory Lifecycle**: Due to Render Free's 512 MB memory constraint, the three Keras models are **not** kept permanently resident in RAM. Instead, each model is loaded sequentially on-demand, evaluated on the input tensor, and immediately dereferenced via `del model`, `tf.keras.backend.clear_session()`, and `gc.collect()`.

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
│       ├── inference.py            # Sequential model inference & majority voting
│       └── rate_limiter.py         # In-memory sliding-window rate limiting
│
├── frontend/                       # Modern React + Vite Forensics Web App
│   ├── src/
│   │   ├── components/             # Modular UI components (Waveform, Results, etc.)
│   │   ├── config/api.js           # Centralized API network layer with timeouts
│   │   ├── App.jsx                 # Application state, startup readiness & UX
│   │   └── index.css               # Tailwind CSS & custom forensics theme
│   ├── package.json
│   ├── vite.config.js              # Vite configuration with local proxy fallback
│   └── .env.example                # Frontend environment template
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

## 🚀 Production Deployment Guide (100% Free & Native Python)

### 1. Backend Deployment on Render.com (Web Service)
1. Create a free account on [render.com](https://render.com).
2. Click **New +** $\rightarrow$ **Web Service** $\rightarrow$ Connect your GitHub repo.
3. Configure settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables:
   - `FRONTEND_URL`: `https://deepfake-audio-detection-rust.vercel.app,http://localhost:5173`
   - `PYTHON_VERSION`: `3.11.9`
5. Click **Create Web Service**. Copy your live backend URL (`https://deepfake-audio-api-faon.onrender.com`).

### 2. Frontend Deployment on Vercel
1. Go to [vercel.com](https://vercel.com) and click **Add New...** $\rightarrow$ **Project**.
2. Import your GitHub repository.
3. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. In **Environment Variables**, add:
   - **`VITE_API_URL`**: `https://deepfake-audio-api-faon.onrender.com` *(base URL only, no trailing slash, no `/api/predict`)*.
5. Click **Deploy**.

---

## 🛠️ Production Deployment Journey, Issues & Solutions

During the production rollout of this system to a **100% free-tier public portfolio architecture** (Vercel + Render Free), several real-world cloud deployment challenges were encountered and engineered around. This section details the technical root causes, actual log data, and implemented solutions.

---

### 1. Initial Render Python & TensorFlow Resolution Issue

#### The Problem:
On Render's initial automated build, the environment variable `PYTHON_VERSION=3.11.x` could not be resolved by Render's version selector, causing the build environment to fall back to the platform default:
```text
==> Failed to resolve Python version '3.11.x' from environment variable PYTHON_VERSION; falling back to default version
==> Using Python version 3.14.3
==> Running build command 'pip install -r backend/requirements.txt'...
...
ERROR: Could not find a version that satisfies the requirement tensorflow>=2.15.0 (from versions: none)
ERROR: No matching distribution found for tensorflow>=2.15.0
```

#### The Root Cause:
At build time, pre-compiled binary distribution wheels (`.whl`) for `tensorflow` were not available for Python 3.14.3 in PyPI, and the `3.11.x` wildcard syntax failed Render's exact version resolution parser.

#### The Solution:
The environment configuration was updated to specify an exact stable release: `PYTHON_VERSION=3.11.9` (and Python 3.11/3.12 compatibility flags). On subsequent builds, Render successfully installed the complete machine learning dependency set:
- `tensorflow-2.21.0`
- `numpy-1.26.4`
- `fastapi-0.141.1`
- `uvicorn-0.52.4`
- `librosa-0.11.0`
- `scikit-learn-1.9.0`
- `pandas-3.0.5`

---

### 2. Backend Deployment Success & Port Binding

#### Resolution:
With dependencies resolved, Render built and deployed the backend successfully:
```text
==> Build successful 🎉
==> Deploying...
==> Starting service with 'uvicorn backend.app:app --host 0.0.0.0 --port $PORT'
INFO:     Started server process [61]
INFO:     Waiting for application startup.
2026-08-29 09:53:56 [INFO] deepfake.preprocessing - Loaded feature scaler from models/scaler.pkl (26 dimensions).
2026-08-29 09:53:56 [INFO] deepfake.inference - Inference service initialized: All 3 model files verified. Sequential inference mode active.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:10000 (Press CTRL+C to quit)
==> Your service is live 🚀
```

> [!NOTE]
> **Why `$PORT` Binding Matters**: Render assigns an arbitrary port (e.g., `10000`) dynamically at container boot via the `$PORT` environment variable. Hardcoding `--port 8000` causes Render's reverse proxy health checks to fail with HTTP 502/504. The start command `uvicorn backend.app:app --host 0.0.0.0 --port $PORT` guarantees proper binding.
>
> **Root URL `GET /`**: Navigating to `https://deepfake-audio-api-faon.onrender.com/` in a browser returns `{"detail": "Not Found"}` (HTTP 404). This is **expected behavior** as the application explicitly scopes all REST routes under `/api/*` and interactive docs under `/docs`.

---

### 3. Swagger / API Verification

Direct verification via FastAPI's interactive Swagger UI (`/docs`) confirmed that:
1. The backend process was live and reachable over HTTPS.
2. Endpoint schemas and multipart form upload boundaries were registered.
3. Submitting sample `file1004.mp3` returned `HTTP 200 OK` with 3-model consensus:
   ```json
   {
     "status": "success",
     "final_decision": "FAKE",
     "is_fake": true,
     "is_real": false,
     "majority_vote": {
       "agreement": "3/3",
       "ensemble_confidence_pct": 99.9
     },
     "processing_time_ms": 936.9
   }
   ```
*(Swagger testing validated backend integrity, but frontend-to-backend integration required distinct timeout and CORS configurations).*

---

### 4. Frontend Vercel Root Directory Build Mismatch

#### The Problem:
The initial Vercel deployment failed with the build error:
```text
sh: line 1: cd: frontend: No such file or directory
Error: Command "cd frontend && npm run build" exited with 1
```

#### The Root Cause:
In Vercel's project dashboard, the **Root Directory** was already set to `frontend`. When Vercel ran the build command, it had already changed working directory into `frontend/`. Executing `cd frontend` inside `frontend/` failed because `frontend/frontend` does not exist.

#### The Solution:
Updated Vercel Project Settings to:
- **Root Directory**: `frontend`
- **Build Command**: `npm run build` *(runs `vite build` directly in the root directory)*
- **Output Directory**: `dist`

---

### 5. API URL & Dynamic Environment Variables

#### Architecture:
To prevent hardcoding URLs across React components, all API calls route through [`frontend/src/config/api.js`](frontend/src/config/api.js):
```javascript
const rawApiUrl = import.meta.env.VITE_API_URL;
export const API_BASE_URL = (rawApiUrl || '').replace(/\/+$/, '');
```

#### Key Rules:
- **Base URL Only**: `VITE_API_URL` must contain only the host (e.g. `https://deepfake-audio-api-faon.onrender.com`). It must **not** include `/api/predict`.
- **Trailing Slashes**: The regex `.replace(/\/+$/, '')` automatically strips trailing slashes to prevent malformed double-slash routes (e.g. `...com//api/predict`).

---

### 6. Production CORS Configuration

The browser enforces Same-Origin Policy (SOP). Because the frontend lives at `https://deepfake-audio-detection-rust.vercel.app` and the backend lives at `https://deepfake-audio-api-faon.onrender.com`, cross-origin resource sharing must be permitted by the backend.

In `backend/config.py`:
```python
FRONTEND_URL_RAW = os.environ.get(
    "FRONTEND_URL",
    "http://localhost:5173,http://localhost:3000,https://deepfake-audio-detection-rust.vercel.app"
)
ALLOWED_ORIGINS = [url.strip() for url in FRONTEND_URL_RAW.split(",") if url.strip()]
```
And configured in `backend/app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```

---

### 7. The 45-Second Client Timeout Problem

#### The Problem:
When users submitted audio files on the live Vercel frontend, the UI displayed:
```text
"Request timed out after 45s. The server may be busy."
```
Yet inspecting Render server logs revealed that the backend was still alive and eventually completed the inference successfully.

#### The Root Cause:
The frontend `fetchWithTimeout` helper used a client-side timeout:
```javascript
const DEFAULT_TIMEOUT_MS = 45000; // 45 seconds
```
On Render Free, an inactive container can take **30 to 45 seconds just to boot from sleep** before Uvicorn even receives the incoming HTTP packet. Combined with sequential model loading and Librosa audio decoding, the total elapsed time exceeded 45 seconds, causing the browser's `AbortController` to abort the connection prematurely.

#### The Solution:
Increased the client prediction timeout to **120 seconds (120,000 ms)** in [`frontend/src/config/api.js`](frontend/src/config/api.js):
```javascript
const DEFAULT_TIMEOUT_MS = 120000; // 120 seconds
const HEALTH_TIMEOUT_MS = 30000;    // 30 seconds for health/ready probes
```
*(This is a client-side allowance; it gives Render Free sufficient headroom to complete cold-start boot and inference).*

---

### 8. Render Free Cold-Start Latency Analysis

On Render's Free tier, instances spin down after 15 minutes of inactivity. Real-world observed request durations from project test logs:

| Scenario / Request Sequence | Observed Total Latency | Notes / Root Cause |
| :--- | :--- | :--- |
| **Initial Cold Request** | **~366,326 ms** (~366.3s ≈ 6.1 min) | Cold container initialization + OS boot + Python spin-up + initial package loading under cold disk cache. |
| **Subsequent Request 1** | **~53,400 ms** (~53.4s) | Container awake; initial Keras/TF native C++ runtime initialization. |
| **Subsequent Request 2** | **~43,373 ms** (~43.4s) | Sequential 3-model execution on warmed container. |
| **Warm Production Request 3** | **~31,902 ms** (~31.9s) | Audio decoding + sequential inference. |
| **Warm Production Request 4** | **~20,446 ms** (~20.4s) | Audio decoding + sequential inference with warm OS file cache. |

> [!NOTE]
> These figures represent actual observed project measurements under varying free-tier host loads, not guaranteed performance metrics. Warm requests are consistently faster than cold starts.

---

### 9. Why the Same File Times Out First and Succeeds Second

A common source of confusion during testing:
1. **Attempt 1 (Timeout)**: The user clicks *Analyze Audio*. The frontend timer runs for 120 seconds while Render spins up the container. If spin-up takes longer than the timeout window, the browser client displays a timeout error. **However**, Render continues booting in the background.
2. **Attempt 2 (Immediate Success)**: The user clicks *Analyze Audio* again. The backend container is now **warm and running**. The request is processed immediately in ~20–30 seconds, returning a `200 OK` prediction verdict.

---

### 10. Health Probe vs. Readiness Probe (`/api/health` vs `/api/ready`)

| Endpoint | Type | Exact Check Performed | HTTP 200 Meaning |
| :--- | :--- | :--- | :--- |
| **`GET /api/health`** | Liveness Probe | Confirms FastAPI process is responsive. | FastAPI process is running and accepting TCP connections. Does **not** check models or scaler. |
| **`GET /api/ready`** | Readiness Probe | Checks that `scaler.pkl` is loaded and all 3 model files exist on disk (`os.path.exists`). | Structural readiness: Scaler is in memory and all 3 `.h5` model files are accessible on disk for sequential loading. |

> [!CAUTION]
> **Important Distinction**: `/api/ready` does **NOT** mean all three TensorFlow models are loaded in RAM simultaneously. Keeping all 3 models in RAM simultaneously consumes ~500 MB+ and would trigger Render's Out-Of-Memory (OOM) killer.

---

### 11. Memory Constraints & Sequential Model Lifecycle

#### The 512 MB RAM Barrier:
Render's Free tier enforces a strict **512 MB RAM ceiling**. Keeping all 3 deep learning models (`CNN-RNN`, `Deep 1D CNN`, `CNN-Transformer`) loaded simultaneously in memory resulted in process RSS values of:
```text
568 MB / 574 MB / 575 MB / 588 MB
```
which triggers container SIGKILL (HTTP 502 Bad Gateway).

#### The Sequential Architecture:
In [`backend/services/inference.py`](backend/services/inference.py), the engine evaluates models sequentially with explicit session clearing:

```python
# 1. Evaluate Model 1
model = tf.keras.models.load_model(drashya_path, compile=False)
p1 = float(np.squeeze(model(X_input, training=False).numpy()))
del model
tf.keras.backend.clear_session()
gc.collect()

# 2. Evaluate Model 2
model = tf.keras.models.load_model(devesh_path, compile=False)
p2 = float(np.squeeze(model(X_input, training=False).numpy()))
del model
tf.keras.backend.clear_session()
gc.collect()

# 3. Evaluate Model 3
model = tf.keras.models.load_model(swayam_path, compile=False, custom_objects={'PositionalEncoding': PositionalEncoding})
p3 = float(np.squeeze(model(X_input, training=False).numpy()))
del model
tf.keras.backend.clear_session()
gc.collect()
```

- A process-local `threading.Lock()` serializes model loading across concurrent requests to guarantee at most one model occupies RAM at any instant.

---

### 12. Frontend Startup Warm-Up & Readiness UX

To prevent users from experiencing cold-start timeouts when uploading audio, the frontend actively initiates a background warm-up as soon as the React application mounts.

In [`frontend/src/App.jsx`](frontend/src/App.jsx):

```text
User opens Web App
        │
        ▼
React Mounts: backendStatus = 'checking'
        │
        ├── UI displays: "Preparing AI Detector... Waking up inference server..."
        ├── Analyze Button is visually disabled
        │
        ▼
Background Polling loop triggers:
GET /api/health ──► GET /api/ready (every 3s)
        │
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
Success (HTTP 200)             Timeout (90s elapsed)
   │                                  │
   ▼                                  ▼
backendStatus = 'ready'        backendStatus = 'error'
- UI shows: "Detector Ready"   - UI shows: "Server taking longer than expected"
- Normal Upload/Analyze active - Shows: [ 🔄 Retry Connection ] button
```

---

### 13. Polling Safeguards, Timeout & Retry Handling

The frontend polling implementation includes robust resilience features:
- **3-Second Interval**: Calls `checkBackendReady()` every 3 seconds while `backendStatus === 'checking'`.
- **Immediate Halt on Success**: Clears polling timer as soon as `/api/ready` returns HTTP 200.
- **90-Second Finite Deadline**: If the backend does not respond within 90 seconds, polling halts gracefully and displays the retry state.
- **Strict React 18 Cleanup**: Uses `isMountedRef` and `pollTimerRef` to cancel timers on unmount, preventing React StrictMode double-interval bugs.
- **Graceful Error Catching**: All polling network errors are caught silently as *"still waking up"* without uncaught promise rejections or React component crashes.
- **In-Place Retry**: Clicking **Retry Connection** resets the timer and begins a fresh 90s cycle without reloading the web page.

---

### 14. Pre-Prediction Readiness Verification

Immediately before dispatching the audio payload in `App.jsx`, a secondary readiness check is performed:
```text
User clicks "Analyze Audio"
        │
        ▼
Check backendStatus === 'ready' (abort if not ready)
        │
        ▼
GET /api/ready (pre-flight check)
        │
        ▼
POST /api/predict (multipart/form-data)
```
This ensures expensive `POST /api/predict` payloads are never dispatched against an uninitialized backend.

---

### 15. Lessons Learned & Root Cause Summary Table

| # | Problem | Symptom | Technical Root Cause | Final Engineering Solution |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Python Version Resolution** | Render build failed with `Could not find version tensorflow>=2.15.0`. | Render failed to parse wildcard `3.11.x` and fell back to Python 3.14. | Set exact `PYTHON_VERSION=3.11.9` in Render environment settings. |
| **2** | **Vercel Build Path** | `cd frontend: No such file or directory` on Vercel build. | Root directory was set to `frontend`, causing `cd frontend` command to fail. | Configured Root Directory = `frontend` and Build Command = `npm run build`. |
| **3** | **Port Binding** | Backend inaccessible or 502 on Render. | Hardcoding port 8000 conflicted with Render's assigned port. | Passed `--port $PORT` to Uvicorn start command. |
| **4** | **Root Route 404** | `GET /` returned 404 Not Found. | No root index handler defined in FastAPI. | Standard behavior; documented that REST API routes exist under `/api/*` and docs at `/docs`. |
| **5** | **Client 45s Timeout** | "Request timed out after 45s" on initial prediction. | Client timeout was shorter than Render Free cold-start wake-up latency. | Increased prediction client timeout to 120 seconds (`DEFAULT_TIMEOUT_MS = 120000`). |
| **6** | **Render Cold Start** | Initial request took ~30–50s while subsequent requests took ~20s. | Free-tier container spins down after 15 min of inactivity. | Implemented automatic background health/readiness warm-up on React app mount. |
| **7** | **First Timeout / Second Success** | Attempt 1 timed out, Attempt 2 succeeded with identical file. | Attempt 1 woke container; Attempt 2 hit warm running process. | Documented cold vs. warm behavior; added startup polling UX. |
| **8** | **512 MB RAM Ceiling** | Out-of-memory crash (SIGKILL / 502) when running predictions. | Loading all 3 Keras models simultaneously in RAM exceeded 512 MB. | Refactored `InferenceService` to load, evaluate, and unload models sequentially with `clear_session()`. |
| **9** | **Readiness Semantics** | Misconception that `/api/ready` loads all models into RAM. | Loading models at startup would violate memory ceiling. | Defined `/api/ready` as verification of `scaler.pkl` in RAM and `.h5` file presence on disk. |
| **10** | **CORS Restrictions** | Browser blocked API requests from Vercel domain. | Backend `ALLOWED_ORIGINS` did not include Vercel production domain. | Configured `FRONTEND_URL` environment variable with Vercel origin. |
| **11** | **Premature Submissions** | Users clicking Analyze while backend was waking received errors. | UI allowed submissions before backend connection was established. | Added startup readiness banner, disabled Analyze button during check, and added 90s retry UX. |

---

### 16. Current End-to-End Production Request Flow

```text
1. User navigates to https://deepfake-audio-detection-rust.vercel.app
   │
2. React App mounts ──► Displays "Preparing AI Detector... Waking up inference server..."
   │
3. App polls GET https://deepfake-audio-api-faon.onrender.com/api/ready every 3s
   │
4. Render container wakes up ──► /api/ready responds HTTP 200 {"status": "ready"}
   │
5. React UI updates ──► "Detector Ready" (Upload & Analyze controls enabled)
   │
6. User uploads audio file or selects a pre-packaged benchmark sample
   │
7. User clicks "Analyze Audio"
   │
8. React app sends POST /api/predict (multipart/form-data) [120s timeout budget]
   │
9. FastAPI backend receives request:
   ├─ Checks rate limit (15 req/min/IP)
   ├─ Librosa decodes audio waveform (16 kHz mono)
   ├─ Computes 26 acoustic features (Chroma, RMS, Spectral, MFCCs 1–20)
   ├─ Normalizes features with StandardScaler -> tensor (1, 26, 1)
   ├─ Acquires process-local inference lock
   ├─ Model 1 (Drashya CNN-RNN): Load -> Predict -> Delete -> Clear Session & GC
   ├─ Model 2 (Devesh Deep CNN): Load -> Predict -> Delete -> Clear Session & GC
   ├─ Model 3 (Swayam Transformer): Load -> Predict -> Delete -> Clear Session & GC
   └─ Majority Voting Consensus Engine calculates verdict and confidence
   │
10. Backend responds with HTTP 200 JSON payload
   │
11. Frontend renders interactive Forensics Results Dashboard with audio playback
```

---

### 17. Free Tier Architecture Limitations

This application is engineered specifically for a **$0 budget** (Vercel + Render Free):

- **Inactivity Spin-Down**: Render Free services spin down after 15 minutes without traffic. The first visit will incur a 30–50 second container cold start.
- **Sequential Latency Overhead**: Loading models sequentially from disk adds ~1.5–2.5 seconds to total inference time compared to keeping all models in RAM. This trade-off is mandatory to remain within 512 MB RAM.
- **Concurrency Serialization**: Process-local locking serializes model execution, meaning concurrent requests from multiple users queue up sequentially.
- **Hobby / Portfolio Workload**: This architecture is optimized for portfolio demonstrations, academic evaluation, and testing rather than commercial high-concurrency throughput.

---

### 18. Production Troubleshooting Guide

#### If the frontend shows *"Inference server is taking longer than expected"*:
1. Check Render dashboard to verify the `deepfake-audio-api` service status is **Live**.
2. Test the liveness probe directly in your browser: [https://deepfake-audio-api-faon.onrender.com/api/health](https://deepfake-audio-api-faon.onrender.com/api/health).
3. Test the readiness probe: [https://deepfake-audio-api-faon.onrender.com/api/ready](https://deepfake-audio-api-faon.onrender.com/api/ready).
4. Click **Retry Connection** in the frontend banner.

#### If a prediction request times out:
1. Inspect Render logs for memory errors or container restarts.
2. Verify log lines show:
   - `Received audio analysis request for ...`
   - `Loading model: ...`
   - `Completed inference: ...`
   - `Analysis complete for ...: Verdict=...`
3. If the backend was in a deep cold-start, wait 10 seconds and click **Analyze Audio** again; warm requests complete in ~20–30s.

#### If navigating to the backend root URL shows 404:
- `GET /` returning 404 is normal. REST endpoints are located at `/api/*` and interactive documentation is at `/docs`.

#### If Vercel build fails:
- Ensure **Root Directory** is `frontend`, **Build Command** is `npm run build`, and **Output Directory** is `dist`.

---

### 19. Deployment Verification Checklist

#### Backend (Render):
- [x] Python 3.11 environment active (`PYTHON_VERSION=3.11.9`).
- [x] All dependencies installed (`tensorflow 2.21.0`, `librosa`, `fastapi`, `uvicorn`).
- [x] Uvicorn bound to dynamic `$PORT` (`0.0.0.0:$PORT`).
- [x] `GET /api/health` returns `200 OK`.
- [x] `GET /api/ready` returns `200 OK` with 3 models available and scaler loaded.
- [x] Interactive Swagger UI active at `/docs`.
- [x] `POST /api/predict` returns valid 3-model majority voting verdict.

#### Frontend (Vercel):
- [x] Vercel project configured with Root Directory `frontend` and Build Command `npm run build`.
- [x] `VITE_API_URL` environment variable set to `https://deepfake-audio-api-faon.onrender.com`.
- [x] Production bundle builds cleanly in `dist/`.
- [x] Startup readiness polling active (polls `/api/ready` every 3s with 90s deadline).
- [x] Startup loader and failure Retry button functional.
- [x] Audio file upload and sample testing operational.
- [x] 120-second prediction timeout configured.

#### Cross-Origin Integration:
- [x] Backend `FRONTEND_URL` allows `https://deepfake-audio-detection-rust.vercel.app`.
- [x] Pre-flight CORS `OPTIONS` requests succeed.
- [x] End-to-end audio analysis returns verdict to frontend without errors.

---

## 🛡️ Security & Hardening Features

- **Strict CORS**: Origins are dynamically restricted via `FRONTEND_URL`.
- **In-Memory Rate Limiting**: Protects inference endpoints from spam (15 req/min/IP).
- **File Validation**: Enforces extension checks, MIME types, duration ceilings (60s), and 25 MB payload limits.
- **Safe Temp File Isolation**: Audio conversions use cryptographically random UUID filenames with strict `try ... finally` disk cleanup.
- **Readiness Probes**: `/api/ready` verifies assets and pre-computed scalers before serving traffic.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
