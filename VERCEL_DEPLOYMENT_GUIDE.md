# 🚀 Deployment Guide: Deepfake Audio Detection (Native Python + Vercel)

This guide provides end-to-end instructions for deploying the project to production **without Docker** using a decoupled architecture:
- **Frontend**: Hosted on **Vercel** (Global Edge CDN, instant loading).
- **Backend**: Hosted on **Render.com** (Native Python Web Service) or **Hugging Face Spaces** (Free 16 GB RAM CPU tier).

---

## 🏗️ Architecture

```text
┌────────────────────────────────────────────────────────┐
│               🌐 Vercel (Global Edge CDN)              │
│  React + Vite + Tailwind CSS + Web Audio Visualizers  │
└───────────────────────────┬────────────────────────────┘
                            │
               HTTPS / CORS REST API
                            │
┌───────────────────────────▼────────────────────────────┐
│      🐍 Native Python ML Backend (Render / HF)         │
│  FastAPI + TensorFlow + Librosa + 3 Best Models        │
│  (Drashya CNN-RNN, Devesh CNN, Swayam Transformer)    │
└────────────────────────────────────────────────────────┘
```

---

## ⚡ Option 1: Backend on Render.com (100% Free)

1. Sign up at [render.com](https://render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository: `drashya7405/deepfake-audio-detection`.
4. Configure service settings:
   - **Name**: `deepfake-audio-api`
   - **Environment**: `Python`
   - **Region**: Oregon or Frankfurt
   - **Branch**: `main`
   - **Build Command**:
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.app:app --host 0.0.0.0 --port $PORT
     ```
5. **Environment Variables**:
   - `FRONTEND_URL`: `https://your-vercel-app.vercel.app` (Add after deploying frontend)
   - `LOG_LEVEL`: `INFO`
   - `MAX_UPLOAD_MB`: `25`
   - `MAX_AUDIO_DURATION_SECONDS`: `60`
   - `RATE_LIMIT_PER_MINUTE`: `15`
6. Click **Create Web Service**.
7. Copy your live Render URL: `https://deepfake-audio-api.onrender.com`.

---

## ⚡ Option 2: Backend on Hugging Face Spaces (16 GB RAM Free Tier)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Configure settings:
   - **Space name**: `deepfake-audio-detector-api`
   - **License**: Select `mit` or `none` (open-source label, free)
   - **Space SDK**: Select **Gradio** (Runs standard Python natively)
   - **Space hardware**: `CPU basic · 2 vCPU · 16 GB RAM · Free`
   - **Visibility**: **Public**
3. Push your repository to Hugging Face Spaces:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR-USERNAME>/deepfake-audio-detector-api
   git push hf main
   ```
4. Copy your direct Space URL:
   ```text
   https://<YOUR-USERNAME>-deepfake-audio-detector-api.hf.space
   ```

---

## 🌐 Deploy Frontend to Vercel

1. Log in to [vercel.com](https://vercel.com).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Import your GitHub repository: `drashya7405/deepfake-audio-detection`.
4. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. **Environment Variables**:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://deepfake-audio-api.onrender.com` (or your Hugging Face Space URL, with no trailing slash).
6. Click **Deploy**.

