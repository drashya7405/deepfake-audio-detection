# 🚀 Complete Deployment Guide: Deepfake Audio Detection (100% Free)

This guide shows you how to deploy the project **100% free with NO paid plans, NO Docker fees, and NO commercial licenses needed**.

---

## 🏗️ Architecture

- **Frontend**: Hosted on **Vercel** (Free global CDN).
- **Backend (ML Engine)**: Hosted on **Hugging Face Spaces** (Free 16 GB RAM CPU tier) or **Render.com** (Free tier).

---

## 🏆 Option 2: Deploying to Hugging Face Spaces (100% Free, No Docker)

Hugging Face Spaces offers a **permanent Free Tier (2 vCPUs · 16 GB RAM)** that runs Python natively without needing Docker or credit cards.

---

### Step 1: Create a Free Hugging Face Space

1. Go to [huggingface.co](https://huggingface.co) and sign in (create a free account if you don't have one).
2. Click **New Space** (or visit [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Fill in the fields:
   - **Space name**: `deepfake-audio-detector-api`
   - **License**: Select **`mit`** or **`openrail`** from the dropdown.
     *(Note: This is just an open-source tag for your public code. It is 100% free and does NOT require you to own or buy a license).*
   - **Select the Space SDK**: Choose **Gradio** (This runs standard Python without Docker for free!).
   - **Space hardware**: Select **CPU basic · 2 vCPU · 16 GB RAM · Free**.
   - **Space visibility**: Select **Public** (so your Vercel frontend can call the API).
4. Click **Create Space**.

---

### Step 2: Upload Project Files to Hugging Face

You can upload the files using either **Git CLI** (Method A) or directly via **Web Browser** (Method B).

#### Method A: Via Git CLI (Fastest)

Run these commands in your computer's terminal:

```bash
# 1. Clone your empty Hugging Face Space
git clone https://huggingface.co/spaces/<YOUR-HF-USERNAME>/deepfake-audio-detector-api hf-space

# 2. Copy the project files into the cloned folder
cp app.py hf-space/
cp requirements.txt hf-space/
cp packages.txt hf-space/
cp DATASET-balanced.csv hf-space/
cp gehra_hua.mp3 hf-space/
cp -r backend/ hf-space/backend/
cp -r BestModels/ hf-space/BestModels/
cp -r FAKE_AUDIOS/ hf-space/FAKE_AUDIOS/

# 3. Enter the folder, commit, and push
cd hf-space
git add .
git commit -m "Deploy deepfake detection ML backend"
git push
```

*(When prompted for password, enter your Hugging Face Access Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).*

---

#### Method B: Via Browser Upload

1. Open your Hugging Face Space in your browser.
2. Click the **Files** tab $\rightarrow$ **Add file** $\rightarrow$ **Upload files**.
3. Drag & drop the following files into the browser:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `DATASET-balanced.csv`
   - `gehra_hua.mp3`
   - The folders `backend/`, `BestModels/`, `FAKE_AUDIOS/`
4. Click **Commit changes to main**.

---

### Step 3: Verify Your Backend API

1. Hugging Face will automatically install the requirements and launch the FastAPI server.
2. Wait until the top status badge turns from **Building** $\rightarrow$ **Running** (green).
3. **Your direct API URL** will be:
   ```text
   https://<YOUR-HF-USERNAME>-deepfake-audio-detector-api.hf.space
   ```
4. Test it in your browser:
   - `https://<YOUR-HF-USERNAME>-deepfake-audio-detector-api.hf.space/docs` (Interactive Swagger Docs)
   - `https://<YOUR-HF-USERNAME>-deepfake-audio-detector-api.hf.space/api/health` (Health Check)

---

### Step 4: Deploy the Frontend to Vercel

1. Push your project code to **GitHub**.
2. Go to **[vercel.com](https://vercel.com)** $\rightarrow$ click **Add New...** $\rightarrow$ **Project**.
3. Import your GitHub repository.
4. Configure settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click `Edit` and select `frontend` (or keep root).
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. **Environment Variable**:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://<YOUR-HF-USERNAME>-deepfake-audio-detector-api.hf.space`
     *(Ensure there is NO trailing slash `/` at the end)*
6. Click **Deploy**. Your frontend will be live on Vercel!

---

## ⚡ Option 1: Deploying Backend to Render.com (100% Free, 1-Click)

If you prefer using [Render.com](https://render.com) (no Hugging Face account needed):

1. Go to [render.com](https://render.com) and create a free account.
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository.
4. Settings:
   - **Name**: `deepfake-audio-api`
   - **Language**: `Python`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Click **Create Web Service** (Free).
6. Copy your Render URL: `https://deepfake-audio-api.onrender.com`.
7. In Vercel, set `VITE_API_URL` = `https://deepfake-audio-api.onrender.com` and deploy!

---

## 💻 Running Locally

```bash
# Start both Backend and Frontend locally
python run.py
```
- **Web App**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000/docs`
