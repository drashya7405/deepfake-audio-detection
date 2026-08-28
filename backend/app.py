import os
import io
import time
import shutil
import tempfile
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --- 1. SETUP PATHS & CUSTOM KERAS LAYER ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "BestModels")
DATASET_PATH = os.path.join(PROJECT_ROOT, "DATASET-balanced.csv")
SAMPLE_AUDIOS_DIR = PROJECT_ROOT

FEATURE_COLUMNS = [
    'chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
    'rolloff', 'zero_crossing_rate',
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8',
    'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
    'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
]

# Lazy import tensorflow to speed up start
os.environ["KERAS_HOME"] = os.path.join(PROJECT_ROOT, ".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Custom Positional Encoding layer required by Swayam's CNN-Transformer model.
    """
    def __init__(self, max_steps=1000, max_dims=512, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims
        
        dims = max_dims
        if dims % 2 == 1:
            dims += 1 
        
        p, i = np.meshgrid(np.arange(max_steps), np.arange(dims // 2))
        pos_emb = np.empty((1, max_steps, dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / dims)).T
        self.positional_encoding = tf.constant(pos_emb, dtype=tf.float32)

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_encoding[:, :shape[1], :shape[2]]
        
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "max_steps": self.max_steps,
            "max_dims": self.max_dims
        })
        return config


# --- 2. GLOBAL CACHE FOR MODELS & SCALER ---
class ModelManager:
    def __init__(self):
        self.scaler_mean = None
        self.scaler_scale = None
        self.models: Dict[str, Any] = {}
        self.is_initialized = False

    def initialize(self):
        if self.is_initialized:
            return
            
        print("Initializing Deepfake Audio Detection ML Engine...")
        # 1. Fit or Load Scaler Statistics from DATASET-balanced.csv
        if os.path.exists(DATASET_PATH):
            print(f"Loading training data from {DATASET_PATH} to compute feature normalization...")
            df = pd.read_csv(DATASET_PATH)
            X = df[FEATURE_COLUMNS].astype(float).values
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_scale = np.std(X, axis=0)
            # Avoid division by zero
            self.scaler_scale[self.scaler_scale == 0.0] = 1.0
            print("Feature Scaler fitted successfully.")
        else:
            raise FileNotFoundError(f"Dataset CSV not found at {DATASET_PATH}")

        # 2. Load 3 Best Models
        model_configs = [
            {
                "id": "drashya",
                "name": "Drashya Model (CNN-RNN Hybrid)",
                "filename": "drashya_best_deepfake_audio_model.h5",
                "architecture": "1D CNN + Recurrent Neural Network (LSTM/GRU)",
                "accuracy": "98.59%"
            },
            {
                "id": "devesh",
                "name": "Devesh Model (Deep CNN Architecture)",
                "filename": "devesh_best_deepfake_audio_model.h5",
                "architecture": "Deep 1D CNN with Batch Normalization & Dropout",
                "accuracy": "95.80%"
            },
            {
                "id": "swayam",
                "name": "Swayam Model (CNN-Transformer)",
                "filename": "swayam_best_deepfake_audio_model.h5",
                "architecture": "1D CNN + Multi-Head Self-Attention with Positional Encoding",
                "accuracy": "97.40%"
            }
        ]

        for cfg in model_configs:
            model_path = os.path.join(MODELS_DIR, cfg["filename"])
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found.")
                continue
            
            print(f"Loading model: {cfg['name']} from {cfg['filename']}...")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects={'PositionalEncoding': PositionalEncoding}
                )
                self.models[cfg["id"]] = {
                    "model": model,
                    "info": cfg
                }
                print(f"✓ Loaded {cfg['name']} successfully.")
            except Exception as e:
                print(f"✗ Failed loading {cfg['name']}: {e}")

        self.is_initialized = True
        print(f"Initialization complete. {len(self.models)}/3 models loaded into memory.")

    def scale_features(self, feature_values: List[float]) -> np.ndarray:
        raw = np.array(feature_values, dtype=float)
        scaled = (raw - self.scaler_mean) / self.scaler_scale
        # Reshape to (1, 26, 1) for Conv1D / sequence models
        return np.reshape(scaled, (1, len(feature_values), 1))


model_manager = ModelManager()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not model_manager.is_initialized:
        model_manager.initialize()
    yield

# --- 3. FASTAPI APPLICATION SETUP ---
app = FastAPI(
    title="Deepfake Audio Detection API",
    description="Multi-Model Ensemble Deepfake Audio Classification API with Majority Voting",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 4. FEATURE EXTRACTION & AUDIO CONVERSION HELPERS ---
def load_and_convert_audio(audio_bytes: bytes, original_filename: str):
    """
    Accepts any audio file format (MP3, WAV, M4A, OGG, FLAC, WEBM, AAC),
    saves to temporary storage, converts if needed, and loads audio waveform with librosa.
    """
    suffix = os.path.splitext(original_filename)[1].lower() if original_filename else ".wav"
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
        temp_in.write(audio_bytes)
        temp_in_path = temp_in.name

    wav_path = temp_in_path
    converted_temp = None

    try:
        # First attempt standard librosa load
        try:
            y, sr = librosa.load(temp_in_path, sr=None, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
        except Exception:
            # Fallback: Convert via pydub to clean 16-bit WAV
            converted_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            converted_temp_path = converted_temp.name
            converted_temp.close()

            audio_seg = AudioSegment.from_file(temp_in_path)
            audio_seg = audio_seg.set_channels(1)  # Mono
            audio_seg.export(converted_temp_path, format="wav")
            
            y, sr = librosa.load(converted_temp_path, sr=None, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            wav_path = converted_temp_path

        if len(y) == 0:
            raise ValueError("Audio signal is empty or corrupt.")

        # Extract the 26 acoustic features
        chroma_stft = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = [float(x) for x in np.mean(mfccs, axis=1)]

        feature_values = [
            chroma_stft, rms, spectral_centroid, spectral_bandwidth,
            rolloff, zero_crossing_rate, *mfcc_means
        ]

        feature_dict = dict(zip(FEATURE_COLUMNS, [round(v, 6) for v in feature_values]))

        audio_info = {
            "duration_seconds": round(duration, 2),
            "sample_rate": int(sr),
            "samples_count": len(y),
            "original_filename": original_filename or "audio_input"
        }

        return feature_values, feature_dict, audio_info

    finally:
        # Cleanup temp files
        if os.path.exists(temp_in_path):
            os.unlink(temp_in_path)
        if converted_temp and os.path.exists(converted_temp_path):
            os.unlink(converted_temp_path)


# --- 5. ENDPOINTS ---

@app.get("/api/health")
async def health_check():
    if not model_manager.is_initialized:
        model_manager.initialize()
    return {
        "status": "healthy",
        "models_loaded": list(model_manager.models.keys()),
        "total_models": len(model_manager.models)
    }


@app.get("/api/models-info")
async def get_models_info():
    if not model_manager.is_initialized:
        model_manager.initialize()
    return {
        "models": [m["info"] for m in model_manager.models.values()],
        "features_count": len(FEATURE_COLUMNS),
        "features_list": FEATURE_COLUMNS,
        "ensemble_method": "Majority Voting (>=2 of 3) with Soft-Probability Averaging"
    }


@app.get("/api/samples")
async def get_sample_audios():
    """
    Returns list of pre-packaged audio samples for immediate testing.
    """
    samples = []
    
    # Check sample files in workspace
    known_samples = [
        {"id": "sample_real_1", "name": "gehra_hua.mp3", "path": "gehra_hua.mp3", "tag": "Sample Audio 1", "type": "MP3 Audio"},
        {"id": "sample_fake_1", "name": "file1004.mp3", "path": "FAKE_AUDIOS/file1004.mp3", "tag": "AI Synthetic Voice", "type": "MP3 Audio"},
        {"id": "sample_fake_2", "name": "file1004.wav", "path": "FAKE_AUDIOS/file1004.wav", "tag": "AI Synthetic Voice", "type": "WAV Audio"},
        {"id": "sample_fake_3", "name": "file1019.wav", "path": "FAKE_AUDIOS/file1019.wav", "tag": "AI Synthetic Voice", "type": "WAV Audio"}
    ]

    for item in known_samples:
        full_p = os.path.join(PROJECT_ROOT, item["path"])
        if os.path.exists(full_p):
            samples.append({
                "id": item["id"],
                "name": item["name"],
                "path": item["path"],
                "tag": item["tag"],
                "type": item["type"],
                "size_kb": round(os.path.getsize(full_p) / 1024, 1)
            })

    return {"samples": samples}


@app.get("/api/sample-audio/{sample_id}")
async def get_sample_audio_file(sample_id: str):
    """
    Streams a sample audio file to the browser.
    """
    sample_mapping = {
        "sample_real_1": "gehra_hua.mp3",
        "sample_fake_1": "FAKE_AUDIOS/file1004.mp3",
        "sample_fake_2": "FAKE_AUDIOS/file1004.wav",
        "sample_fake_3": "FAKE_AUDIOS/file1019.wav"
    }

    if sample_id not in sample_mapping:
        raise HTTPException(status_code=404, detail="Sample not found")

    file_path = os.path.join(PROJECT_ROOT, sample_mapping[sample_id])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample file missing on disk")

    media_type = "audio/mpeg" if file_path.endswith(".mp3") else "audio/wav"
    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(file_path))


@app.post("/api/predict")
async def predict_audio(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None)
):
    """
    Inference endpoint:
    1. Extracts 26 acoustic features.
    2. Scales using StandardScaler parameters.
    3. Feeds features to the 3 Deep Learning models.
    4. Computes individual model votes and confidence levels.
    5. Calculates final result via Majority Voting (>=2 models).
    """
    if not model_manager.is_initialized:
        model_manager.initialize()

    audio_bytes = None
    filename = "audio_input"

    # Option A: Pre-packaged sample requested
    if sample_id:
        sample_mapping = {
            "sample_real_1": "gehra_hua.mp3",
            "sample_fake_1": "FAKE_AUDIOS/file1004.mp3",
            "sample_fake_2": "FAKE_AUDIOS/file1004.wav",
            "sample_fake_3": "FAKE_AUDIOS/file1019.wav"
        }
        if sample_id in sample_mapping:
            sample_rel_path = sample_mapping[sample_id]
            sample_full_path = os.path.join(PROJECT_ROOT, sample_rel_path)
            if os.path.exists(sample_full_path):
                with open(sample_full_path, "rb") as f:
                    audio_bytes = f.read()
                filename = os.path.basename(sample_rel_path)

    # Option B: Uploaded file
    if audio_bytes is None and file is not None:
        audio_bytes = await file.read()
        filename = file.filename or "uploaded_audio"

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio file or valid sample provided.")

    start_time = time.time()

    # 1. Feature Extraction & Conversion
    try:
        feature_values, feature_dict, audio_info = load_and_convert_audio(audio_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing error: {str(e)}")

    # 2. Scale & Reshape Features
    X_input = model_manager.scale_features(feature_values)

    # 3. Model Inference (3 Models)
    model_results = {}
    probabilities = []
    votes = []

    model_order = ["drashya", "devesh", "swayam"]

    for m_id in model_order:
        if m_id not in model_manager.models:
            continue

        item = model_manager.models[m_id]
        model = item["model"]
        info = item["info"]

        # Predict probability
        raw_pred = float(model.predict(X_input, verbose=0)[0][0])
        probabilities.append(raw_pred)

        # LabelEncoder: 0 = FAKE, 1 = REAL
        # raw_pred represents P(REAL)
        model_vote = "REAL" if raw_pred > 0.5 else "FAKE"
        votes.append(model_vote)

        # Confidence is distance from 0.5 decision boundary
        confidence_pct = (raw_pred if model_vote == "REAL" else (1.0 - raw_pred)) * 100.0

        model_results[m_id] = {
            "id": m_id,
            "name": info["name"],
            "architecture": info["architecture"],
            "prediction": model_vote,
            "raw_probability": round(raw_pred, 4),
            "real_probability_pct": round(raw_pred * 100, 2),
            "fake_probability_pct": round((1.0 - raw_pred) * 100, 2),
            "confidence_pct": round(confidence_pct, 2)
        }

    # 4. Majority Voting Logic
    real_votes_count = votes.count("REAL")
    fake_votes_count = votes.count("FAKE")
    total_votes = len(votes)

    # Majority decision: whichever has >= 2 votes
    final_decision = "REAL" if real_votes_count >= 2 else "FAKE"
    majority_count = max(real_votes_count, fake_votes_count)
    agreement_str = f"{majority_count}/{total_votes}"

    # Average ensemble probability
    avg_real_prob = float(np.mean(probabilities)) if probabilities else 0.5
    ensemble_confidence_pct = (avg_real_prob if final_decision == "REAL" else (1.0 - avg_real_prob)) * 100.0

    inference_time_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "status": "success",
        "final_decision": final_decision,
        "is_fake": final_decision == "FAKE",
        "is_real": final_decision == "REAL",
        "majority_vote": {
            "decision": final_decision,
            "agreement": agreement_str,
            "real_votes": real_votes_count,
            "fake_votes": fake_votes_count,
            "total_models": total_votes,
            "avg_real_probability": round(avg_real_prob, 4),
            "ensemble_confidence_pct": round(ensemble_confidence_pct, 2)
        },
        "models": model_results,
        "features": feature_dict,
        "audio_info": audio_info,
        "processing_time_ms": inference_time_ms
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
