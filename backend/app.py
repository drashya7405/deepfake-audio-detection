"""
Deepfake Audio Detection - FastAPI Production Server
Exposes secure REST endpoints for audio feature extraction, multi-model ensemble inference, and health monitoring.
Optimized for 512 MB memory-constrained hosting environments (Render Free).
"""

import os
import sys
import time
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from backend.config import (
    ALLOWED_ORIGINS,
    LOG_LEVEL,
    PROJECT_ROOT,
    FEATURE_COLUMNS,
    MODEL_CONFIGS,
    KNOWN_SAMPLES,
    RATE_LIMIT_PER_MINUTE
)
from backend.services.rate_limiter import InMemoryRateLimiter
from backend.services.preprocessing import preprocessing_service
from backend.services.inference import inference_service, get_current_memory_mb
from backend.services.audio import audio_service

# --- 1. Structured Logging Setup ---
numeric_log_level = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=numeric_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("deepfake.app")

# Rate limiter instance
rate_limiter = InMemoryRateLimiter(requests_per_minute=RATE_LIMIT_PER_MINUTE)

# --- 2. Application Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Deepfake Audio Detection Backend (Sequential Inference Mode)...")
    try:
        preprocessing_service.initialize()
        inference_service.initialize()
        logger.info(
            f"RSS | Backend services initialized and ready. Current Live RSS: {get_current_memory_mb():.2f} MB "
            f"(TensorFlow in sys.modules: {'tensorflow' in sys.modules})."
        )
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}", exc_info=True)
    yield
    logger.info("Shutting down Deepfake Audio Detection Backend.")

# --- 3. FastAPI App Initialization ---
app = FastAPI(
    title="Deepfake Audio Detection API",
    description="Production-grade AI Audio Forensics & Sequential Ensemble Inference API",
    version="1.0.0",
    lifespan=lifespan
)

# --- 4. Configurable CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- 5. Health and Readiness Probes ---

@app.get("/api/health", tags=["Monitoring"])
async def health_check():
    """
    Liveness probe: Confirms that the server process is responsive.
    """
    return {
        "status": "ok",
        "service": "deepfake-audio-detector"
    }


@app.get("/api/ready", tags=["Monitoring"])
async def readiness_check():
    """
    Readiness probe: Verifies that the scaler is loaded and all 3 model files exist.
    Optimized for Render Free (512 MB RAM): does not require all 3 models to remain loaded in memory.
    """
    is_scaler_ready = preprocessing_service.is_loaded and preprocessing_service.scaler is not None
    is_inference_ready = inference_service.is_ready

    if is_scaler_ready and is_inference_ready:
        return {
            "status": "ready",
            "models_available": len(MODEL_CONFIGS),
            "required_models": len(MODEL_CONFIGS),
            "scaler_loaded": True,
            "mode": "sequential_inference"
        }

    logger.warning(f"Readiness check failed (scaler_ready={is_scaler_ready}, inference_ready={is_inference_ready})")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "status": "not_ready",
            "models_available": len(MODEL_CONFIGS) if is_inference_ready else 0,
            "required_models": len(MODEL_CONFIGS),
            "scaler_loaded": is_scaler_ready,
            "message": "Inference engine is still initializing or required model files are missing."
        }
    )


@app.get("/api/models-info", tags=["Metadata"])
async def get_models_info():
    """
    Returns architectural specifications and validation split metrics for the 3 evaluated models.
    """
    return {
        "models": MODEL_CONFIGS,
        "features_count": len(FEATURE_COLUMNS),
        "features_list": FEATURE_COLUMNS,
        "ensemble_method": "Sequential Majority Voting (>=2 of 3) with Soft-Probability Calibration"
    }


@app.get("/api/samples", tags=["Samples"])
async def get_sample_audios():
    """
    Returns pre-packaged audio clips available for immediate benchmark testing.
    """
    available_samples = []
    for item in KNOWN_SAMPLES:
        full_path = os.path.join(PROJECT_ROOT, item["path"])
        if os.path.exists(full_path):
            available_samples.append({
                "id": item["id"],
                "name": item["name"],
                "path": item["path"],
                "tag": item["tag"],
                "type": item["type"],
                "size_kb": round(os.path.getsize(full_path) / 1024, 1)
            })

    return {"samples": available_samples}


@app.get("/api/sample-audio/{sample_id}", tags=["Samples"])
async def get_sample_audio_stream(sample_id: str):
    """
    Streams audio bytes of a demo sample for in-browser playback.
    """
    sample_map = {item["id"]: item["path"] for item in KNOWN_SAMPLES}
    if sample_id not in sample_map:
        raise HTTPException(status_code=404, detail="Sample audio not found.")

    rel_path = sample_map[sample_id]
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio file missing on disk.")

    media_type = "audio/mpeg" if full_path.endswith(".mp3") else "audio/wav"
    return FileResponse(full_path, media_type=media_type, filename=os.path.basename(full_path))


# --- 6. Prediction Endpoint ---

@app.post("/api/predict", tags=["Inference"])
async def predict_audio(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None)
):
    """
    Production Deepfake Audio Prediction Pipeline (Memory-Optimized):
    1. Enforces rate limits.
    2. Ingests and safely decodes audio stream.
    3. Extracts 26 acoustic features via Librosa with prompt buffer release.
    4. Normalizes features with pre-computed StandardScaler.
    5. Evaluates the 3 deep learning models sequentially (one-by-one with session clearing).
    6. Calculates majority voting consensus and ensemble confidence.
    """
    req_id = uuid.uuid4().hex[:8]

    # 1. Rate limiting check
    rate_limiter.check_rate_limit(request)

    # 2. Check service readiness
    if not preprocessing_service.is_loaded or not inference_service.is_ready:
        logger.warning(f"[Req {req_id}] Inference requested before services were fully ready. Attempting initialization...")
        try:
            preprocessing_service.initialize()
            inference_service.initialize()
        except Exception as err:
            logger.error(f"[Req {req_id}] Inference blocked - service initialization failed: {err}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference models are currently unavailable. Please try again shortly."
            )

    start_time = time.time()
    audio_bytes = None
    filename = "audio_input"

    # Option A: Pre-packaged sample requested
    if sample_id:
        sample_map = {item["id"]: item["path"] for item in KNOWN_SAMPLES}
        if sample_id in sample_map:
            sample_rel_path = sample_map[sample_id]
            sample_full_path = os.path.join(PROJECT_ROOT, sample_rel_path)
            if os.path.exists(sample_full_path):
                with open(sample_full_path, "rb") as f:
                    audio_bytes = f.read()
                filename = os.path.basename(sample_rel_path)
        else:
            raise HTTPException(status_code=404, detail="Selected sample ID does not exist.")

    # Option B: Uploaded audio file
    if audio_bytes is None and file is not None:
        audio_bytes = await file.read()
        filename = file.filename or "uploaded_audio"

    if not audio_bytes or len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="No audio file or valid sample provided.")

    logger.info(f"[Req {req_id}] Received audio analysis request for '{filename}' ({len(audio_bytes)/1024:.1f} KB)")

    # 3. Audio Decoding & 26-Feature Extraction
    feature_values, feature_dict, audio_info = audio_service.process_and_extract_features(audio_bytes, filename)

    # 4. Feature Scaling & Tensor Preparation
    X_input = preprocessing_service.transform_and_reshape(feature_values)

    # 5. Sequential 3-Model Inference & Majority Voting
    # Render Free has a 512 MB memory limit, so the ensemble is evaluated sequentially to minimize peak RAM usage.
    ensemble_results = inference_service.predict_ensemble(X_input, request_id=req_id)

    total_processing_ms = round((time.time() - start_time) * 1000, 1)

    logger.info(
        f"[Req {req_id}] Analysis complete for '{filename}': Verdict={ensemble_results['final_decision']} "
        f"({ensemble_results['majority_vote']['agreement']} models, {ensemble_results['majority_vote']['ensemble_confidence_pct']}%) "
        f"in {total_processing_ms} ms (Final Live RSS: {get_current_memory_mb():.2f} MB)"
    )

    return {
        "status": "success",
        "final_decision": ensemble_results["final_decision"],
        "is_fake": ensemble_results["is_fake"],
        "is_real": ensemble_results["is_real"],
        "majority_vote": ensemble_results["majority_vote"],
        "models": ensemble_results["models"],
        "features": feature_dict,
        "audio_info": audio_info,
        "processing_time_ms": total_processing_ms
    }


if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=False)
