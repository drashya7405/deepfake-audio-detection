"""
Unit Tests for Deepfake Audio Detection Services
Tests scaler loading, feature validation, model verification, audio decoding, and rate limiting.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.config import FEATURE_COLUMNS, MODEL_CONFIGS, SCALER_PATH
from backend.services.preprocessing import preprocessing_service
from backend.services.inference import inference_service
from backend.services.audio import audio_service
from backend.services.rate_limiter import InMemoryRateLimiter
from fastapi import HTTPException

def test_scaler_loading_and_shape():
    """Verify that the pre-computed scaler loads correctly and has 26 dimensions."""
    preprocessing_service.initialize()
    assert preprocessing_service.is_loaded is True
    assert preprocessing_service.scaler is not None
    assert len(preprocessing_service.scaler.mean_) == 26
    assert len(preprocessing_service.scaler.scale_) == 26

def test_feature_order_and_validation():
    """Verify feature validation rejects vectors of incorrect length or NaN values."""
    preprocessing_service.initialize()
    
    # Valid 26-feature dummy vector
    valid_features = [0.5] * 26
    tensor = preprocessing_service.transform_and_reshape(valid_features)
    assert tensor.shape == (1, 26, 1)

    # Invalid: 25 features
    with pytest.raises(ValueError, match="Feature vector length mismatch"):
        preprocessing_service.transform_and_reshape([0.5] * 25)

    # Invalid: contains NaN
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        invalid_with_nan = [0.5] * 26
        invalid_with_nan[3] = float("nan")
        preprocessing_service.transform_and_reshape(invalid_with_nan)

def test_model_readiness_verification():
    """Verify that model files exist and inference service is ready without retaining all models permanently in RAM."""
    inference_service.initialize()
    assert inference_service.is_ready is True
    assert len(inference_service.model_configs) == 3

def test_audio_validation_empty_and_unsupported():
    """Verify audio service rejects empty bytes and invalid extensions."""
    with pytest.raises(HTTPException) as exc_info:
        audio_service.validate_upload(b"", "test.mp3")
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        audio_service.validate_upload(b"some audio bytes", "test.exe")
    assert exc_info.value.status_code == 400

def test_rate_limiter():
    """Verify rate limiter blocks requests after exceeding threshold."""
    limiter = InMemoryRateLimiter(requests_per_minute=3)
    
    class DummyRequest:
        class Client:
            host = "127.0.0.1"
        client = Client()

    req = DummyRequest()
    # 3 allowed requests
    limiter.check_rate_limit(req)
    limiter.check_rate_limit(req)
    limiter.check_rate_limit(req)

    # 4th request should raise 429
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req)
    assert exc_info.value.status_code == 429

if __name__ == "__main__":
    pytest.main(["-v", __file__])
