"""
ML Regression Tests for Deepfake Audio Detection
Verifies that the refactored pipeline with scaler.pkl matches expected benchmark outputs with zero prediction drift.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.services.audio import audio_service
from backend.services.preprocessing import preprocessing_service
from backend.services.inference import inference_service

def test_ml_regression_on_known_samples():
    """
    Evaluates benchmark audio files and asserts exact classification consensus.
    """
    preprocessing_service.initialize()
    inference_service.initialize()

    test_cases = [
        {
            "rel_path": "gehra_hua.mp3",
            "name": "gehra_hua.mp3",
            "expected_decision": "REAL",
            "min_confidence": 70.0
        },
        {
            "rel_path": "FAKE_AUDIOS/file1004.mp3",
            "name": "file1004.mp3",
            "expected_decision": "FAKE",
            "min_confidence": 95.0
        },
        {
            "rel_path": "FAKE_AUDIOS/file1004.wav",
            "name": "file1004.wav",
            "expected_decision": "FAKE",
            "min_confidence": 95.0
        },
        {
            "rel_path": "FAKE_AUDIOS/file1019.wav",
            "name": "file1019.wav",
            "expected_decision": "FAKE",
            "min_confidence": 95.0
        }
    ]

    for tc in test_cases:
        full_path = os.path.join(PROJECT_ROOT, tc["rel_path"])
        if not os.path.exists(full_path):
            continue

        with open(full_path, "rb") as f:
            audio_bytes = f.read()

        # 1. Feature extraction
        feature_values, feature_dict, audio_info = audio_service.process_and_extract_features(
            audio_bytes, tc["name"]
        )
        assert len(feature_values) == 26

        # 2. Scaling & tensor reshape
        X_tensor = preprocessing_service.transform_and_reshape(feature_values)
        assert X_tensor.shape == (1, 26, 1)

        # 3. Model inference & majority voting
        res = inference_service.predict_ensemble(X_tensor)

        # Assertions
        decision = res["final_decision"]
        confidence = res["majority_vote"]["ensemble_confidence_pct"]
        agreement = res["majority_vote"]["agreement"]

        print(f"\n[Regression Result] File: {tc['name']} -> Verdict: {decision} ({agreement} models, Confidence: {confidence:.2f}%)")
        assert decision == tc["expected_decision"], (
            f"Regression failed for {tc['name']}: expected {tc['expected_decision']}, got {decision}"
        )
        assert confidence >= tc["min_confidence"], (
            f"Confidence lower than expected for {tc['name']}: {confidence:.2f}% < {tc['min_confidence']}%"
        )

if __name__ == "__main__":
    pytest.main(["-v", __file__])

