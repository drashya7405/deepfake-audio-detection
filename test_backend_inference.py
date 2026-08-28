import os
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["KERAS_HOME"] = os.path.join(PROJECT_ROOT, ".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

from backend.services.audio import audio_service
from backend.services.preprocessing import preprocessing_service
from backend.services.inference import inference_service

def test_inference():
    print("=" * 60)
    print("TESTING BACKEND ML ENGINE & 3-MODEL ENSEMBLE")
    print("=" * 60)

    # 1. Initialize models and scaler
    preprocessing_service.initialize()
    inference_service.initialize()
    assert len(inference_service.models) == 3, f"Expected 3 models, got {len(inference_service.models)}"
    print("✓ All 3 models loaded into memory successfully.")

    # 2. Test test audio files
    test_files = [
        ("gehra_hua.mp3", "gehra_hua.mp3"),
        ("FAKE_AUDIOS/file1004.mp3", "file1004.mp3"),
        ("FAKE_AUDIOS/file1004.wav", "file1004.wav"),
        ("FAKE_AUDIOS/file1019.wav", "file1019.wav")
    ]

    for rel_path, name in test_files:
        full_path = os.path.join(os.path.dirname(__file__), rel_path)
        if not os.path.exists(full_path):
            print(f"Skipping {rel_path} (not found)")
            continue

        print(f"\n--- Testing File: {rel_path} ---")
        with open(full_path, "rb") as f:
            audio_bytes = f.read()

        # Extract features
        feature_values, feature_dict, audio_info = audio_service.process_and_extract_features(audio_bytes, name)
        assert len(feature_values) == 26, f"Expected 26 features, got {len(feature_values)}"
        print(f"✓ Extracted 26 acoustic features. Audio duration: {audio_info['duration_seconds']}s")

        # Scale features
        X_scaled = preprocessing_service.transform_and_reshape(feature_values)
        assert X_scaled.shape == (1, 26, 1), f"Expected shape (1, 26, 1), got {X_scaled.shape}"

        # Run ensemble inference
        res = inference_service.predict_ensemble(X_scaled)
        for m_id, item in res["models"].items():
            print(f"  ├─ {item['name']}: {item['prediction']} (Prob: {item['raw_probability']:.4f}, Confidence: {item['confidence_pct']:.2f}%)")

        majority = res["final_decision"]
        agreement = res["majority_vote"]["agreement"]
        ensemble_conf = res["majority_vote"]["ensemble_confidence_pct"]
        print(f"  └─ MAJORITY VOTE: {majority} ({agreement} models agreed, Ensemble Confidence: {ensemble_conf:.2f}%)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()
