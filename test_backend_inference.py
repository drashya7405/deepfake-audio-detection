import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["KERAS_HOME"] = os.path.join(PROJECT_ROOT, ".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

from backend.app import model_manager, load_and_convert_audio

def test_inference():
    print("=" * 60)
    print("TESTING BACKEND ML ENGINE & 3-MODEL ENSEMBLE")
    print("=" * 60)

    # 1. Initialize models and scaler
    model_manager.initialize()
    assert len(model_manager.models) == 3, f"Expected 3 models, got {len(model_manager.models)}"
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
        feature_values, feature_dict, audio_info = load_and_convert_audio(audio_bytes, name)
        assert len(feature_values) == 26, f"Expected 26 features, got {len(feature_values)}"
        print(f"✓ Extracted 26 acoustic features. Audio duration: {audio_info['duration_seconds']}s")

        # Scale features
        X_scaled = model_manager.scale_features(feature_values)
        assert X_scaled.shape == (1, 26, 1), f"Expected shape (1, 26, 1), got {X_scaled.shape}"

        # Predict across 3 models
        votes = []
        probs = []
        for m_id in ["drashya", "devesh", "swayam"]:
            item = model_manager.models[m_id]
            model = item["model"]
            p = float(model.predict(X_scaled, verbose=0)[0][0])
            vote = "REAL" if p > 0.5 else "FAKE"
            conf = (p if vote == "REAL" else 1.0 - p) * 100.0
            votes.append(vote)
            probs.append(p)
            print(f"  ├─ {item['info']['name']}: {vote} (Prob: {p:.4f}, Confidence: {conf:.2f}%)")

        real_count = votes.count("REAL")
        fake_count = votes.count("FAKE")
        majority = "REAL" if real_count >= 2 else "FAKE"
        avg_prob = np.mean(probs)
        ensemble_conf = (avg_prob if majority == "REAL" else 1.0 - avg_prob) * 100.0

        print(f"  └─ MAJORITY VOTE: {majority} ({max(real_count, fake_count)}/3 models agreed, Ensemble Confidence: {ensemble_conf:.2f}%)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()
