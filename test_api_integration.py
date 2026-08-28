import os
import sys
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.app import app

def test_api():
    print("=" * 60)
    print("TESTING FASTAPI REST API ENDPOINTS")
    print("=" * 60)

    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/api/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        print(f"✓ GET /api/health passed: {res.json()}")

        # 2. Readiness check
        res = client.get("/api/ready")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        print(f"✓ GET /api/ready passed: {res.json()}")

        # 3. Models info
        res = client.get("/api/models-info")
        assert res.status_code == 200
        info = res.json()
        print(f"✓ GET /api/models-info passed ({len(info['models'])} models, {info['features_count']} features)")

        # 4. Samples list
        res = client.get("/api/samples")
        assert res.status_code == 200
        samples = res.json()["samples"]
        print(f"✓ GET /api/samples passed ({len(samples)} samples available)")

        # 5. Predict with sample_id
        res = client.post("/api/predict", data={"sample_id": "sample_real_1"})
        assert res.status_code == 200
        pred = res.json()
        print(f"✓ POST /api/predict (sample_id='sample_real_1'): {pred['final_decision']} ({pred['majority_vote']['agreement']} models agree, {pred['majority_vote']['ensemble_confidence_pct']}%)")

        # 6. Predict with file upload
        test_file = os.path.join(PROJECT_ROOT, "FAKE_AUDIOS", "file1004.mp3")
        with open(test_file, "rb") as f:
            res = client.post("/api/predict", files={"file": ("file1004.mp3", f, "audio/mpeg")})
        assert res.status_code == 200
        pred = res.json()
        print(f"✓ POST /api/predict (file upload='file1004.mp3'): {pred['final_decision']} ({pred['majority_vote']['agreement']} models agree, {pred['majority_vote']['ensemble_confidence_pct']}%)")

    print("\n" + "=" * 60)
    print("ALL API INTEGRATION TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_api()
