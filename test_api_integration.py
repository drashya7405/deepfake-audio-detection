import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["KERAS_HOME"] = os.path.join(PROJECT_ROOT, ".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_api():
    print("=" * 60)
    print("TESTING FASTAPI REST API ENDPOINTS")
    print("=" * 60)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print("✓ GET /api/health passed:", health_data)
    assert health_data["total_models"] == 3

    # 2. Models info
    res = client.get("/api/models-info")
    assert res.status_code == 200, f"Models info failed: {res.text}"
    info_data = res.json()
    print(f"✓ GET /api/models-info passed ({len(info_data['models'])} models, {info_data['features_count']} features)")

    # 3. Samples list
    res = client.get("/api/samples")
    assert res.status_code == 200, f"Samples endpoint failed: {res.text}"
    samples_data = res.json()
    print(f"✓ GET /api/samples passed ({len(samples_data['samples'])} samples available)")

    # 4. Predict via sample_id
    res = client.post("/api/predict", data={"sample_id": "sample_real_1"})
    assert res.status_code == 200, f"Predict sample failed: {res.text}"
    pred_data = res.json()
    print("✓ POST /api/predict (sample_id='sample_real_1'):", pred_data["final_decision"], f"({pred_data['majority_vote']['agreement']} models agree, {pred_data['majority_vote']['ensemble_confidence_pct']}%)")
    assert pred_data["final_decision"] == "REAL"

    # 5. Predict via uploaded file
    with open(os.path.join(PROJECT_ROOT, "FAKE_AUDIOS", "file1004.mp3"), "rb") as f:
        res = client.post("/api/predict", files={"file": ("file1004.mp3", f, "audio/mpeg")})
    assert res.status_code == 200, f"Predict upload failed: {res.text}"
    upload_pred = res.json()
    print("✓ POST /api/predict (file upload='file1004.mp3'):", upload_pred["final_decision"], f"({upload_pred['majority_vote']['agreement']} models agree, {upload_pred['majority_vote']['ensemble_confidence_pct']}%)")
    assert upload_pred["final_decision"] == "FAKE"

    print("\n" + "=" * 60)
    print("ALL API INTEGRATION TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_api()

