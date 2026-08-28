#!/usr/bin/env python3
"""
Deepfake Audio Detector - Unified Application Launcher
Starts both the FastAPI ML Backend and Vite Frontend in development mode.
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python")

if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

def main():
    print("=" * 65)
    print("  🎙️  DEEPFAKE AUDIO DETECTION - ENSEMBLE SYSTEM  🎙️")
    print("=" * 65)
    print("Starting ML Backend & Modern Web Frontend...\n")

    # 1. Start Backend
    print("[1/2] Launching FastAPI Backend on http://localhost:8000 ...")
    backend_env = os.environ.copy()
    backend_env["KERAS_HOME"] = os.path.join(PROJECT_ROOT, ".keras")
    backend_env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    backend_env["PYTHONUNBUFFERED"] = "1"
    backend_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=PROJECT_ROOT,
        env=backend_env
    )

    # 2. Start Frontend
    print("[2/2] Launching Vite Frontend on http://localhost:5173 ...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR
    )

    time.sleep(2)
    url = "http://localhost:5173"
    print("\n" + "=" * 65)
    print(f" ✨ System is READY!")
    print(f" 🌐 Web Interface: {url}")
    print(f" 🔌 Backend API:   http://localhost:8000/docs")
    print("=" * 65)
    print("Press Ctrl+C to stop all servers.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    def cleanup(sig, frame):
        print("\nShutting down servers...")
        try:
            backend_proc.terminate()
            frontend_proc.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Wait for processes
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(None, None)

if __name__ == "__main__":
    main()

