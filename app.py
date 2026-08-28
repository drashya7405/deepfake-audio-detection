"""
Root entry point for Native Python Deployments (Hugging Face Spaces, Render, Railway)
Launches the production FastAPI server using the host platform's $PORT environment variable.
"""

import os
import sys
import uvicorn

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=False)

