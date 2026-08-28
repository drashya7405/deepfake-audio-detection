# Use lightweight official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_HOME=/app/.keras \
    PORT=8000

# Install system audio dependencies (ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application files, models, and dataset
COPY backend/ /app/backend/
COPY BestModels/ /app/BestModels/
COPY DATASET-balanced.csv /app/DATASET-balanced.csv
COPY gehra_hua.mp3 /app/gehra_hua.mp3
COPY FAKE_AUDIOS/ /app/FAKE_AUDIOS/

# Expose port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

