"""
Deepfake Audio Detection - Audio Processing & Feature Extraction Service
Handles audio validation, format decoding, Librosa acoustic feature extraction,
and proactive memory cleanup for 512 MB constrained environments.
Reuses single-pass STFT spectrograms to eliminate redundant memory allocations.
"""

import os
import gc
import time
import uuid
import tempfile
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from fastapi import HTTPException

from backend.config import (
    FEATURE_COLUMNS,
    SUPPORTED_AUDIO_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    MAX_AUDIO_DURATION_SECONDS
)

logger = logging.getLogger("deepfake.audio")

class AudioProcessingService:
    def validate_upload(self, audio_bytes: bytes, filename: str) -> str:
        """
        Validates uploaded file size, non-emptiness, and file extension.
        Returns normalized lowercase extension.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty (0 bytes).")

        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file size ({len(audio_bytes) / (1024*1024):.1f} MB) exceeds maximum allowed size of {MAX_UPLOAD_MB} MB."
            )

        # Extract and sanitize extension
        ext = os.path.splitext(filename)[1].lower() if filename else ".wav"
        if not ext or ext not in SUPPORTED_AUDIO_EXTENSIONS:
            allowed_str = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format '{ext}'. Allowed formats: {allowed_str}"
            )

        return ext

    def process_and_extract_features(
        self,
        audio_bytes: bytes,
        original_filename: str
    ) -> Tuple[List[float], Dict[str, float], Dict[str, Any]]:
        """
        Ingests raw audio bytes, converts to mono PCM, validates duration,
        and computes the 26 acoustic features.
        Uses single-pass STFT calculation to prevent 4 redundant multi-megabyte FFT allocations.
        Guarantees cleanup of all temporary files and in-memory arrays.
        """
        ext = self.validate_upload(audio_bytes, original_filename)

        temp_in_path = None
        temp_wav_path = None
        y = None
        sr = None
        S = None
        S_power = None
        melspec = None
        mfccs = None

        try:
            # 1. Write incoming bytes to secure temporary file
            with tempfile.NamedTemporaryFile(delete=False, prefix=f"input_{uuid.uuid4().hex[:8]}_", suffix=ext) as f_in:
                f_in.write(audio_bytes)
                temp_in_path = f_in.name

            # 2. Decode audio waveform
            duration = 0.0

            try:
                # Primary attempt with Librosa / soundfile
                y, sr = librosa.load(temp_in_path, sr=None, mono=True)
                duration = float(librosa.get_duration(y=y, sr=sr))
            except Exception as librosa_err:
                logger.debug(f"Direct librosa.load failed ({librosa_err}), attempting pydub conversion...")
                # Fallback: Convert via pydub / ffmpeg to clean 16-bit WAV
                with tempfile.NamedTemporaryFile(delete=False, prefix=f"conv_{uuid.uuid4().hex[:8]}_", suffix=".wav") as f_conv:
                    temp_wav_path = f_conv.name

                audio_seg = AudioSegment.from_file(temp_in_path)
                audio_seg = audio_seg.set_channels(1)  # Force mono
                audio_seg.export(temp_wav_path, format="wav")
                del audio_seg

                y, sr = librosa.load(temp_wav_path, sr=None, mono=True)
                duration = float(librosa.get_duration(y=y, sr=sr))

            # 3. Audio Validation Checks
            if y is None or len(y) == 0:
                raise HTTPException(status_code=422, detail="Audio file could not be decoded or contains empty audio stream.")

            if np.all(y == 0):
                raise HTTPException(status_code=422, detail="Audio file contains only silent/null audio signal.")

            if duration > MAX_AUDIO_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration ({duration:.1f}s) exceeds maximum allowed duration of {MAX_AUDIO_DURATION_SECONDS:.0f} seconds."
                )

            # 4. Extract the exact 26 Acoustic Features in strict sequence
            # Single-pass STFT calculation eliminates 4 redundant Fourier transforms and multi-spectrogram RAM allocations
            S = np.abs(librosa.stft(y=y))
            S_power = S ** 2

            # (1) Chroma STFT (computed from power spectrogram)
            chroma_stft = float(np.mean(librosa.feature.chroma_stft(S=S_power, sr=sr)))

            # (2) RMS Energy (computed from waveform)
            rms = float(np.mean(librosa.feature.rms(y=y)))

            # (3) Spectral Centroid (computed from magnitude spectrogram)
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))

            # (4) Spectral Bandwidth (computed from magnitude spectrogram)
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))

            # (5) Spectral Rolloff (computed from magnitude spectrogram)
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))

            # (6) Zero Crossing Rate (computed from waveform)
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

            # (7-26) 20 MFCCs (computed from Mel-spectrogram of power spectrogram)
            melspec = librosa.feature.melspectrogram(S=S_power, sr=sr)
            mfccs = librosa.feature.mfcc(S=librosa.power_to_db(melspec), sr=sr, n_mfcc=20)
            mfcc_means = [float(x) for x in np.mean(mfccs, axis=1)]

            feature_values = [
                chroma_stft, rms, spectral_centroid, spectral_bandwidth,
                rolloff, zero_crossing_rate, *mfcc_means
            ]

            feature_dict = dict(zip(FEATURE_COLUMNS, [round(v, 6) for v in feature_values]))

            audio_info = {
                "duration_seconds": round(duration, 2),
                "sample_rate": int(sr),
                "samples_count": len(y),
                "original_filename": original_filename or "uploaded_audio",
                "format": ext.replace(".", "").upper()
            }

            return feature_values, feature_dict, audio_info

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error extracting features from {original_filename}: {e}", exc_info=True)
            raise HTTPException(status_code=422, detail=f"Audio processing error: Failed to decode audio file. ({str(e)})")

        finally:
            # Guaranteed cleanup of all temp files & memory structures
            del y
            del sr
            del S
            del S_power
            del melspec
            del mfccs
            gc.collect()

            for temp_f in [temp_in_path, temp_wav_path]:
                if temp_f and os.path.exists(temp_f):
                    try:
                        os.unlink(temp_f)
                    except Exception as err:
                        logger.warning(f"Failed removing temp file {temp_f}: {err}")


audio_service = AudioProcessingService()
