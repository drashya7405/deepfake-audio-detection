"""
Deepfake Audio Detection - Preprocessing & Feature Normalization Service
Manages StandardScaler loading and feature tensor preparation.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from backend.config import (
    FEATURE_COLUMNS,
    SCALER_PATH,
    FALLBACK_SCALER_PATH,
    DATASET_PATH
)

logger = logging.getLogger("deepfake.preprocessing")

class PreprocessingService:
    def __init__(self):
        self.scaler: StandardScaler = None
        self.is_loaded: bool = False

    def initialize(self):
        if self.is_loaded and self.scaler is not None:
            return

        # 1. Try loading dedicated scaler artifact
        for path in [SCALER_PATH, FALLBACK_SCALER_PATH]:
            if os.path.exists(path):
                try:
                    self.scaler = joblib.load(path)
                    if hasattr(self.scaler, "mean_") and len(self.scaler.mean_) == len(FEATURE_COLUMNS):
                        self.is_loaded = True
                        logger.info(f"Loaded feature scaler from {path} (26 dimensions).")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load scaler from {path}: {e}")

        # 2. Fallback: Fit on DATASET-balanced.csv if available
        if os.path.exists(DATASET_PATH):
            logger.info(f"Fitting StandardScaler from training dataset at {DATASET_PATH}...")
            df = pd.read_csv(DATASET_PATH)
            self.scaler = StandardScaler()
            self.scaler.fit(df[FEATURE_COLUMNS].astype(float))
            
            # Save artifact for future fast startup
            try:
                os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
                joblib.dump(self.scaler, SCALER_PATH)
                logger.info(f"Exported fitted scaler to {SCALER_PATH}.")
            except Exception as e:
                logger.warning(f"Could not persist scaler to {SCALER_PATH}: {e}")
                
            self.is_loaded = True
            return

        raise FileNotFoundError(
            f"Scaler artifact not found at {SCALER_PATH} and fallback dataset missing at {DATASET_PATH}"
        )

    def validate_features(self, feature_values: List[float]) -> None:
        if len(feature_values) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Feature vector length mismatch. Expected exactly {len(FEATURE_COLUMNS)} features, received {len(feature_values)}."
            )
        for i, val in enumerate(feature_values):
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Feature at index {i} ('{FEATURE_COLUMNS[i]}') contains NaN or Inf.")

    def transform_and_reshape(self, feature_values: List[float]) -> np.ndarray:
        if not self.is_loaded or self.scaler is None:
            self.initialize()

        self.validate_features(feature_values)

        # Convert to DataFrame with feature names to match scaler training schema cleanly
        raw_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS, dtype=np.float64)
        scaled_features = self.scaler.transform(raw_df)

        # Reshape to sequence: shape (1, 26, 1) for 1D CNN / RNN sequence layers
        reshaped_tensor = np.reshape(scaled_features, (1, len(FEATURE_COLUMNS), 1))
        return reshaped_tensor


preprocessing_service = PreprocessingService()

