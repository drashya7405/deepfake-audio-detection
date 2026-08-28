"""
Deepfake Audio Detection - Inference & Ensemble Engine
Loads the 3 pre-trained deep learning models and computes majority voting consensus.
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure environment is configured before importing tensorflow
from backend.config import MODELS_DIR, MODEL_CONFIGS

import tensorflow as tf

logger = logging.getLogger("deepfake.inference")

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Custom Sinusoidal Positional Encoding layer required by Swayam's CNN-Transformer model.
    """
    def __init__(self, max_steps: int = 1000, max_dims: int = 512, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims
        
        dims = max_dims
        if dims % 2 == 1:
            dims += 1 
        
        p, i = np.meshgrid(np.arange(max_steps), np.arange(dims // 2))
        pos_emb = np.empty((1, max_steps, dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / dims)).T
        self.positional_encoding = tf.constant(pos_emb, dtype=tf.float32)

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_encoding[:, :shape[1], :shape[2]]
        
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "max_steps": self.max_steps,
            "max_dims": self.max_dims
        })
        return config


class InferenceService:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.is_loaded: bool = False

    def initialize(self):
        if self.is_loaded:
            return

        logger.info("Loading 3 Deep Learning models into memory...")
        loaded_count = 0

        for cfg in MODEL_CONFIGS:
            model_path = os.path.join(MODELS_DIR, cfg["filename"])
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                continue

            try:
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects={'PositionalEncoding': PositionalEncoding}
                )
                self.models[cfg["id"]] = {
                    "model": model,
                    "info": cfg
                }
                loaded_count += 1
                logger.info(f"✓ Loaded {cfg['name']} from {cfg['filename']}.")
            except Exception as e:
                logger.error(f"✗ Failed to load model {cfg['name']} from {model_path}: {e}", exc_info=True)

        if loaded_count == len(MODEL_CONFIGS):
            self.is_loaded = True
            logger.info(f"Ensemble ready: All {loaded_count}/3 models loaded successfully.")
        else:
            logger.warning(f"Incomplete ensemble: {loaded_count}/{len(MODEL_CONFIGS)} models loaded.")

    def predict_ensemble(self, X_input: np.ndarray) -> Dict[str, Any]:
        """
        Runs feature tensor (shape: 1, 26, 1) through all 3 models and computes majority voting consensus.
        """
        if not self.is_loaded or len(self.models) < 3:
            self.initialize()
            if len(self.models) == 0:
                raise RuntimeError("No deep learning models could be loaded into memory.")

        model_results = {}
        probabilities = []
        votes = []

        model_order = ["drashya", "devesh", "swayam"]

        for m_id in model_order:
            if m_id not in self.models:
                continue

            item = self.models[m_id]
            model = item["model"]
            info = item["info"]

            # Predict probability scalar
            raw_pred = float(model.predict(X_input, verbose=0)[0][0])
            probabilities.append(raw_pred)

            # Binary Label Encoding: 0 = FAKE, 1 = REAL
            # raw_pred represents P(REAL)
            model_vote = "REAL" if raw_pred > 0.5 else "FAKE"
            votes.append(model_vote)

            confidence_pct = (raw_pred if model_vote == "REAL" else (1.0 - raw_pred)) * 100.0

            model_results[m_id] = {
                "id": m_id,
                "name": info["name"],
                "architecture": info["architecture"],
                "prediction": model_vote,
                "raw_probability": round(raw_pred, 4),
                "real_probability_pct": round(raw_pred * 100, 2),
                "fake_probability_pct": round((1.0 - raw_pred) * 100, 2),
                "confidence_pct": round(confidence_pct, 2)
            }

        # Majority voting calculation
        real_votes_count = votes.count("REAL")
        fake_votes_count = votes.count("FAKE")
        total_votes = len(votes)

        final_decision = "REAL" if real_votes_count >= 2 else "FAKE"
        majority_count = max(real_votes_count, fake_votes_count)
        agreement_str = f"{majority_count}/{total_votes}"

        avg_real_prob = float(np.mean(probabilities)) if probabilities else 0.5
        ensemble_confidence_pct = (avg_real_prob if final_decision == "REAL" else (1.0 - avg_real_prob)) * 100.0

        return {
            "final_decision": final_decision,
            "is_fake": final_decision == "FAKE",
            "is_real": final_decision == "REAL",
            "majority_vote": {
                "decision": final_decision,
                "agreement": agreement_str,
                "real_votes": real_votes_count,
                "fake_votes": fake_votes_count,
                "total_models": total_votes,
                "avg_real_probability": round(avg_real_prob, 4),
                "ensemble_confidence_pct": round(ensemble_confidence_pct, 2)
            },
            "models": model_results
        }


inference_service = InferenceService()

