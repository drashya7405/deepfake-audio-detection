"""
Deepfake Audio Detection - Memory-Optimized Sequential Inference Engine
Evaluates the 3 pre-trained deep learning models sequentially to operate within
Render Free's 512 MB RAM limit while preserving the full ensemble accuracy.

Render Free has a 512 MB memory limit, so the ensemble is evaluated sequentially to minimize peak RAM usage.
"""

import os
import sys
import gc
import logging
import threading
import resource
from typing import Dict, Any, List, Optional
import numpy as np

from backend.config import MODELS_DIR, MODEL_CONFIGS

import tensorflow as tf

logger = logging.getLogger("deepfake.inference")

def get_current_memory_mb() -> float:
    """
    Returns current maximum resident set size (RAM) in megabytes.
    Cross-platform support: macOS reports in bytes, Linux (Render) in kilobytes.
    """
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:
        return 0.0


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
    """
    Manages sequential deepfake detection inference.
    To adhere to Render Free's 512 MB memory ceiling, models are loaded one at a time,
    evaluated on the feature tensor, and immediately cleared from memory.
    """
    def __init__(self):
        self.model_configs = MODEL_CONFIGS
        self.is_ready: bool = False
        # Process-local lock to prevent concurrent requests from loading multiple models simultaneously
        self._inference_lock = threading.Lock()

    def initialize(self):
        """
        Verifies that all 3 model files exist on disk without permanently retaining all 3 in RAM.
        """
        missing_models = []
        for cfg in self.model_configs:
            model_path = os.path.join(MODELS_DIR, cfg["filename"])
            if not os.path.exists(model_path):
                missing_models.append(cfg["filename"])

        if missing_models:
            self.is_ready = False
            logger.error(f"Initialization failed: Missing model files: {missing_models}")
            raise FileNotFoundError(f"Missing required model files in {MODELS_DIR}: {missing_models}")

        self.is_ready = True
        logger.info(
            f"Inference service initialized: All {len(self.model_configs)} model files verified. "
            f"Sequential inference mode active (RAM footprint: {get_current_memory_mb():.1f} MB)."
        )

    def predict_single_model(self, model_cfg: Dict[str, Any], X_input: np.ndarray) -> float:
        """
        Loads exactly one Keras model, runs inference on the single sample tensor,
        extracts the scalar probability, and immediately releases all memory.
        """
        model_name = model_cfg["name"]
        model_path = os.path.join(MODELS_DIR, model_cfg["filename"])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        mem_before = get_current_memory_mb()
        logger.info(f"Loading model: {model_name} (RAM before load: {mem_before:.1f} MB)")

        model = None
        pred_tensor = None
        try:
            # 1. Load single model without compile overhead
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={'PositionalEncoding': PositionalEncoding}
            )

            # 2. Run inference without batch generator overhead
            logger.info(f"Running inference: {model_name}")
            pred_tensor = model(X_input, training=False)
            
            # Extract scalar float probability P(REAL)
            raw_prob = float(np.squeeze(pred_tensor.numpy()))
            
            logger.info(f"Completed inference: {model_name} -> P(REAL) = {raw_prob:.4f}")
            return raw_prob

        except Exception as e:
            logger.error(f"Inference error in model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Model evaluation failed for '{model_name}': {str(e)}")

        finally:
            # 3. Explicit memory cleanup after EACH model
            # Render Free has a 512 MB memory limit, so the ensemble is evaluated sequentially to minimize peak RAM usage.
            del model
            del pred_tensor
            tf.keras.backend.clear_session()
            gc.collect()

            mem_after = get_current_memory_mb()
            logger.info(f"Released model: {model_name} (RAM after release: {mem_after:.1f} MB)")

    def predict_ensemble(self, X_input: np.ndarray) -> Dict[str, Any]:
        """
        Sequentially executes all 3 models with memory cleanup between iterations,
        then calculates the majority voting consensus.
        """
        if not self.is_ready:
            self.initialize()

        # Enforce single-request model execution to avoid memory spikes under concurrent load
        with self._inference_lock:
            model_results = {}
            probabilities = []
            votes = []

            for cfg in self.model_configs:
                m_id = cfg["id"]
                raw_prob = self.predict_single_model(cfg, X_input)
                probabilities.append(raw_prob)

                # Binary Classification: 0 = FAKE, 1 = REAL
                model_vote = "REAL" if raw_prob > 0.5 else "FAKE"
                votes.append(model_vote)

                confidence_pct = (raw_prob if model_vote == "REAL" else (1.0 - raw_prob)) * 100.0

                model_results[m_id] = {
                    "id": m_id,
                    "name": cfg["name"],
                    "architecture": cfg["architecture"],
                    "prediction": model_vote,
                    "raw_probability": round(raw_prob, 4),
                    "real_probability_pct": round(raw_prob * 100, 2),
                    "fake_probability_pct": round((1.0 - raw_prob) * 100, 2),
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
