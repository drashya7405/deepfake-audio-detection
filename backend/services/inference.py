"""
Deepfake Audio Detection - Memory-Optimized Sequential Inference Engine
Evaluates the 3 pre-trained deep learning models sequentially to operate comfortably within
Render Free's 512 MB RAM limit while preserving 100% full ensemble accuracy.

Primary Production Engine: Standalone FP32 TensorFlow Lite / LiteRT runtime.
Rollback / Reference Engine: Keras (.h5) lazy-loaded sequential evaluation.
"""

import os
import sys
import gc
import uuid
import logging
import threading
import resource
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from backend.config import MODELS_DIR, MODEL_CONFIGS

logger = logging.getLogger("deepfake.inference")

def get_live_rss_mb() -> float:
    """
    Returns current instantaneous physical resident set size (Live RSS) in Megabytes.
    Uses Linux /proc/self/status VmRSS on Render, task_vm_info phys_footprint on macOS, or psutil.
    Does not rely on ru_maxrss (which only reports lifetime high-water peak).
    """
    # 1. Try psutil if available
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass

    # 2. Linux /proc/self/status VmRSS (Render Free environment)
    if os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass

    # 3. macOS / Darwin physical footprint (task_vm_info.phys_footprint)
    if sys.platform == "darwin":
        try:
            import ctypes
            TASK_VM_INFO = 22
            class task_vm_info(ctypes.Structure):
                _fields_ = [
                    ('virtual_size', ctypes.c_uint64),
                    ('region_count', ctypes.c_int32),
                    ('page_size', ctypes.c_int32),
                    ('resident_size', ctypes.c_uint64),
                    ('resident_size_peak', ctypes.c_uint64),
                    ('device', ctypes.c_uint64),
                    ('device_peak', ctypes.c_uint64),
                    ('internal', ctypes.c_uint64),
                    ('internal_peak', ctypes.c_uint64),
                    ('external', ctypes.c_uint64),
                    ('external_peak', ctypes.c_uint64),
                    ('reusable', ctypes.c_uint64),
                    ('reusable_peak', ctypes.c_uint64),
                    ('purgeable_volatile_pmap', ctypes.c_uint64),
                    ('purgeable_volatile_resident', ctypes.c_uint64),
                    ('purgeable_volatile_virtual', ctypes.c_uint64),
                    ('compressed', ctypes.c_uint64),
                    ('compressed_peak', ctypes.c_uint64),
                    ('compressed_lifetime', ctypes.c_uint64),
                    ('phys_footprint', ctypes.c_uint64),
                ]
            info = task_vm_info()
            count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
            libc = ctypes.CDLL(None)
            res = libc.task_info(libc.mach_task_self(), TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count))
            if res == 0:
                return float(info.phys_footprint) / (1024.0 * 1024.0)
        except Exception:
            pass

    # 4. Fallback: resource.getrusage
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0
    except Exception:
        return 0.0


def get_current_memory_mb() -> float:
    """Alias for backward compatibility, returning current live RSS in MB."""
    return get_live_rss_mb()


# ---------------------------------------------------------------------------
# 1. Standalone TFLite / LiteRT Runtime Loader (Primary Production Path)
# ---------------------------------------------------------------------------

def create_tflite_interpreter(model_path: str):
    """
    Instantiates a TFLite interpreter from the best available runtime.
    Order of preference:
      1. ai_edge_litert.interpreter (Google LiteRT standalone)
      2. tflite_runtime.interpreter (TFLite runtime standalone)
      3. tensorflow.lite.python.interpreter (TensorFlow bundled fallback)
    """
    # Option A: Google LiteRT standalone
    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    except ImportError:
        pass

    # Option B: tflite-runtime standalone
    try:
        from tflite_runtime.interpreter import Interpreter
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    except ImportError:
        pass

    # Option C: TensorFlow bundled interpreter fallback
    from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


# ---------------------------------------------------------------------------
# 2. Lazy Keras Reference Engine (Rollback / Verification Reference Path)
# ---------------------------------------------------------------------------

_tf = None
_PositionalEncoding = None

def get_tf_and_custom_objects(request_id: Optional[str] = None):
    """
    Lazily imports TensorFlow on demand if Keras fallback inference is requested.
    """
    global _tf, _PositionalEncoding
    req_tag = f"Req {request_id}" if request_id else "Init"
    if _tf is None:
        rss_before = get_live_rss_mb()
        logger.info(f"[{req_tag}] RSS | Before TensorFlow lazy import: {rss_before:.2f} MB (sys.modules: {'tensorflow' in sys.modules})")
        import tensorflow as tf

        @tf.keras.utils.register_keras_serializable()
        class PositionalEncoding(tf.keras.layers.Layer):
            def __init__(self, max_steps: int = 1000, max_dims: int = 512, **kwargs):
                super(PositionalEncoding, self).__init__(**kwargs)
                self.max_steps = max_steps
                self.max_dims = max_dims
                dims = max_dims if max_dims % 2 == 0 else max_dims + 1
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

        _tf = tf
        _PositionalEncoding = PositionalEncoding
        rss_after = get_live_rss_mb()
        logger.info(f"[{req_tag}] RSS | After TensorFlow lazy import: {rss_after:.2f} MB (Delta: +{rss_after - rss_before:.2f} MB, TF {tf.__version__})")

    return _tf, _PositionalEncoding


# ---------------------------------------------------------------------------
# 3. Main Sequential Inference Service
# ---------------------------------------------------------------------------

class InferenceService:
    """
    Manages sequential deepfake detection inference.
    Evaluates models one-by-one using FP32 TFLite to operate well within Render Free's 512 MB ceiling.
    """
    def __init__(self):
        self.model_configs = MODEL_CONFIGS
        self.is_ready: bool = False
        # Process-local lock ensures only one request evaluates models at a time
        self._inference_lock = threading.Lock()

    def initialize(self):
        """
        Verifies that model artifacts exist on disk without pre-loading into RAM.
        Prefers .tflite models with fallback verification for .h5 models.
        """
        missing_models = []
        for cfg in self.model_configs:
            tflite_file = cfg.get("tflite_filename", cfg["filename"].replace(".h5", ".tflite"))
            h5_file = cfg["filename"]
            
            tflite_path = os.path.join(MODELS_DIR, tflite_file)
            h5_path = os.path.join(MODELS_DIR, h5_file)

            if not os.path.exists(tflite_path) and not os.path.exists(h5_path):
                missing_models.append(f"{tflite_file} / {h5_file}")

        if missing_models:
            self.is_ready = False
            logger.error(f"Initialization failed: Missing model files: {missing_models}")
            raise FileNotFoundError(f"Missing required model files in {MODELS_DIR}: {missing_models}")

        self.is_ready = True
        startup_rss = get_live_rss_mb()
        logger.info(
            f"RSS | Inference service initialized: All {len(self.model_configs)} models verified. "
            f"Sequential FP32 TFLite mode active (Startup Live RSS: {startup_rss:.2f} MB)."
        )

    def predict_single_model_tflite(
        self,
        model_cfg: Dict[str, Any],
        X_input: np.ndarray,
        request_id: Optional[str] = None
    ) -> float:
        """
        Loads one TFLite model, invokes inference on tensor (1, 26, 1), and releases memory.
        """
        req_tag = f"Req {request_id}" if request_id else "Inference"
        model_name = model_cfg["name"]
        tflite_file = model_cfg.get("tflite_filename", model_cfg["filename"].replace(".h5", ".tflite"))
        tflite_path = os.path.join(MODELS_DIR, tflite_file)

        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"TFLite model file not found: {tflite_path}")

        rss_pre_load = get_live_rss_mb()
        logger.info(f"[{req_tag}] RSS | Before {model_name} TFLite load: {rss_pre_load:.2f} MB")

        interp = None
        try:
            # 1. Instantiate lightweight TFLite interpreter
            interp = create_tflite_interpreter(tflite_path)
            rss_post_load = get_live_rss_mb()
            logger.info(f"[{req_tag}] RSS | {model_name} TFLite loaded: {rss_post_load:.2f} MB")

            # 2. Set tensor data
            in_details = interp.get_input_details()
            out_details = interp.get_output_details()
            
            # Ensure float32 tensor
            tensor_data = X_input.astype(np.float32) if X_input.dtype != np.float32 else X_input
            interp.set_tensor(in_details[0]["index"], tensor_data)

            # 3. Invoke inference
            interp.invoke()
            raw_prob = float(np.squeeze(interp.get_tensor(out_details[0]["index"])))

            rss_inferred = get_live_rss_mb()
            logger.info(f"[{req_tag}] RSS | {model_name} TFLite inference: {rss_inferred:.2f} MB (P(REAL)={raw_prob:.6f})")
            return raw_prob

        except Exception as e:
            logger.error(f"[{req_tag}] TFLite inference error in '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"TFLite model evaluation failed for '{model_name}': {str(e)}")

        finally:
            # Explicit cleanup after EACH model
            del interp
            gc.collect()
            rss_cleanup = get_live_rss_mb()
            logger.info(f"[{req_tag}] RSS | {model_name} TFLite cleanup: {rss_cleanup:.2f} MB")

    def predict_single_model_keras(
        self,
        model_cfg: Dict[str, Any],
        X_input: np.ndarray,
        request_id: Optional[str] = None
    ) -> float:
        """
        Keras (.h5) Reference / Rollback Path.
        """
        req_tag = f"Req {request_id}" if request_id else "Inference"
        tf, PositionalEncoding = get_tf_and_custom_objects(request_id)

        model_name = model_cfg["name"]
        model_path = os.path.join(MODELS_DIR, model_cfg["filename"])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        rss_before_load = get_live_rss_mb()
        logger.info(f"[{req_tag}] RSS | Before {model_name} Keras load: {rss_before_load:.2f} MB")

        model = None
        pred_tensor = None
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={'PositionalEncoding': PositionalEncoding}
            )

            rss_after_load = get_live_rss_mb()
            logger.info(f"[{req_tag}] RSS | {model_name} Keras loaded: {rss_after_load:.2f} MB")

            pred_tensor = model(X_input, training=False)
            raw_prob = float(np.squeeze(pred_tensor.numpy()))
            
            rss_after_infer = get_live_rss_mb()
            logger.info(f"[{req_tag}] RSS | {model_name} Keras inference: {rss_after_infer:.2f} MB (P(REAL)={raw_prob:.6f})")
            return raw_prob

        except Exception as e:
            logger.error(f"[{req_tag}] Keras inference error in '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Keras model evaluation failed for '{model_name}': {str(e)}")

        finally:
            del model
            del pred_tensor
            tf.keras.backend.clear_session()
            gc.collect()
            rss_after_cleanup = get_live_rss_mb()
            logger.info(f"[{req_tag}] RSS | {model_name} Keras cleanup: {rss_after_cleanup:.2f} MB")

    def predict_single_model(
        self,
        model_cfg: Dict[str, Any],
        X_input: np.ndarray,
        request_id: Optional[str] = None,
        force_keras: bool = False
    ) -> float:
        """
        Executes prediction for a single model.
        Uses FP32 TFLite by default with graceful fallback to Keras .h5 if requested or needed.
        """
        if force_keras:
            return self.predict_single_model_keras(model_cfg, X_input, request_id=request_id)

        tflite_file = model_cfg.get("tflite_filename", model_cfg["filename"].replace(".h5", ".tflite"))
        tflite_path = os.path.join(MODELS_DIR, tflite_file)

        if os.path.exists(tflite_path):
            try:
                return self.predict_single_model_tflite(model_cfg, X_input, request_id=request_id)
            except Exception as e:
                logger.warning(f"TFLite failed for {model_cfg['name']}, falling back to Keras reference: {e}")
                return self.predict_single_model_keras(model_cfg, X_input, request_id=request_id)

        return self.predict_single_model_keras(model_cfg, X_input, request_id=request_id)

    def predict_ensemble(
        self,
        X_input: np.ndarray,
        request_id: Optional[str] = None,
        force_keras: bool = False
    ) -> Dict[str, Any]:
        """
        Sequentially executes all 3 models with memory cleanup between iterations,
        then calculates the majority voting consensus.
        """
        if request_id is None:
            request_id = uuid.uuid4().hex[:8]

        req_tag = f"Req {request_id}"
        if not self.is_ready:
            self.initialize()

        probabilities: List[float] = []
        model_results: Dict[str, Any] = {}
        votes: List[str] = []

        # Process-local lock ensures only one request evaluates models at a time
        with self._inference_lock:
            for cfg in self.model_configs:
                model_id = cfg["id"]
                raw_prob = self.predict_single_model(cfg, X_input, request_id=request_id, force_keras=force_keras)

                # Binary classification: P(REAL) > 0.5 -> REAL, else FAKE
                prediction = "REAL" if raw_prob > 0.5 else "FAKE"
                confidence_pct = raw_prob * 100.0 if prediction == "REAL" else (1.0 - raw_prob) * 100.0

                probabilities.append(raw_prob)
                votes.append(prediction)

                raw_real_pct = round(raw_prob * 100.0, 2)
                raw_fake_pct = round((1.0 - raw_prob) * 100.0, 2)

                model_results[model_id] = {
                    "name": cfg["name"],
                    "architecture": cfg["architecture"],
                    "test_split_accuracy": cfg["test_split_accuracy"],
                    "raw_probability": round(raw_prob, 4),
                    "real_probability_pct": raw_real_pct,
                    "fake_probability_pct": raw_fake_pct,
                    "confidence_pct": round(confidence_pct, 2),
                    "prediction": prediction
                }

        # 4. Majority Voting Consensus (>= 2 of 3 models)
        real_votes = votes.count("REAL")
        fake_votes = votes.count("FAKE")

        avg_real_prob = float(np.mean(probabilities))
        avg_fake_prob = 1.0 - avg_real_prob

        if real_votes >= 2:
            final_decision = "REAL"
            agreement_count = real_votes
            # Soft ensemble confidence: average P(REAL)
            ensemble_confidence = avg_real_prob * 100.0
        else:
            final_decision = "FAKE"
            agreement_count = fake_votes
            # Soft ensemble confidence: average P(FAKE) = 1 - P(REAL)
            ensemble_confidence = avg_fake_prob * 100.0

        rss_end = get_live_rss_mb()
        logger.info(f"[{req_tag}] RSS | Complete prediction end: {rss_end:.2f} MB")

        return {
            "final_decision": final_decision,
            "is_fake": final_decision == "FAKE",
            "is_real": final_decision == "REAL",
            "majority_vote": {
                "decision": final_decision,
                "real_votes": real_votes,
                "fake_votes": fake_votes,
                "total_models": len(self.model_configs),
                "agreement": f"{agreement_count}/{len(self.model_configs)}",
                "ensemble_confidence_pct": round(ensemble_confidence, 2),
                "avg_real_probability": round(avg_real_prob, 4),
                "avg_fake_probability": round(avg_fake_prob, 4)
            },
            "models": model_results
        }


inference_service = InferenceService()
