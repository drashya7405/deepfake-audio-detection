import os
import glob
import argparse
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, max_steps=1000, max_dims=512, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims
        
        dims = max_dims
        if dims % 2 == 1: dims += 1 
        
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
# ---------------------------------------------------------

# --- PATHS AND CONFIGURATIONS ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()

DRIVE_MODELS_DIR = '/content/drive/MyDrive/SoftComputing/BestModels'
MODEL_FILES = [
    'drashya_best_deepfake_audio_model.h5',
    'devesh_best_deepfake_audio_model.h5',
    'swayam_best_deepfake_audio_model.h5'
]
DEFAULT_MODEL_PATHS = [os.path.join(DRIVE_MODELS_DIR, f) for f in MODEL_FILES]
DEFAULT_TRAINING_DATA = os.path.join(PROJECT_ROOT, 'DATASET-balanced.csv')

FEATURE_COLUMNS = [
    'chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
    'rolloff', 'zero_crossing_rate',
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8',
    'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
    'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
]

# --- 1. FEATURE EXTRACTION ---
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Extract features
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    
    feature_values = [
        chroma_stft, rms, spectral_centroid, spectral_bandwidth,
        rolloff, zero_crossing_rate, *mfcc_means
    ]
    
    return dict(zip(FEATURE_COLUMNS, feature_values))

# --- 2. ENSEMBLE CLASSIFICATION ---
def classify_audio_features(features_df, model_paths, training_dataset_path):
    print(f"\nLoading training data to fit scalers from: {training_dataset_path}...")
    if not os.path.exists(training_dataset_path):
        raise FileNotFoundError(f"Training data not found: {training_dataset_path}")
        
    training_df = pd.read_csv(training_dataset_path)
    
    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(training_df[FEATURE_COLUMNS])

    # Fit Label Encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(training_df['LABEL'])

    # Preprocess new features
    X_new = features_df[FEATURE_COLUMNS].astype(float)
    X_new_scaled = scaler.transform(X_new)
    # Reshape for 1D CNN / RNN input
    X_new_reshaped = np.reshape(X_new_scaled, (X_new_scaled.shape[0], X_new_scaled.shape[1], 1))

    all_pred_probs = []

    # Load and predict with each model
    print("\nStarting ensemble predictions...")
    for m_path in model_paths:
        if not os.path.exists(m_path):
            raise FileNotFoundError(f'Model file not found: {m_path}')
        
        print(f" -> Predicting with model: {os.path.basename(m_path)}...")
        
        # --- CRITICAL FIX: Pass the custom object dictionary here ---
        model = tf.keras.models.load_model(
            m_path, 
            compile=False,
            custom_objects={'PositionalEncoding': PositionalEncoding}
        )
        # -------------------------------------------------------------
        
        probs = model.predict(X_new_reshaped, verbose=0)
        all_pred_probs.append(probs)

    # Calculate average prediction across all 3 models (Ensemble)
    avg_pred_probs = np.mean(all_pred_probs, axis=0)
    final_pred_labels = (avg_pred_probs > 0.5).astype(int).flatten()
    final_class_names = label_encoder.inverse_transform(final_pred_labels)

    # Format results
    results_df = pd.DataFrame({
        'drashya_prob': all_pred_probs[0].flatten(),
        'devesh_prob': all_pred_probs[1].flatten(),
        'swayam_prob': all_pred_probs[2].flatten(),
        'avg_ensemble_prob': avg_pred_probs.flatten(),
        'final_predicted_label': final_class_names
    })

    if 'audio_file' in features_df.columns:
        results_df.insert(0, 'audio_file', features_df['audio_file'])

    print('\n' + '='*60)
    print(' FINAL ENSEMBLE PREDICTION RESULTS')
    print('='*60)

    for _, row in results_df.iterrows():
        audio_name = row['audio_file'] if 'audio_file' in results_df.columns else 'input_audio'
        print(f"File: {audio_name}")
        print(f"Final Decision: {row['final_predicted_label']} (Avg REAL probability: {row['avg_ensemble_prob']:.4f})")
        print(f"  ├─ Drashya model probability: {row['drashya_prob']:.4f}")
        print(f"  ├─ Devesh model probability:  {row['devesh_prob']:.4f}")
        print(f"  └─ Swayam model probability:  {row['swayam_prob']:.4f}")
        print("-" * 60)

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from an MP3 and classify using an ensemble of 3 deepfake audio models.")
    parser.add_argument("input_path", help="Path to a specific .mp3 file or a directory containing .mp3 files.")
    parser.add_argument("--training_data", default=DEFAULT_TRAINING_DATA, help="Path to the original balanced training dataset CSV (needed for scaler).")
    
    args = parser.parse_args()
    input_path = args.input_path

    # Determine if input is a file or directory
    if os.path.isfile(input_path) and input_path.lower().endswith('.mp3'):
        mp3_files = [input_path]
    elif os.path.isdir(input_path):
        mp3_files = sorted(glob.glob(os.path.join(input_path, '*.mp3')))
    else:
        raise FileNotFoundError(f"Invalid input. Please provide a valid .mp3 file or directory: {input_path}")

    if not mp3_files:
        raise FileNotFoundError(f"No .mp3 files found for input: {os.path.abspath(input_path)}")

    print(f"Found {len(mp3_files)} audio file(s) to process.")
    
    # Process files
    rows = []
    for mp3_file in mp3_files:
        try:
            print(f"Extracting features from: {os.path.basename(mp3_file)}...")
            feature_row = extract_audio_features(mp3_file)
            feature_row['audio_file'] = os.path.basename(mp3_file)
            rows.append(feature_row)
        except Exception as e:
            print(f"Skipping {mp3_file} due to extraction error: {e}")

    # Build DataFrame and run classification
    features_df = pd.DataFrame(rows)
    
    if features_df.empty:
        raise ValueError("No feature rows were created. Please check the input audio files.")
    
    # Run the models directly on the dataframe
    classify_audio_features(features_df, DEFAULT_MODEL_PATHS, args.training_data)