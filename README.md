# Deepfake Audio Detection Using Soft Computing and Deep Learning

## Final Project Documentation

This repository is a complete deepfake-audio detection project built for final submission. The system classifies an input audio sample as `REAL` or `FAKE` using handcrafted speech/audio features and a set of deep learning models trained on a balanced dataset.

The project does not depend on only one architecture. Instead, it compares **9 different models** across three experiment tracks, saves the best model from each track, and also provides an inference script that combines those best models as an **ensemble**.

## 1. Problem Statement

AI-based voice cloning and speech synthesis systems can generate highly realistic speech. Because of that, it is becoming increasingly difficult to tell whether an audio clip is genuine or artificially generated. This creates real risks in:

- voice impersonation
- misinformation
- fraud and spoofing
- fake evidence generation
- social engineering attacks

The goal of this project is to build a practical deepfake-audio detector that can learn the difference between real and manipulated speech using acoustic patterns extracted from audio recordings.

## 2. Project Objective

The main objectives of this project are:

- to detect whether an audio sample is `REAL` or `FAKE`
- to compare multiple deep learning architectures on the same balanced dataset
- to study whether handcrafted acoustic features are sufficient for deepfake-audio classification
- to identify the best-performing architecture among different model families
- to save deployable trained models for future inference
- to support prediction on unseen `.mp3` files using the same feature pipeline

## 3. What Makes This Project Complete

This repository includes the full workflow:

1. A balanced feature dataset in CSV format
2. Three model-training pipelines
3. Nine total deep learning architectures
4. Saved best models in `BestModels/`
5. Result plots in `Results/`
6. A prediction script for new audio files
7. A notebook version containing the recorded experiment outputs

So this is not just a training script. It is an end-to-end deepfake-audio classification pipeline.

## 4. Repository Structure

```text
SoftComputing/
├── README.md
├── SoftComputing.ipynb
├── DATASET-balanced.csv
├── deveshModels.py
├── drashyaModels.py
├── swayamModels.py
├── testModel.py
├── gehra_hua.mp3
├── FAKE_AUDIOS/
│   ├── file1004.mp3
│   ├── file1004.wav
│   ├── file1019.wav
├── BestModels/
│   ├── devesh_best_deepfake_audio_model.h5
│   ├── drashya_best_deepfake_audio_model.h5
│   ├── swayam_best_deepfake_audio_model.h5
└── Results/
    ├── BiLSTM_confusion_matrix.png
    ├── BiLSTM_training_history.png
    ├── CNN-RNN_confusion_matrix.png
    ├── CNN-Transformer_confusion_matrix.png
    ├── CNN_RNN_training_history.png
    ├── CNN_Transformer_training_history.png
    ├── GRU_confusion_matrix.png
    ├── GRU_training_history.png
    ├── RNN_confusion_matrix.png
    ├── RNN_training_history.png
    └── model_performance_comparison.csv
```

## 5. Dataset Details

The project uses `DATASET-balanced.csv` as the main training dataset.

### Dataset facts

- Total rows: `11,778`
- Total columns: `27`
- Feature columns: `26`
- Label column: `1`
- Task type: binary classification
- Classes:
  - `FAKE`: `5,889`
  - `REAL`: `5,889`

### Why the dataset is balanced

The dataset was balanced using **SMOTE**.

SMOTE stands for **Synthetic Minority Over-sampling Technique**. It creates synthetic samples for the minority class instead of simply duplicating records. This matters because a strongly imbalanced dataset can produce misleadingly high accuracy if the model learns to favor the majority class.

Using a balanced dataset makes the comparison between architectures much more meaningful.

## 6. Input Features Used

Each audio sample is represented using 26 extracted acoustic features:

```text
chroma_stft
rms
spectral_centroid
spectral_bandwidth
rolloff
zero_crossing_rate
mfcc1
mfcc2
mfcc3
mfcc4
mfcc5
mfcc6
mfcc7
mfcc8
mfcc9
mfcc10
mfcc11
mfcc12
mfcc13
mfcc14
mfcc15
mfcc16
mfcc17
mfcc18
mfcc19
mfcc20
```

### Meaning of the features

- `chroma_stft`: captures energy distribution across pitch classes
- `rms`: represents signal energy or loudness
- `spectral_centroid`: indicates the brightness of the sound
- `spectral_bandwidth`: measures spread of frequencies
- `rolloff`: identifies where most of the spectral energy is concentrated
- `zero_crossing_rate`: reflects signal noisiness and waveform activity
- `mfcc1` to `mfcc20`: summarize timbre and speech characteristics in a compact form

### Why these features were chosen

These features are widely used in speech and audio classification because they capture complementary information:

- spectral shape
- energy behavior
- timbre
- frequency concentration
- cepstral voice characteristics

Deepfake speech often sounds natural to humans, but synthetic generation can still leave measurable artifacts in spectral and cepstral patterns. MFCCs and spectral features are therefore a strong and practical choice for a final-year project that needs both accuracy and manageable computation.

## 7. Preprocessing Pipeline

All model scripts follow the same core preprocessing flow:

1. Load `DATASET-balanced.csv`
2. Separate input features and labels
3. Encode `REAL` and `FAKE` using `LabelEncoder`
4. Normalize numerical features using `StandardScaler`
5. Reshape data from `(samples, 26)` to `(samples, 26, 1)`
6. Split data into:
   - training: `70%`
   - validation: `15%`
   - testing: `15%`

### Why reshape to `(26, 1)`

Although the project uses handcrafted features instead of raw waveforms, the scripts treat the ordered feature vector as a sequence so that:

- `Conv1D` layers can learn local feature interactions
- `RNN`, `LSTM`, and `GRU` layers can model sequential dependencies
- hybrid architectures can combine both behaviors

## 8. Why 9 Models Were Chosen

The project compares **9 models** because one model family alone is not enough to show a meaningful academic comparison. The chosen set gives coverage across the major deep-learning ideas relevant to structured audio-feature classification.

### The 9-model design was a good choice for four reasons

#### 1. Family-level comparison

The project compares:

- convolution-based models
- recurrent models
- gated recurrent models
- bidirectional sequence models
- hybrid CNN-sequence models
- attention-based models

This makes the project stronger than a single-model solution.

#### 2. Fair benchmarking

All models use the same dataset, same feature space, same preprocessing, and nearly the same split strategy. That allows a fair comparison of architecture quality instead of comparing unrelated setups.

#### 3. Progressive complexity

The model list moves from simpler to more advanced architectures:

- CNN
- Simple RNN
- GRU
- BiLSTM
- CNN + RNN
- CNN + BiLSTM
- CNN + GRU-style recurrent hybrid
- CNN + Transformer

This helps answer an important project question:

**Does increasing architectural sophistication actually improve deepfake-audio detection on handcrafted features?**

#### 4. Strong final submission value

For a final project, comparing 9 models shows:

- experimentation depth
- engineering effort
- analytical thinking
- performance tradeoff analysis

So the choice of nine models is not arbitrary. It gives a broad and academically defensible architecture study.

## 9. Model Inventory

The repository contains three separate experiment tracks, each contributed through a dedicated script.

### A. `drashyaModels.py`

This script evaluates three models:

1. **BiLSTM**
2. **GRU**
3. **CNN-RNN**

#### Why these three were included

- **BiLSTM** was chosen to capture context in both directions across the feature sequence
- **GRU** was chosen as a lighter gated recurrent alternative to LSTM
- **CNN-RNN** was chosen to combine local pattern extraction with sequential modeling

### B. `deveshModels.py`

This script evaluates three models:

1. **CNN**
2. **Simple RNN**
3. **CNN-BiLSTM**

#### Why these three were included

- **CNN** was chosen as a strong baseline for local feature-pattern learning
- **Simple RNN** was chosen to test whether a basic recurrent model is already sufficient
- **CNN-BiLSTM** was chosen to combine convolutional extraction with bidirectional memory

### C. `swayamModels.py`

This script evaluates three models:

1. **Simple RNN**
2. **CNN-RNN**
3. **CNN-Transformer**

#### Why these three were included

- **Simple RNN** provides a recurrent baseline
- **CNN-RNN** tests a hybrid architecture with stronger local feature extraction
- **CNN-Transformer** explores whether attention can outperform recurrence on this feature representation

## 10. Architecture Summary

| Model | Script | Main idea | Why it matters |
|---|---|---|---|
| CNN | `deveshModels.py` | Learns local patterns using `Conv1D` | Good baseline for compact feature vectors |
| Simple RNN | `deveshModels.py` | Sequential modeling with recurrent memory | Tests whether ordered features behave like a sequence |
| CNN-BiLSTM | `deveshModels.py` | CNN front-end + bidirectional memory | Combines local extraction with context from both directions |
| BiLSTM | `drashyaModels.py` | Bidirectional long-term memory | Strong sequence learner for structured feature dependencies |
| GRU | `drashyaModels.py` | Lighter gated recurrent architecture | Faster and simpler than LSTM while still modeling long dependencies |
| CNN-RNN | `drashyaModels.py` | CNN + bidirectional GRU hybrid | Strong hybrid for extracting and integrating sequential patterns |
| Simple RNN | `swayamModels.py` | Recurrent baseline | Useful for cross-track comparison |
| CNN-RNN | `swayamModels.py` | CNN + RNN hybrid | Tests whether feature extraction before recurrence helps |
| CNN-Transformer | `swayamModels.py` | CNN front-end + self-attention | Explores attention-based modeling of the feature sequence |

## 11. Training Setup

### Common training choices

- Loss function: `binary_crossentropy`
- Output activation: `sigmoid`
- Label type: binary encoded labels
- Optimizer: `AdamW` when available, otherwise `Adam`
- Regularization:
  - `Dropout`
  - `BatchNormalization` in CNN-based models
- Callbacks:
  - `EarlyStopping`
  - `ReduceLROnPlateau`

### Epoch settings observed in the scripts

- `drashyaModels.py`: 5 epochs per model
- `deveshModels.py`: 5 epochs per model
- `swayamModels.py`:
  - RNN: 5 epochs
  - CNN-RNN: 5 epochs
  - CNN-Transformer: up to 30 epochs with callback-based stopping

## 12. Results

This project stores result artifacts inside the `Results/` folder, and the notebook `SoftComputing.ipynb` also contains recorded metric tables for the experiment runs.

### Important documentation note

The repository currently contains:

- a direct CSV result file for the `drashya` experiment track inside `Results/model_performance_comparison.csv`
- plot files for selected models inside `Results/`
- notebook output tables for the other experiment tracks

Because of that, the tables below are presented in two categories:

1. **Directly exported results from `Results/`**
2. **Recorded experiment results visible inside `SoftComputing.ipynb`**

This preserves the actual repository state without inventing missing files.

### 12.1 Direct result file in `Results/`

Source: `Results/model_performance_comparison.csv`

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| BiLSTM | 0.917374 | 0.938169 | 0.893545 | 0.915313 |
| GRU | 0.807583 | 0.821302 | 0.785957 | 0.803241 |
| CNN-RNN | 0.985852 | 0.978795 | 0.993205 | 0.985947 |

#### Best model from this result file

- **CNN-RNN**
- Accuracy: **98.59%**
- F1-score: **0.985947**

### 12.2 Recorded results for `deveshModels.py` from the notebook

Source: final comparison table printed in `SoftComputing.ipynb`

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN | 0.988115 | 0.979955 | 0.996602 | 0.988209 |
| RNN | 0.966044 | 0.957731 | 0.975085 | 0.966330 |
| CNN-BiLSTM | 0.985286 | 0.982002 | 0.988675 | 0.985327 |

#### Best model in this track

- **CNN**
- Accuracy: **98.81%**
- F1-score: **0.988209**

### 12.3 Recorded results for `swayamModels.py` from the notebook

The notebook contains **two recorded comparison outputs** for this track. Since both appear in the notebook history, both are documented here for transparency.

#### Notebook-recorded run A

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| RNN | 0.982456 | 0.978652 | 0.986410 | 0.982516 |
| CNN-RNN | 0.975099 | 0.976163 | 0.973952 | 0.975057 |
| CNN-Transformer | 0.983022 | 0.973363 | 0.993205 | 0.983184 |

Best model in run A:

- **CNN-Transformer**
- Accuracy: **98.30%**

#### Notebook-recorded run B

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| RNN | 0.983022 | 0.978676 | 0.987542 | 0.983089 |
| CNN-RNN | 0.984720 | 0.986364 | 0.983012 | 0.984685 |
| CNN-Transformer | 0.737974 | 0.990654 | 0.480181 | 0.646834 |

Best model in run B:

- **CNN-RNN**
- Accuracy: **98.47%**

### 12.4 Result interpretation

Across the repository, a few clear patterns appear:

- Hybrid models are consistently among the strongest performers
- Pure recurrent models are not always the best choice on handcrafted feature vectors
- CNN-based models perform extremely well because local feature interactions appear highly informative
- Transformer-based performance is unstable across the two recorded `swayam` runs, which suggests that attention did not always train consistently in this setup

### 12.5 Highest recorded performances in the project

Based on the metric tables currently available in the repository:

- Best exported result in `Results/`: **CNN-RNN** with **98.59% accuracy**
- Best recorded `devesh` result: **CNN** with **98.81% accuracy**
- Best recorded `swayam` result: **CNN-Transformer** with **98.30% accuracy** in run A
- Best recorded `swayam` result in the later notebook output: **CNN-RNN** with **98.47% accuracy** in run B

If a single final model must be highlighted from the currently documented evidence, the strongest recorded score in the repository is:

- **CNN from `deveshModels.py`**
- Accuracy: **98.81%**
- Recall: **99.66%**
- F1-score: **0.988209**

## 13. Result Artifacts in `Results/`

The `Results/` folder currently stores these artifacts:

- `BiLSTM_confusion_matrix.png`
- `BiLSTM_training_history.png`
- `GRU_confusion_matrix.png`
- `GRU_training_history.png`
- `CNN-RNN_confusion_matrix.png`
- `CNN_RNN_training_history.png`
- `RNN_confusion_matrix.png`
- `RNN_training_history.png`
- `CNN-Transformer_confusion_matrix.png`
- `CNN_Transformer_training_history.png`
- `model_performance_comparison.csv`

These plots are useful because they show:

- training vs validation behavior
- convergence pattern
- underfitting or overfitting trends
- class-wise classification performance through confusion matrices

## 14. Best Saved Models

The best model from each experiment track is saved in `BestModels/`:

- `BestModels/drashya_best_deepfake_audio_model.h5`
- `BestModels/devesh_best_deepfake_audio_model.h5`
- `BestModels/swayam_best_deepfake_audio_model.h5`

These saved models are later used by the inference pipeline.

## 15. Inference and Ensemble Prediction

The file `testModel.py` is used to classify new `.mp3` files.

### What it does

1. Loads a new audio file
2. Extracts the same 26 features using `librosa`
3. Fits scaler and label encoder using `DATASET-balanced.csv`
4. Loads all three best saved models
5. Gets prediction probabilities from each model
6. Averages the three probabilities
7. Produces the final ensemble prediction

### Output columns produced by the script

- `drashya_prob`
- `devesh_prob`
- `swayam_prob`
- `avg_ensemble_prob`
- `final_predicted_label`

### Why ensemble prediction is useful

Using an ensemble improves robustness because:

- it reduces dependence on one model
- it combines strengths of different architectures
- it can smooth out unstable predictions from a single network

## 16. How to Run the Project

### A. Train models

Run any of the training scripts:

```bash
python drashyaModels.py
python deveshModels.py
python swayamModels.py
```

### B. Test on a new `.mp3` file

```bash
python testModel.py path/to/audio.mp3
```

### C. Test all `.mp3` files inside a directory

```bash
python testModel.py path/to/folder
```

## 17. Main Libraries Used

- `Python`
- `NumPy`
- `Pandas`
- `Matplotlib`
- `Seaborn`
- `scikit-learn`
- `TensorFlow / Keras`
- `librosa`

## 18. Strengths of the Project

- compares 9 deep learning models instead of relying on only one
- uses a balanced dataset
- includes feature engineering, training, evaluation, and inference
- saves best-performing models
- provides visual result artifacts
- includes an ensemble-based final prediction system
- explores CNN, RNN, LSTM, GRU, hybrid, and Transformer ideas in one project

## 19. Limitations

Like every project, this one also has some limitations:

- the project uses handcrafted features rather than raw waveform end-to-end learning
- some file paths inside scripts are written for Google Colab / Google Drive usage
- the `Results/` folder does not currently contain a complete exported artifact set for every single model track
- no external benchmark dataset split is included beyond the provided processed CSV

## 20. Possible Future Improvements

- train on raw spectrograms or mel-spectrogram images
- export all result files for all 9 models in a fully organized way
- add ROC-AUC and PR-AUC metrics
- use k-fold cross-validation
- evaluate on more unseen real-world deepfake samples
- build a web interface for uploading and classifying audio
- convert saved models from legacy `.h5` format to modern `.keras`

## 21. Final Conclusion

This project successfully demonstrates that deepfake-audio detection can be performed effectively using handcrafted acoustic features and deep learning.

The comparison across 9 models shows that:

- architecture choice matters significantly
- CNN-based and hybrid CNN-sequence models are especially strong
- recurrent-only models can work well, but not always best
- attention-based models can be powerful, but may be less stable in this project configuration

Overall, the project is strong because it combines:

- good feature design
- balanced data
- multiple architecture comparisons
- model saving
- result visualization
- practical ensemble inference

For a final project, this provides both implementation depth and analytical comparison, which makes the work more complete, more credible, and more valuable than presenting a single model alone.

## 22. Final Summary in One Paragraph

This repository is a final deepfake-audio detection project that uses 26 handcrafted audio features, a balanced dataset of 11,778 samples, and 9 deep learning models across CNN, RNN, BiLSTM, GRU, CNN-BiLSTM, CNN-RNN, and CNN-Transformer architectures. The project compares model performance, stores plots and result files in `Results/`, saves the best model from each experiment track in `BestModels/`, and performs ensemble prediction on new `.mp3` files through `testModel.py`. The strongest recorded performance in the repository is the **CNN model from `deveshModels.py` with 98.81% accuracy**, while hybrid CNN-RNN models also perform extremely strongly and consistently.
