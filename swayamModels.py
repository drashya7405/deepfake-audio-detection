import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, SimpleRNN, Conv1D, MaxPooling1D, 
                                     Dropout, BatchNormalization, Flatten, 
                                     MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- CRITICAL FIX 1: Register the layer and add get_config ---
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    """
    Injects positional information into the sequence so the Transformer 
    MultiHeadAttention layer understands time/order.
    """
    def __init__(self, max_steps=1000, max_dims=512, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims
        
        dims = max_dims
        if dims % 2 == 1: dims += 1 # Ensure even dimension
        
        p, i = np.meshgrid(np.arange(max_steps), np.arange(dims // 2))
        pos_emb = np.empty((1, max_steps, dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / dims)).T
        self.positional_encoding = tf.constant(pos_emb, dtype=tf.float32)

    def call(self, inputs):
        shape = tf.shape(inputs)
        # Add the positional encoding to the input sequence
        return inputs + self.positional_encoding[:, :shape[1], :shape[2]]
        
    def get_config(self):
        # This allows Keras to save and load the custom layer arguments correctly
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "max_steps": self.max_steps,
            "max_dims": self.max_dims
        })
        return config

# Eager execution for debugging/custom layers
tf.config.run_functions_eagerly(True)

# --- REVISED OPTIMIZER ---
def create_optimizer():
    """
    Creates a new instance of the AdamW optimizer.
    This prevents the "Unknown variable" error from reusing the same optimizer.
    """
    try:
        from tensorflow.keras.optimizers import AdamW
        print("Creating new tf.keras.optimizers.AdamW instance.")
        return AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    except ImportError:
        print("Built-in AdamW not found. Creating standard Adam optimizer instance.")
        return tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

# --- 1. Data Loading and Preprocessing ---
print("Starting the deepfake audio detection training script...")
print("Current time in India:", pd.Timestamp.now(tz='Asia/Kolkata').strftime('%A, %B %d, %Y at %I:%M %p %Z'))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_MODELS_DIR = '/content/drive/MyDrive/SoftComputing/BestModels'
os.makedirs(BEST_MODELS_DIR, exist_ok=True)

try:
    df = pd.read_csv('/content/drive/MyDrive/SoftComputing/DATASET-balanced.csv')
    print("\nDataset 'DATASET-balanced.csv' loaded successfully.")
except FileNotFoundError:
    print("\nError: 'DATASET-balanced.csv' not found. Please ensure the dataset file is in the correct directory.")
    exit()

print("\n--- Data Preprocessing ---")
print("Data Head:")
print(df.head())

X = df.drop('LABEL', axis=1)
y = df['LABEL']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nLabels encoded. Classes found: {label_encoder.classes_}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features have been scaled using StandardScaler.")

X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

X_train, X_temp, y_train, y_temp = train_test_split(X_reshaped, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("\nDataset has been split into training, validation, and testing sets.")
print(f"Training data shape:   {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape:    {X_test.shape}")


# --- 2. Utility Functions for Plotting and Evaluation ---
def plot_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    print(f"\nSaved training history plot to '{model_name}_training_history.png'")
    plt.close()


def evaluate_model(model, X_test, y_test, model_name):
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n--- {model_name} Evaluation on Test Set ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, annot_kws={"size": 14})
    plt.title(f'{model_name} Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(f'{model_name}_confusion_matrix.png')
    print(f"Saved confusion matrix plot to '{model_name}_confusion_matrix.png'")
    plt.close()
    return {'model': model_name, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}


# --- 3. Model Implementation, Training, and Evaluation ---
results_list = []
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# --- Model 1: Simple Recurrent Neural Network (RNN) ---
print("\n\n--- Building Model 1: Simple RNN ---")
rnn_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(32),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer=create_optimizer(), loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()
print("\n--- Training RNN Model ---")
rnn_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[early_stopping, lr_scheduler], verbose=1)
plot_history(rnn_history, 'RNN')
results_list.append(evaluate_model(rnn_model, X_test, y_test, 'RNN'))

# --- Model 2: Hybrid CNN + RNN ---
print("\n\n--- Building Model 2: Hybrid CNN + RNN ---")
cnn_rnn_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    SimpleRNN(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn_rnn_model.compile(optimizer=create_optimizer(), loss='binary_crossentropy', metrics=['accuracy'])
cnn_rnn_model.summary()
print("\n--- Training CNN + RNN Model ---")
cnn_rnn_history = cnn_rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val),
                                    callbacks=[early_stopping, lr_scheduler], verbose=1)
plot_history(cnn_rnn_history, 'CNN_RNN')
results_list.append(evaluate_model(cnn_rnn_model, X_test, y_test, 'CNN-RNN'))

# --- Model 3: Hybrid CNN + Transformer ---
print("\n\n--- Building Model 3: Hybrid CNN + Transformer ---")
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = PositionalEncoding(max_steps=1000, max_dims=64)(x)

attn_output = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
x = LayerNormalization(epsilon=1e-6)(x + attn_output)

ffn = Dense(64, activation='relu')(x)
ffn = Dense(64)(ffn)
x = LayerNormalization(epsilon=1e-6)(x + ffn)

x = GlobalAveragePooling1D()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

cnn_transformer_model = Model(inputs=inputs, outputs=outputs)
cnn_transformer_model.compile(optimizer=create_optimizer(), loss='binary_crossentropy', metrics=['accuracy'])
cnn_transformer_model.summary()

print("\n--- Training CNN + Transformer Model ---")
cnn_transformer_history = cnn_transformer_model.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_scheduler], 
    verbose=1
)

plot_history(cnn_transformer_history, 'CNN_Transformer')
results_list.append(evaluate_model(cnn_transformer_model, X_test, y_test, 'CNN-Transformer'))

# --- 4. Final Performance Comparison ---
models_dict = {
    'RNN': rnn_model,
    'CNN-RNN': cnn_rnn_model,
    'CNN-Transformer': cnn_transformer_model
}

results_df = pd.DataFrame(results_list)
print("\n\n--- Final Model Performance Comparison ---")
print(results_df.to_string())
results_df.to_csv('model_performance_comparison.csv', index=False)
print("\nSaved final model performance comparison to 'model_performance_comparison.csv'")


# --- ADDED CODE TO SAVE THE BEST MODEL ---
try:
    best_model_index = results_df['accuracy'].idxmax()
    best_model_stats = results_df.loc[best_model_index]
    best_model_name = best_model_stats['model']
    best_model_accuracy = best_model_stats['accuracy']

    best_model_object = models_dict[best_model_name]

    # CRITICAL FIX 2: Fixed the save path so it doesn't try to join two absolute directories
    save_path = os.path.join(BEST_MODELS_DIR, 'swayam_best_deepfake_audio_model.h5')

    # Save the model
    best_model_object.save(save_path)

    print(f"\n--- Best Model Saved ---")
    print(f"The best performing model was '{best_model_name}' with {best_model_accuracy:.4f} accuracy.")
    print(f"Model has been successfully saved to '{save_path}'")

except Exception as e:
    print(f"\nAn error occurred while saving the best model: {e}")

print("\nScript finished successfully.")