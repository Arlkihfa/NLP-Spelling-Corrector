# scripts/train_model.py

import numpy as np
import json
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
# PERUBAHAN: Impor Dropout dan EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. TENTUKAN HYPERPARAMETERS AWAL ---
EMBEDDING_DIM = 256
LSTM_UNITS = 512
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
EPOCHS = 200 # Kita bisa set epoch tinggi, karena EarlyStopping akan menghentikannya

# --- 2. FUNGSI UNTUK MEMUAT DATA & PARAMETER (DENGAN PATH DINAMIS) ---
def load_data_and_params():
    """Memuat data pelatihan (X, Y) dan parameter dari file."""
    print("Memuat data pelatihan dan parameter...")
    try:
        # Menentukan path secara dinamis relatif terhadap lokasi file ini
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

        # Membangun path ke file data dari root proyek
        X_PATH = os.path.join(PROJECT_ROOT, "data", "X.npy")
        Y_PATH = os.path.join(PROJECT_ROOT, "data", "Y.npy")
        CHAR2IDX_PATH = os.path.join(PROJECT_ROOT, "data", "char2idx.json")

        X = np.load(X_PATH)
        Y = np.load(Y_PATH)
        with open(CHAR2IDX_PATH, "r") as f:
            char2idx = json.load(f)
        
        vocab_size = len(char2idx)
        max_seq_len = X.shape[1]

        print("Data dan parameter berhasil dimuat.")
        return X, Y, vocab_size, max_seq_len, PROJECT_ROOT
    except FileNotFoundError as e:
        print(f"ERROR: File data tidak ditemukan. Pastikan file-file .npy dan .json sudah ada di folder 'data'. Detail: {e}")
        return None, None, None, None, None

# --- 3. FUNGSI UNTUK MEMBANGUN MODEL (DENGAN DROPOUT) ---
def build_model(vocab_size, max_seq_len):
    """Membangun arsitektur Encoder-Decoder LSTM dengan Dropout."""
    print("Membangun arsitektur model...")
    model = Sequential()
    
    # --- ENCODER ---
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_seq_len, mask_zero=True))
    # PERUBAHAN: Tambahkan Dropout setelah Embedding
    model.add(Dropout(0.3))
    model.add(LSTM(LSTM_UNITS))

    # --- JEMBATAN ---
    model.add(RepeatVector(max_seq_len))

    # --- DECODER ---
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    # PERUBAHAN: Tambahkan Dropout setelah LSTM Decoder
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    return model
    
# --- FUNGSI UNTUK PLOT HASIL PELATIHAN ---
def plot_history(history, project_root):
    """Membuat plot untuk loss dan accuracy."""
    plt.figure(figsize=(12, 5))
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Menyimpan plot di direktori utama proyek
    plot_path = os.path.join(project_root, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Grafik hasil pelatihan disimpan sebagai {plot_path}")
    plt.show()

# --- SKRIP UTAMA ---
if __name__ == "__main__":
    X, Y, vocab_size, max_seq_len, project_root = load_data_and_params()
    
    if X is not None:
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"Data dibagi: {len(X_train)} training, {len(X_val)} validation.")

        model = build_model(vocab_size, max_seq_len)
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.summary()
        
        # PERUBAHAN: Definisikan callback EarlyStopping
        # Ini akan menghentikan pelatihan jika val_loss tidak membaik selama 10 epoch
        # dan akan mengembalikan bobot terbaik yang pernah dicapai.
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("\nMemulai pelatihan model...")
        history = model.fit(
            X_train, Y_train,
            batch_size=128,
            epochs=EPOCHS,
            validation_data=(X_val, Y_val),
            # PERUBAHAN: Tambahkan callback ke proses pelatihan
            callbacks=[early_stopping]
        )
        
        print("\nPelatihan selesai.")
        
        # Pastikan folder 'models' ada
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Menyimpan model di dalam folder 'models'
        model_path = os.path.join(models_dir, "spelling_corrector_model.h5")
        model.save(model_path)
        print(f"Model berhasil disimpan sebagai {model_path}")

        plot_history(history, project_root)