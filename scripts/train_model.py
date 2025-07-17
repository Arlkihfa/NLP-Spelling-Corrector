# scripts/train_model.py

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. TENTUKAN HYPERPARAMETERS AWAL ---
EMBEDDING_DIM = 128
LSTM_UNITS = 256
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 10 # Pelatihan awal dengan jumlah epoch kecil

# --- 2. FUNGSI UNTUK MEMUAT DATA & PARAMETER ---
def load_data_and_params():
    """Memuat data pelatihan (X, Y) dan parameter dari file."""
    print("Memuat data pelatihan dan parameter...")
    try:
        X = np.load("../data/X.npy")
        Y = np.load("../data/Y.npy")
        with open("../data/char2idx.json", "r") as f:
            char2idx = json.load(f)

        vocab_size = len(char2idx)
        max_seq_len = X.shape[1]

        print("Data dan parameter berhasil dimuat.")
        return X, Y, vocab_size, max_seq_len
    except FileNotFoundError as e:
        print(f"ERROR: File data tidak ditemukan. Pastikan Anda sudah menjalankan skrip data sebelumnya. Detail: {e}")
        return None, None, None, None

# --- 3. FUNGSI UNTUK MEMBANGUN MODEL ---
def build_model(vocab_size, max_seq_len):
    """Membangun arsitektur Encoder-Decoder LSTM."""
    print("Membangun arsitektur model...")
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_seq_len, mask_zero=True))
    model.add(LSTM(LSTM_UNITS))
    model.add(RepeatVector(max_seq_len))
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    return model

# --- FUNGSI UNTUK PLOT HASIL PELATIHAN ---
def plot_history(history):
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

    plt.savefig('training_history.png')
    print("Grafik hasil pelatihan disimpan sebagai training_history.png")
    plt.show()

# --- SKRIP UTAMA ---
if __name__ == "__main__":
    X, Y, vocab_size, max_seq_len = load_data_and_params()

    if X is not None:
        # Reshape Y untuk loss function sparse_categorical_crossentropy
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

        # Pisahkan data menjadi training dan validation set (80% train, 20% val)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"Data dibagi: {len(X_train)} training, {len(X_val)} validation.")

        model = build_model(vocab_size, max_seq_len)

        # Kompilasi model: Mengonfigurasi proses belajar
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        print("\nMemulai pelatihan model...")
        # Lakukan pelatihan model
        history = model.fit(
            X_train, Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, Y_val)
        )

        print("\nPelatihan selesai.")

        # Simpan model yang sudah dilatih
        model.save("spelling_corrector_model.h5")
        print("Model berhasil disimpan sebagai spelling_corrector_model.h5")

        # Buat plot dari history pelatihan
        plot_history(history)