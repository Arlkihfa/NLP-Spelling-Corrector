# scripts/train_model.py

import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

# --- 1. TENTUKAN HYPERPARAMETERS AWAL ---
EMBEDDING_DIM = 128
LSTM_UNITS = 256
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 10 # Kita akan mulai dengan sedikit epoch dulu

# --- 2. FUNGSI UNTUK MEMUAT DATA & PARAMETER ---
def load_data_and_params():
    """Memuat data pelatihan (X, Y) dan parameter dari file."""
    print("Memuat data pelatihan dan parameter...")
    try:
        X = np.load("data/X.npy")
        Y = np.load("data/Y.npy")
        with open("data/char2idx.json", "r") as f:
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
    """
    Membangun arsitektur Encoder-Decoder LSTM menggunakan Keras Sequential API.
    """
    print("Membangun arsitektur model...")
    model = Sequential()

    # --- ENCODER ---
    # Input shape tidak perlu ditentukan di lapisan pertama Sequential
    # Lapisan Embedding
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_seq_len, mask_zero=True))
    # Lapisan LSTM Encoder
    model.add(LSTM(LSTM_UNITS))

    # --- JEMBATAN ---
    # Lapisan RepeatVector untuk menduplikasi output Encoder
    model.add(RepeatVector(max_seq_len))

    # --- DECODER ---
    # Lapisan LSTM Decoder
    # return_sequences=True penting agar outputnya adalah urutan, bukan hanya output terakhir
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    # Lapisan TimeDistributed(Dense) untuk prediksi di setiap langkah waktu
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    return model

# --- SKRIP UTAMA ---
if __name__ == "__main__":
    # Muat data
    X_train, Y_train, vocab_size, max_seq_len = load_data_and_params()

    # Jika data berhasil dimuat, lanjutkan
    if X_train is not None:
        # Bangun model
        model = build_model(vocab_size, max_seq_len)

        # Compile model
        optimizer = Adam(learning_rate=LEARNING_RATE)
        # Menggunakan 'sparse_categorical_crossentropy' karena target (Y) kita adalah integer (bukan one-hot)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Cetak ringkasan arsitektur untuk verifikasi
        print("\n--- Ringkasan Arsitektur Model ---")
        model.summary()

        # Di sini kita akan menambahkan kode untuk melatih model di hari berikutnya
        # print("\nModel siap untuk dilatih.")