# scripts/model_architecture.py

# 1. Impor pustaka yang dibutuhkan
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import json

# 2. Fungsi untuk memuat parameter dari file data
def load_params_from_data():
    # Muat kamus karakter untuk mengetahui jumlah karakter unik
    with open("data/char2idx.json", "r") as f:
        char2idx = json.load(f)

    # Ambil nilai-nilai penting
    num_unique_chars = len(char2idx)
    max_seq_len = 20 # Diambil dari MAX_SEQ_LEN di data_vectorizer.py

    return num_unique_chars, max_seq_len

# 3. Definisikan Hyperparameter Model
DIMENSI_EMBEDDING = 128
DIMENSI_LSTM = 256

# 4. Bangun arsitektur dalam sebuah fungsi
def build_model(jumlah_karakter_unik, panjang_maksimum):
    """
    Fungsi untuk membangun arsitektur model Encoder-Decoder LSTM.
    """
    # --- ENCODER ---
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    encoder_embedding = Embedding(jumlah_karakter_unik, DIMENSI_EMBEDDING)(encoder_inputs)
    _, state_h, state_c = LSTM(DIMENSI_LSTM, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # --- DECODER ---
    decoder_inputs = Input(shape=(None,), name="decoder_input")
    decoder_embedding = Embedding(jumlah_karakter_unik, DIMENSI_EMBEDDING)(decoder_inputs)
    decoder_lstm_output, _, _ = LSTM(DIMENSI_LSTM, return_sequences=True, return_state=True)(decoder_embedding, initial_state=encoder_states)
    decoder_dense_output = Dense(jumlah_karakter_unik, activation='softmax')(decoder_lstm_output)

    # --- GABUNGKAN MODEL ---
    model = Model([encoder_inputs, decoder_inputs], decoder_dense_output)

    return model

# 5. Uji cetak biru dengan mencetak ringkasan
if __name__ == '__main__':
    # Jalankan dulu data_vectorizer.py untuk membuat file .json dan .npy
    print("Pastikan Anda sudah menjalankan 'data_vectorizer.py' terlebih dahulu.")

    # Muat parameter dari data yang sudah diproses
    JUMLAH_KARAKTER, PANJANG_MAKS = load_params_from_data()

    print(f"Jumlah karakter unik terdeteksi: {JUMLAH_KARAKTER}")
    print(f"Panjang sekuens maksimum: {PANJANG_MAKS}")

    # Bangun model dengan parameter yang benar
    model = build_model(JUMLAH_KARAKTER, PANJANG_MAKS)

    print("\n--- Cetak Biru Arsitektur Model ---")
    model.summary()