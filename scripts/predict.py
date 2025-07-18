# scripts/predict.py

import numpy as np
import json
from tensorflow.keras.models import load_model
import os
import sys

# --- KONFIGURASI PATH DINAMIS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "spelling_corrector_model.h5")
CHAR2IDX_PATH = os.path.join(PROJECT_ROOT, "data", "char2idx.json")
IDX2CHAR_PATH = os.path.join(PROJECT_ROOT, "data", "idx2char.json")
MAX_SEQ_LEN = 20 # Harus sama dengan yang digunakan saat training

# --- FUNGSI UNTUK MEMUAT SEMUA SUMBER DAYA ---
def load_resources():
    """Memuat model dan kamus dari path yang sudah ditentukan."""
    print("Memuat model dan kamus...")
    try:
        model = load_model(MODEL_PATH)
        with open(CHAR2IDX_PATH, "r") as f:
            char2idx = json.load(f)
        with open(IDX2CHAR_PATH, "r") as f:
            idx2char_str_keys = json.load(f)
            idx2char = {int(k): v for k, v in idx2char_str_keys.items()}
        print("Model dan kamus berhasil dimuat.")
        return model, char2idx, idx2char
    except Exception as e:
        print(f"Error saat memuat model atau kamus: {e}")
        print("\nPASTIKAN file 'spelling_corrector_model.h5' ada di dalam folder 'models/' dan file-file di folder 'data' sudah ada.")
        return None, None, None

# --- FUNGSI UNTUK PREDIKSI ---
def predict_correction(word, model, char2idx, idx2char):
    """Menerima kata salah eja dan mengembalikan prediksinya."""
    word = word.lower()
    encoded = [char2idx.get(c, 0) for c in word]
    
    if len(encoded) < MAX_SEQ_LEN:
        encoded += [char2idx.get('<PAD>', 0)] * (MAX_SEQ_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_SEQ_LEN]
    
    input_seq = np.array([encoded])
    prediction = model.predict(input_seq, verbose=0)
    predicted_indices = np.argmax(prediction, axis=-1)[0]
    
    corrected_word = ""
    for idx in predicted_indices:
        char = idx2char.get(idx, "")
        if char == '<EOS>':
            break
        if char not in ['<SOS>', '<PAD>']:
            corrected_word += char
    
    return corrected_word

# --- UJI COBA LANGSUNG ---
if __name__ == "__main__":
    model, char2idx, idx2char = load_resources()

    if model is None:
        sys.exit(1)

    while True:
        test_word = input("Masukkan kata salah eja (atau ketik 'keluar' untuk berhenti): ")
        if test_word.lower() == 'keluar':
            break
        
        corrected = predict_correction(test_word, model, char2idx, idx2char)
        print(f"Input: {test_word}")
        print(f"Prediksi Koreksi: {corrected}\n")
