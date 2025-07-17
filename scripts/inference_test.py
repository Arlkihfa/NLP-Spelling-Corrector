import numpy as np
import json
from tensorflow.keras.models import load_model

# ======== LOAD MODEL & KAMUS =========
model = load_model("spelling_corrector_model.h5")

with open("../data/char2idx.json") as f:
    char2idx = json.load(f)

with open("../data/idx2char.json") as f:
    idx2char = json.load(f)


MAX_SEQ_LEN = 20  # Sesuaikan dengan hasil training

# ======== DECODE HASIL PREDIKSI =========
def decode_prediction(pred_seq):
    result = []
    found_real_char = False
    for idx in pred_seq:
        idx = int(idx)
        if idx == 0 and not found_real_char:
            continue  # skip leading padding
        if idx == 0 and found_real_char:
            break     # stop if padding after real chars
        found_real_char = True
        result.append(idx2char.get(str(idx), ''))
    return ''.join(result)


# ======== KOREKSI TYPO =========
def koreksi_typo(kata_typo):
    encoded = [char2idx.get(c, 1) for c in kata_typo]  # 1 = unknown
    padded = encoded + [0] * (MAX_SEQ_LEN - len(encoded))
    input_arr = np.array([padded])

    pred = model.predict(input_arr)
    pred_indices = np.argmax(pred, axis=-1)[0]
    print("Predicted indices:", pred_indices)
    print("Raw prediction (argmax):", list(pred_indices))
    print("Decoded result:", decode_prediction(pred_indices))
    return decode_prediction(pred_indices)

# ======== TESTING =========
if __name__ == "__main__":
    while True:
        typo = input("Masukkan kata typo ('exit' untuk keluar): ").strip().lower()
        if typo == "exit":
            break
        hasil = koreksi_typo(typo)
        print(f"Hasil koreksi: {hasil}\n")
