import pandas as pd
import numpy as np
import json

# Parameters
MAX_SEQ_LEN = 20  # kamu bisa sesuaikan nanti setelah lihat panjang maksimum di dataset

# 1. Load training pairs
df = pd.read_csv("../data/training_pairs_clean.csv")
typos = df["typo"].astype(str).tolist()
benars = df["benar"].astype(str).tolist()

# 2. Buat daftar semua karakter unik
all_text = typos + benars
unique_chars = sorted(list(set("".join(all_text))))

# Tambahkan token khusus
unique_chars = ['<PAD>', '<SOS>', '<EOS>'] + unique_chars
char2idx = {ch: idx for idx, ch in enumerate(unique_chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

# 3. Simpan kamus karakter
with open("../data/char2idx.json", "w") as f:
    json.dump(char2idx, f)
with open("../data/idx2char.json", "w") as f:
    json.dump(idx2char, f)

# 4. Fungsi encode kata → indeks
def encode(word):
    word = '<SOS>' + word + '<EOS>'
    encoded = [char2idx.get(c, 0) for c in word]
    if len(encoded) < MAX_SEQ_LEN:
        encoded += [char2idx['<PAD>']] * (MAX_SEQ_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_SEQ_LEN]
    return encoded

# 5. Buat array input-output
X = np.array([encode(w) for w in typos])
Y = np.array([encode(w) for w in benars])

# 6. Simpan sebagai .npy
np.save("../data/X.npy", X)
np.save("../data/Y.npy", Y)

print("Selesai! Dataset vektorisasi disimpan.")
print(f"Jumlah pasangan: {len(X)}, Panjang urutan: {MAX_SEQ_LEN}")
