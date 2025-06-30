import sys
import os

# Tambahkan folder utama ke path agar bisa import antar-folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.aturan_eyd import koreksi_eyd

# Baca kalimat dari file bersih
with open("data/clean_sentences.txt", encoding="utf-8") as f:
    kalimat_bersih = [line.strip() for line in f]

# Terapkan koreksi EYD
kalimat_koreksi = [koreksi_eyd(kal) for kal in kalimat_bersih]

# Simpan hasil
with open("data/clean_sentences_eyd.txt", "w", encoding="utf-8") as f:
    for kal in kalimat_koreksi:
        f.write(kal + "\n")

print("✅ Semua kalimat sudah diproses dan disimpan di data/clean_sentences_eyd.txt")
