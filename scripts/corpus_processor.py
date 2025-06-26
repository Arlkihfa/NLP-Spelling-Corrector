import re

def bersihkan_kalimat(kalimat):
    # Lowercase
    kalimat = kalimat.lower()
    # Hapus semua karakter kecuali huruf dan spasi
    kalimat = re.sub(r'[^a-z\s]', '', kalimat)
    # Normalisasi spasi
    kalimat = re.sub(r'\s+', ' ', kalimat).strip()
    return kalimat

# Baca file korpus mentah
with open("data/ind_mixed_2013_100K-sentences.txt", encoding="utf-8") as f:
    kalimat_mentah = f.readlines()

# Bersihkan semua kalimat
kalimat_bersih = [bersihkan_kalimat(kal) for kal in kalimat_mentah]

# Simpan hasil ke file baru
with open("data/clean_sentences.txt", "w", encoding="utf-8") as f:
    for kal in kalimat_bersih:
        f.write(kal + "\n")

print("✅ Semua kalimat berhasil dibersihkan dan disimpan di 'data/clean_sentences.txt'")
