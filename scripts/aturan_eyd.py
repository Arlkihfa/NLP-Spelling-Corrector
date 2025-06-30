import re

def koreksi_eyd(kalimat):
    # Gabungkan awalan 'di ' + kata kerja
    kalimat = re.sub(r'\bdi\s+(?=\w)', lambda m: 'di' if m.end() >= len(kalimat) else 'di' + kalimat[m.end()], kalimat)

    # Gabungkan beberapa kata umum lainnya
    kalimat = kalimat.replace("ke luar", "keluar")
    kalimat = kalimat.replace("ke dalam", "kedalam")
    kalimat = kalimat.replace("ke atas", "keatas")
    kalimat = kalimat.replace("ke bawah", "kebawah")
    kalimat = kalimat.replace("ke pinggir", "kepinggir")
    kalimat = kalimat.replace("ke samping", "kesamping")

    return kalimat

# Contoh penggunaan
if __name__ == "__main__":
    uji = [
        "Saya di makan oleh harimau.",
        "Anak itu ke luar rumah tanpa izin.",
        "Ia ke atas panggung dengan percaya diri.",
    ]
    
    for kalimat in uji:
        print(f"❌ Sebelum: {kalimat}")
        print(f"✅ Sesudah: {koreksi_eyd(kalimat)}\n")

from aturan_eyd import koreksi_eyd

# Baca file bersih hasil korpus
with open("data/clean_sentences.txt", encoding="utf-8") as f:
    kalimat_bersih = [line.strip() for line in f]

# Terapkan koreksi EYD
kalimat_koreksi = [koreksi_eyd(kal) for kal in kalimat_bersih]

# Simpan hasil
with open("data/clean_sentences_eyd.txt", "w", encoding="utf-8") as f:
    for kal in kalimat_koreksi:
        f.write(kal + "\n")

print("✅ Selesai! File hasil koreksi heuristik disimpan di 'data/clean_sentences_eyd.txt'")
