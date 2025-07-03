# scripts/syntetic_data_generator.py

import random
import csv
import pandas as pd # Tambahkan impor pandas

# --- KONFIGURASI ---
PATH_SUMBER_KATA = "data/kbbi_v.csv" 
PATH_HASIL_CSV = "data/training_pairs.csv"
JUMLAH_PASANGAN_TARGET = 500000 
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# --- FUNGSI-FUNGSI PEMBUAT TYPO ---

def insertion(word):
    if not word: return word
    pos = random.randint(0, len(word))
    char = random.choice(ALPHABET)
    return word[:pos] + char + word[pos:]

def deletion(word):
    if not word: return word
    pos = random.randint(0, len(word) - 1)
    return word[:pos] + word[pos+1:]

def substitution(word):
    if not word: return word
    pos = random.randint(0, len(word) - 1)
    char = random.choice(ALPHABET)
    return word[:pos] + char + word[pos+1:]

def transposition(word):
    if len(word) < 2: return word
    pos = random.randint(0, len(word) - 2)
    return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

# --- SKRIP UTAMA ---
if __name__ == "__main__":
    print("Memulai proses pembuatan data pelatihan sintetis...")

    # 1. Muat daftar kata benar dari KBBI (CARA YANG DIPERBAIKI)
    print(f"Memuat kata dari {PATH_SUMBER_KATA}...")
    try:
        # Menggunakan pandas untuk membaca CSV, ini lebih andal
        # header=None digunakan jika file CSV tidak memiliki baris header
        df_kbbi = pd.read_csv(PATH_SUMBER_KATA, header=None)

        # Asumsikan kata ada di kolom pertama (indeks 0)
        # Membersihkan data: hapus baris kosong, ubah ke string, lalu ke huruf kecil
        list_kata_benar = df_kbbi[0].dropna().astype(str).str.lower().tolist()

        # Filter lagi untuk memastikan hanya kata yang berisi huruf yang masuk
        list_kata_benar = [word for word in list_kata_benar if word.isalpha()]

        if not list_kata_benar:
            print("ERROR: Tidak ada kata valid yang ditemukan di file CSV. Pastikan file tidak kosong dan formatnya benar.")
            exit()

        print(f"Ditemukan {len(list_kata_benar)} kata benar.")

    except FileNotFoundError:
        print(f"ERROR: File sumber kata '{PATH_SUMBER_KATA}' tidak ditemukan.")
        exit()
    except Exception as e:
        print(f"Terjadi error saat membaca file CSV: {e}")
        exit()

    # 2. Siapkan daftar fungsi typo
    fungsi_typo = [insertion, deletion, substitution, transposition]

    pasangan_pelatihan = []

    print(f"Menghasilkan {JUMLAH_PASANGAN_TARGET} pasangan pelatihan...")
    # 3. Looping untuk membuat pasangan data
    while len(pasangan_pelatihan) < JUMLAH_PASANGAN_TARGET:
        kata_benar = random.choice(list_kata_benar)

        if len(kata_benar) < 3:
            continue

        metode = random.choice(fungsi_typo)
        kata_salah = metode(kata_benar)

        if kata_salah != kata_benar:
            pasangan_pelatihan.append([kata_salah, kata_benar])

    print("Pembuatan data selesai.")

    # 4. Simpan hasilnya ke dalam file CSV
    print(f"Menyimpan data ke {PATH_HASIL_CSV}...")
    with open(PATH_HASIL_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['typo', 'benar'])
        writer.writerows(pasangan_pelatihan)

    print(f"Selesai! {len(pasangan_pelatihan)} pasangan berhasil disimpan.")