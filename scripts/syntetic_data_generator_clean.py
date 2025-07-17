# syntetic_data_generator_clean.py

import random
import csv
import pandas as pd

# --- KONFIGURASI ---
PATH_SUMBER_KATA = "../data/kbbi_v.csv"
PATH_HASIL_CSV = "../data/training_pairs_clean.csv"
JUMLAH_PASANGAN_TARGET = 1000  # ganti sesuai kebutuhan (uji dulu kecil, misal 1000)
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# --- FUNGSI TYPO ---
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

fungsi_typo = [insertion, deletion, substitution, transposition]

# --- MUAT KBBI ---
df_kbbi = pd.read_csv(PATH_SUMBER_KATA, header=None)
list_kata_benar = df_kbbi[0].dropna().astype(str).str.lower().tolist()
list_kata_benar = [word for word in list_kata_benar if word.isalpha()]

pasangan_pelatihan = []

while len(pasangan_pelatihan) < JUMLAH_PASANGAN_TARGET:
    kata_benar = random.choice(list_kata_benar)
    if len(kata_benar) < 3:
        continue
    metode = random.choice(fungsi_typo)
    kata_salah = metode(kata_benar)

    # ✅ Filter typo yang tidak realistis
    if kata_salah != kata_benar and abs(len(kata_salah) - len(kata_benar)) <= 3 and all(c in ALPHABET for c in kata_salah):
        pasangan_pelatihan.append([kata_salah, kata_benar])

# --- SIMPAN CSV ---
with open(PATH_HASIL_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['typo', 'benar'])
    writer.writerows(pasangan_pelatihan)

print(f"Selesai! {len(pasangan_pelatihan)} pasangan disimpan di {PATH_HASIL_CSV}")
