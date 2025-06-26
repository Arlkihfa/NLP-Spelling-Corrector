import random
import pandas as pd

def generate_typo(word):
    if len(word) <= 3:
        return word
    typo_type = random.choice(['insert', 'delete', 'replace', 'swap'])
    i = random.randint(0, len(word) - 2)

    if typo_type == 'insert':
        return word[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[i:]
    elif typo_type == 'delete':
        return word[:i] + word[i+1:]
    elif typo_type == 'replace':
        return word[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[i+1:]
    elif typo_type == 'swap':
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    return word

# Load kata dari file KBBI
with open("data/kbbi_v.csv", encoding="utf-8") as f:
    kbbi_words = [line.strip().lower() for line in f if line.strip().isalpha()]

pairs = []
for idx, word in enumerate(kbbi_words):
    for _ in range(3):
        typo = generate_typo(word)
        if typo != word:
            pairs.append((typo, word))
    if idx % 1000 == 0:
        print(f"{idx} kata diproses...")

# Simpan hasil
df = pd.DataFrame(pairs, columns=["typo", "benar"])
df.to_csv("data/training_pairs.csv", index=False)
print(f"Selesai! Total pasangan: {len(pairs)}")
