# generate_fixed_byte_emb.py
"""
Генерирует 2‑мерные детерминированные эмбеддинги для всех байтов (0–255).
Значения выбираются из набора: -2.0, -1.75, ..., 2.0 (шаг 0.25, 17 значений).
Каждому байту соответствует уникальная пара. Нулевой эмбеддинг (0.0, 0.0) не назначается.
Порядок: последовательный перебор, начиная с (-2.0, -2.0), пропуская (0.0, 0.0).
"""

import os
import torch
import numpy as np
import json

EMBEDDING_DIM = 2
OUTPUT_DIR = "byte_to_emb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALUE_SET = [round(-2.0 + i * 0.25, 2) for i in range(17)]
num_values = len(VALUE_SET)

# Генерируем все возможные пары индексов, исключая (8,8) -> (0.0, 0.0)
pairs = []
for i in range(num_values):
    for j in range(num_values):
        if i == 8 and j == 8:
            continue
        pairs.append((i, j))

assert len(pairs) >= 256, f"Недостаточно комбинаций (исключая нуль), доступно {len(pairs)}"

embeddings = torch.zeros(256, EMBEDDING_DIM)
for byte in range(256):
    i, j = pairs[byte]
    embeddings[byte, 0] = VALUE_SET[i]
    embeddings[byte, 1] = VALUE_SET[j]

# Проверка уникальности и отсутствия нулевого вектора
unique_vectors = torch.unique(embeddings, dim=0)
assert unique_vectors.size(0) == 256, "Ошибка: векторы не уникальны!"
assert not torch.all(embeddings == 0.0, dim=1).any(), "Нулевой вектор присутствует!"

# Сохранение
torch.save(embeddings, os.path.join(OUTPUT_DIR, "byte_embeddings.pt"))
np.save(os.path.join(OUTPUT_DIR, "byte_embeddings.npy"), embeddings.numpy())

byte_to_emb_dict = {byte: emb.tolist() for byte, emb in enumerate(embeddings)}
with open(os.path.join(OUTPUT_DIR, "byte_embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(byte_to_emb_dict, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "byte_to_emb_map.txt"), "w", encoding="utf-8") as f:
    f.write(f"Byte to embedding (dim={EMBEDDING_DIM})\n")
    f.write("=" * 50 + "\n")
    for byte in range(256):
        emb = embeddings[byte]
        emb_str = f"[{emb[0]:.2f}, {emb[1]:.2f}]"
        f.write(f"Byte {byte:3d} (0x{byte:02X}, '{chr(byte) if 32 <= byte < 127 else '?'}'): {emb_str}\n")

print(f"Эмбеддинги для 256 байтов сохранены в '{OUTPUT_DIR}' без нулевого вектора.")