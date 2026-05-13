# generate_fixed_symbol_emb.py
"""
Генерирует 16‑мерные случайные эмбеддинги для всех символов из CUSTOM_VOCAB.
Каждая координата выбирается случайно из сетки [MIN_VAL ... MAX_VAL] с шагом STEP,
из которой исключено значение 0. Таким образом, ни одна координата эмбеддинга не равна 0.
Параметры сетки и seed сохраняются в JSON для воспроизводимости.
"""

import os, json, random
import torch
import numpy as np
from vocab import CUSTOM_VOCAB, CHAR_TO_IDX

OUTPUT_DIR = "symbol_txt_emb_to_txt_emb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_EMB_DIM = 16
MIN_VAL, MAX_VAL = -5.0, 5.0
STEP = 5
SEED = 42

# ----------------------------------------------------------------------
# Фиксируем источники случайности
# ----------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# Сетка значений (без нуля)
# ----------------------------------------------------------------------
VALUES = np.arange(MIN_VAL, MAX_VAL + STEP, STEP)
VALUES = VALUES[VALUES != 0]          # исключаем 0
M = len(VALUES)                       # количество доступных значений

# ----------------------------------------------------------------------
# Генерация уникальных случайных эмбеддингов
# ----------------------------------------------------------------------
while True:
    embeddings = torch.zeros(len(CUSTOM_VOCAB), TEXT_EMB_DIM)
    for i, sym in enumerate(CUSTOM_VOCAB):
        # Случайные индексы в сетке для каждой координаты
        indices = np.random.randint(0, M, size=TEXT_EMB_DIM)
        embeddings[i] = torch.tensor(VALUES[indices], dtype=torch.float32)

    # Проверка уникальности
    unique_set = set(tuple(v.tolist()) for v in embeddings)
    if len(unique_set) == len(CUSTOM_VOCAB):
        # Дополнительная проверка: ни один компонент не равен нулю (гарантировано построением)
        if not (embeddings == 0.0).any():
            break
        else:
            print("Обнаружен нулевой компонент (не должно случиться), повторная генерация...")
    else:
        print(f"Коллизии ({len(CUSTOM_VOCAB) - len(unique_set)} шт.), повторная генерация...")

# ----------------------------------------------------------------------
# Сохранение эмбеддингов в стандартных форматах
# ----------------------------------------------------------------------
torch.save(embeddings, os.path.join(OUTPUT_DIR, "symbol_embeddings.pt"))
np.save(os.path.join(OUTPUT_DIR, "symbol_embeddings.npy"), embeddings.numpy())

json_dict = {sym: embeddings[CHAR_TO_IDX[sym]].tolist() for sym in CUSTOM_VOCAB}
with open(os.path.join(OUTPUT_DIR, "symbol_embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(json_dict, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "symbol_emb_map.txt"), "w", encoding="utf-8") as f:
    f.write(f"Random symbol embeddings (dim={TEXT_EMB_DIM})\n")
    f.write(f"Seed={SEED}, MIN={MIN_VAL}, MAX={MAX_VAL}, STEP={STEP}, Zero excluded\n")
    f.write("=" * 60 + "\n")
    for sym, emb in json_dict.items():
        f.write(f"Symbol '{sym}': {emb}\n")

# ----------------------------------------------------------------------
# Сохраняем параметры генерации отдельно
# ----------------------------------------------------------------------
params = {
    "MIN_VAL": MIN_VAL,
    "MAX_VAL": MAX_VAL,
    "STEP": STEP,
    "SEED": SEED,
    "zero_excluded": True
}
with open(os.path.join(OUTPUT_DIR, "symbol_embeddings_params.json"), "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)

print(f"Созданы случайные 16‑мерные эмбеддинги для {len(CUSTOM_VOCAB)} символов (без нулевых координат).")
print(f"Параметры (MIN={MIN_VAL}, MAX={MAX_VAL}, STEP={STEP}, SEED={SEED}) сохранены в {OUTPUT_DIR}/symbol_embeddings_params.json")