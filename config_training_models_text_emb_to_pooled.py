# config_training_models_text_emb_to_pooled.py
"""
Конфигурация для обучения энкодера + SymbolDecoder на текстовых данных из CSV.
Каждая строка полей question, context, answer — независимый пример.
"""

import os
import torch

# ----------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------
CSV_FILE = "./text_dataset/MTSBerquad_LFQA_Dataset_Pretty.csv"
MAX_TEXTS = 0  # None = использовать все доступные

# ----------------------------------------------------------------------
# Размерности
# ----------------------------------------------------------------------
INPUT_EMB_DIM = 16                # размерность символьного эмбеддинга
HIDDEN_DIM = 1024                # размерность pooled‑вектора (выход энкодера)

# ----------------------------------------------------------------------
# Параметры Encoder
# ----------------------------------------------------------------------
ENCODER_NUM_LAYERS = 8            # число TransformerBlock
ENCODER_NUM_HEADS = 8             # число голов в self‑attention блоков
ENCODER_FF_MULTIPLIER = 8         # множитель скрытого слоя FFN (hidden_dim * multiplier)
ENCODER_DROPOUT = 0.1             # можно оставить общим, здесь для гибкости
ENCODER_USE_CHECKPOINT = False    # gradient checkpointing для Encoder

# ----------------------------------------------------------------------
# Параметры SymbolDecoder
# ----------------------------------------------------------------------
SYM_DECODER_NUM_LAYERS = 16        # число проходов через разделяемый MLP
SYM_DECODER_PRE_MLP_LAYERS = 4    # предварительные полносвязные слои
SYM_DECODER_MLP_MULTIPLIER = 8    # множитель внутреннего измерения MLP
NUM_OCTAVES = 4                   # количество октав в позиционном кодировании
SYM_DECODER_DROPOUT = 0.1
SYM_DECODER_USE_CHECKPOINT = False

# ----------------------------------------------------------------------
# Общие параметры обучения
# ----------------------------------------------------------------------
BATCH_SIZE = 1          # размер батча (обычно 1 для таких моделей)

LEARNING_RATE = 0.00001

SYM_LOSS_WEIGHT = 1.0
ENC_LOSS_WEIGHT = 1.0

SAVE_EVERY_EPOCHS = 5
CLEAR_CACHE_EACH_BATCH = True
RANDOM_SEED = 42
MAX_SEQ_LEN = 4096

# ----------------------------------------------------------------------
# Устройства
# ----------------------------------------------------------------------
ENCODER_DEVICE_STR = "cuda:0"
SYMBOL_DECODER_DEVICE_STR = "cuda:1"

def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: {device_str} not available, falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_str)

ENCODER_DEVICE = _resolve_device(ENCODER_DEVICE_STR)
SYMBOL_DECODER_DEVICE = _resolve_device(SYMBOL_DECODER_DEVICE_STR)

# ----------------------------------------------------------------------
# Дополнительные данные
# ----------------------------------------------------------------------
USE_CHAT_HISTORY = True
CHAT_HISTORY_DIR = "chat_history"

USE_VALIDATION = True
NUM_VAL_TEXTS = 5
VAL_EVERY_EPOCHS = 5

# ----------------------------------------------------------------------
# Пути
# ----------------------------------------------------------------------
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)
