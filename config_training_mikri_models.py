# config_training_mikri_models.py
import os
import torch

# ----------------------------------------------------------------------
# Размерности (всё крутится вокруг HIDDEN_DIM)
# ----------------------------------------------------------------------
INPUT_EMB_DIM = 16          # размерность эмбеддинга символа (из symbol_embeddings)
HIDDEN_DIM = 1024           # размерность pooled (выход энкодера и общая для всей системы)
HIDDEN_DIM_MIKRI = 1024     # скрытая размерность внутри MikriModel

# ----------------------------------------------------------------------
# Параметры архитектуры декодеров
# ----------------------------------------------------------------------
SYM_DECODER_NUM_LAYERS = 8
SYM_DECODER_PRE_MLP_LAYERS = 2
SYM_DECODER_MLP_MULTIPLIER = 4
NUM_OCTAVES = 4

DROPOUT = 0.1
DROPOUT_MIKRI = 0.1

# Параметры MikriModel
NUM_THOUGHT_BLOCKS = 8
NUM_MEMORY_SLOTS = 32
NUM_HEADS_MIKRI = 8

# ----------------------------------------------------------------------
# Устройство (общее, оставлено для совместимости)
# ----------------------------------------------------------------------
DEVICE_STR = "cuda"
if DEVICE_STR.startswith("cuda") and not torch.cuda.is_available():
    DEVICE_STR = "cpu"
DEVICE = torch.device(DEVICE_STR)

# ----------------------------------------------------------------------
# Устройства для каждой модели (можно задать индивидуально)
# По умолчанию: encoder на cuda:0, symbol decoder также
# ----------------------------------------------------------------------
ENCODER_DEVICE_STR = "cuda:0"
SYMBOL_DECODER_DEVICE_STR = "cuda:0"
TEMPORAL_MEMORY_DEVICE_STR = "cuda:0"
MIKRI_MODEL_DEVICE_STR = "cuda:0"

# Преобразуем строки в torch.device с проверкой доступности CUDA
def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: {device_str} requested but CUDA not available, falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_str)

ENCODER_DEVICE = _resolve_device(ENCODER_DEVICE_STR)
SYMBOL_DECODER_DEVICE = _resolve_device(SYMBOL_DECODER_DEVICE_STR)
TEMPORAL_MEMORY_DEVICE = _resolve_device(TEMPORAL_MEMORY_DEVICE_STR)
MIKRI_MODEL_DEVICE = _resolve_device(MIKRI_MODEL_DEVICE_STR)

# ----------------------------------------------------------------------
# Обучение Stage 1
# ----------------------------------------------------------------------
BATCH_SIZE = 1
LEARNING_RATE = 0.00001

SYM_LOSS_WEIGHT = 1.0
ENC_LOSS_WEIGHT = 1.0

# ----------------------------------------------------------------------
# Обучение Stage 2
# ----------------------------------------------------------------------
BATCH_SIZE_MIKRI = 1
LEARNING_RATE_MIKRI = 0.00001

STAGE2_SYM_LOSS_WEIGHT = 1.0         # вес символьной потери в Stage2
STAGE2_POOLED_LOSS_WEIGHT = 1.0      # вес pooled MSE

SAVE_EVERY_EPOCHS = 5
TEST_EVERY_EPOCHS = 20
TEST_MAX_GENERATED_LENGTH = 4096
MAX_CHECKPOINTS_PER_MODEL = 10

CLEAR_MODELS_ON_START = False
CLEAR_CACHE_EACH_BATCH = True

# ----------------------------------------------------------------------
# Пути
# ----------------------------------------------------------------------
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)
STAGE_FILE = f"{MODELS_DIR}/training_stage.txt"
RANDOM_SEED = 42

# Максимальная длина текстовой последовательности (символов) для энкодера и поз. эмбеддингов
MAX_SEQ_LEN = 4096

# Максимальное количество сообщений в истории для TemporalMemory
TEMP_MEM_MAX_SEQ_LEN = 256

# ----------------------------------------------------------------------
# Gradient Checkpointing
# ----------------------------------------------------------------------
USE_GRADIENT_CHECKPOINTING = False   # включить/отключить