# training_mikri_models.py
"""
Сервер обучения моделей через ZeroMQ PULL.
Поддерживает команды: train, train_stage1, train_stage2.
После обучения модели выгружаются из памяти.
"""

import os
import glob
import json
import gc
import re
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import zmq

from modelsEncoder_text_txt_emb_to_txt_emb_and_latent import Encoder
from modelsDecoder_text_txt_emb_to_txt_emb_and_latent import (
    SymbolDecoder
)
from mikri_models import MikriModel, TemporalMemoryModel
from config_training_mikri_models import *
from vocab import CHAR_TO_IDX, VOCAB_SIZE, CUSTOM_VOCAB

from training_mikri_models_Stage_1 import train_stage1, TextEmbDataset, collate_fn
from training_mikri_models_Stage_2 import train_stage2, UserBotPairDataset, pair_collate_fn

# ----------------------------------------------------------------------
# Константы путей
# ----------------------------------------------------------------------
CHAT_HISTORY_DIR = "chat_history"

# ----------------------------------------------------------------------
# Фиксируем seed для воспроизводимости
# ----------------------------------------------------------------------
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ----------------------------------------------------------------------
# Загрузка эмбеддингов символов (остаётся на CPU, потом при необходимости переносится)
# ----------------------------------------------------------------------
embeddings_path = "symbol_txt_emb_to_txt_emb/symbol_embeddings.pt"
if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"Symbol embeddings not found at {embeddings_path}. Run symbol_txt_emb_to_txt_emb.py first.")
symbol_embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)
vocab_size, emb_dim = symbol_embeddings.shape
assert emb_dim == INPUT_EMB_DIM, f"Embedding dim mismatch: {emb_dim} vs {INPUT_EMB_DIM}"

char_to_emb = {ch: symbol_embeddings[idx] for ch, idx in CHAR_TO_IDX.items()}
print(f"Loaded {len(char_to_emb)} symbol embeddings, dim={emb_dim}")

# ----------------------------------------------------------------------
# Функции загрузки данных из чата
# ----------------------------------------------------------------------
def load_texts_and_indices_from_chat_history(folder_path: str) -> Tuple[List[str], List[int]]:
    texts = []
    indices = []
    if not os.path.exists(folder_path):
        print(f"Warning: folder '{folder_path}' not found. Using dummy data.")
        return ["Hello world!", "Привет мир!", "Пример текста."], [0, 1, 2]

    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    if not json_files:
        print(f"Warning: No .json files found in '{folder_path}'. Using dummy data.")
        return ["Hello world!", "Привет мир!", "Пример текста."], [0, 1, 2]

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get("edited_content") or data.get("content")
                index = data.get("index")
                if text and isinstance(text, str) and index is not None:
                    texts.append(text.strip())
                    indices.append(index)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    paired = sorted(zip(indices, texts), key=lambda x: x[0])
    indices, texts = zip(*paired) if paired else ([], [])
    print(f"Loaded {len(texts)} text examples from chat history.")
    return list(texts), list(indices)


def load_all_messages_sorted(folder_path: str) -> List[Dict[str, Any]]:
    messages = []
    if not os.path.exists(folder_path):
        return messages
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                msg = json.load(f)
                role = msg.get('role')
                index = msg.get('index')
                text = msg.get('edited_content') or msg.get('content')
                if role and index is not None and text:
                    messages.append({
                        'index': index,
                        'role': role,
                        'text': text.strip()
                    })
        except:
            continue
    messages.sort(key=lambda x: x['index'])
    return messages


def text_to_embeddings_and_indices(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    embs, indices = [], []
    for ch in text:
        if ch in char_to_emb:
            embs.append(char_to_emb[ch])
            indices.append(CHAR_TO_IDX[ch])
    if not embs:
        return torch.empty(0, INPUT_EMB_DIM), torch.empty(0, dtype=torch.long)
    return torch.stack(embs, dim=0), torch.tensor(indices, dtype=torch.long)


# ----------------------------------------------------------------------
# Инициализация моделей на собственных устройствах
# ----------------------------------------------------------------------
def create_models():
    encoder = Encoder(
        input_emb_dim=INPUT_EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        use_checkpoint=USE_GRADIENT_CHECKPOINTING
    ).to(ENCODER_DEVICE)

    symbol_decoder = SymbolDecoder(
        hidden_dim=HIDDEN_DIM,
        symbol_emb_dim=INPUT_EMB_DIM,
        num_layers=SYM_DECODER_NUM_LAYERS,
        pre_mlp_layers=SYM_DECODER_PRE_MLP_LAYERS,
        mlp_multiplier=SYM_DECODER_MLP_MULTIPLIER,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN,
        num_octaves=NUM_OCTAVES,
        use_checkpoint=USE_GRADIENT_CHECKPOINTING,
        vocab_size=VOCAB_SIZE                 # <-- добавлен параметр
    ).to(SYMBOL_DECODER_DEVICE)

    temporal_memory = TemporalMemoryModel(
        hidden_dim=HIDDEN_DIM,
        num_layers=4,
        num_heads=8,
        dropout=DROPOUT_MIKRI,
        max_seq_len=TEMP_MEM_MAX_SEQ_LEN,
        use_checkpoint=USE_GRADIENT_CHECKPOINTING
    ).to(TEMPORAL_MEMORY_DEVICE)

    mikri_model = MikriModel(
        hidden_dim=HIDDEN_DIM_MIKRI,
        num_thought_blocks=NUM_THOUGHT_BLOCKS,
        num_memory_slots=NUM_MEMORY_SLOTS,
        num_heads=NUM_HEADS_MIKRI,
        dropout=DROPOUT_MIKRI,
        use_checkpoint=USE_GRADIENT_CHECKPOINTING
    ).to(MIKRI_MODEL_DEVICE)

    return encoder, symbol_decoder, temporal_memory, mikri_model


# ----------------------------------------------------------------------
# Функции для работы с чекпоинтами
# ----------------------------------------------------------------------
def get_model_path(model_name: str, stage: int, epoch: int) -> str:
    return os.path.join(MODELS_DIR, f"{model_name}_stage{stage}_epoch{epoch}.pth")


def find_latest_checkpoint(model_name: str, stage: int) -> Optional[Tuple[str, int]]:
    pattern = f"{model_name}_stage{stage}_epoch*.pth"
    files = glob.glob(os.path.join(MODELS_DIR, pattern))
    if not files:
        return None
    def extract_epoch(fname):
        match = re.search(r'epoch(\d+)', fname)
        return int(match.group(1)) if match else -1
    latest = max(files, key=extract_epoch)
    epoch = extract_epoch(latest)
    return latest, epoch


def cleanup_old_checkpoints(model_name: str, stage: int, keep_last: int):
    if keep_last <= 0:
        return
    pattern = f"{model_name}_stage{stage}_epoch*.pth"
    files = glob.glob(os.path.join(MODELS_DIR, pattern))
    if len(files) <= keep_last:
        return
    def extract_epoch(fname):
        match = re.search(r'epoch(\d+)', fname)
        return int(match.group(1)) if match else -1
    files.sort(key=extract_epoch, reverse=True)
    for old_file in files[keep_last:]:
        try:
            os.remove(old_file)
            print(f"Removed old checkpoint: {os.path.basename(old_file)}")
        except OSError as e:
            print(f"Error removing {old_file}: {e}")


def save_checkpoint(encoder, symbol_decoder,
                    temporal_memory, mikri_model, stage: int, epoch: int):
    os.makedirs(MODELS_DIR, exist_ok=True)
    models_to_save = []
    if encoder is not None:
        torch.save(encoder.state_dict(), get_model_path('encoder', stage, epoch))
        models_to_save.append('encoder')
    if symbol_decoder is not None:
        torch.save(symbol_decoder.state_dict(), get_model_path('symbol_decoder', stage, epoch))
        models_to_save.append('symbol_decoder')
    if temporal_memory is not None:
        torch.save(temporal_memory.state_dict(), get_model_path('temporal_memory', stage, epoch))
        models_to_save.append('temporal_memory')
    if mikri_model is not None:
        torch.save(mikri_model.state_dict(), get_model_path('mikri_model', stage, epoch))
        models_to_save.append('mikri_model')
    print(f"Checkpoint saved for stage {stage} epoch {epoch}")
    if MAX_CHECKPOINTS_PER_MODEL > 0:
        for model_name in models_to_save:
            cleanup_old_checkpoints(model_name, stage, MAX_CHECKPOINTS_PER_MODEL)


def load_models_if_exist(encoder, symbol_decoder,
                         temporal_memory, mikri_model):
    models_info = [
        ('encoder', encoder, 1, ENCODER_DEVICE),
        ('symbol_decoder', symbol_decoder, 1, SYMBOL_DECODER_DEVICE),
        ('temporal_memory', temporal_memory, 2, TEMPORAL_MEMORY_DEVICE),
        ('mikri_model', mikri_model, 2, MIKRI_MODEL_DEVICE)
    ]
    for name, model, stage, dev in models_info:
        if model is None:
            continue
        latest = find_latest_checkpoint(name, stage)
        if latest:
            path, epoch = latest
            state_dict = torch.load(path, map_location=dev, weights_only=False)
            model.load_state_dict(state_dict)
            print(f"Loaded {name} from epoch {epoch} ({path})")


# ----------------------------------------------------------------------
# Основные функции обучения
# ----------------------------------------------------------------------
def run_training():
    if CLEAR_MODELS_ON_START:
        for f in glob.glob(os.path.join(MODELS_DIR, "*.pth")):
            os.remove(f)
        print("Cleared old model files.")

    encoder, symbol_decoder, temporal_memory, mikri_model = create_models()
    load_models_if_exist(encoder, symbol_decoder,
                         temporal_memory, mikri_model)

    texts, indices = load_texts_and_indices_from_chat_history(CHAT_HISTORY_DIR)
    if not texts:
        texts = ["Hello world!", "Привет мир!", "Пример текста."]
        indices = [0, 1, 2]

    dataset = TextEmbDataset(texts, indices, char_to_emb, INPUT_EMB_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, pin_memory=True)

    all_msgs = load_all_messages_sorted(CHAT_HISTORY_DIR)
    pair_dataset = UserBotPairDataset([], all_msgs, char_to_emb)
    pair_loader = DataLoader(pair_dataset, batch_size=BATCH_SIZE_MIKRI, shuffle=True,
                             collate_fn=pair_collate_fn, pin_memory=True)

    if os.path.exists(STAGE_FILE):
        os.remove(STAGE_FILE)

    print("Starting Stage 1 training...")
    train_stage1(
        encoder, symbol_decoder,
        dataloader, symbol_embeddings,
        ENCODER_DEVICE, SYMBOL_DECODER_DEVICE,
        save_checkpoint_callback=lambda ep: save_checkpoint(
            encoder, symbol_decoder,
            None, None, stage=1, epoch=ep
        )
    )

    with open(STAGE_FILE, 'w') as f:
        f.write('2')

    print("Stage 1 completed. Starting Stage 2 training...")
    train_stage2(
        encoder, symbol_decoder,
        temporal_memory, mikri_model,
        pair_loader, symbol_embeddings,
        ENCODER_DEVICE, SYMBOL_DECODER_DEVICE,
        TEMPORAL_MEMORY_DEVICE, MIKRI_MODEL_DEVICE,
        save_checkpoint_callback=lambda ep: save_checkpoint(
            None, None,
            temporal_memory, mikri_model, stage=2, epoch=ep
        )
    )
    print("Full training completed.")

    del encoder, symbol_decoder, temporal_memory, mikri_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Models unloaded. Waiting for next command...")


def run_training_stage1():
    if CLEAR_MODELS_ON_START:
        for f in glob.glob(os.path.join(MODELS_DIR, "*.pth")):
            os.remove(f)
        print("Cleared old model files.")

    encoder, symbol_decoder, _, _ = create_models()
    load_models_if_exist(encoder, symbol_decoder, None, None)

    texts, indices = load_texts_and_indices_from_chat_history(CHAT_HISTORY_DIR)
    if not texts:
        texts = ["Hello world!", "Привет мир!", "Пример текста."]
        indices = [0, 1, 2]

    dataset = TextEmbDataset(texts, indices, char_to_emb, INPUT_EMB_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, pin_memory=True)

    if os.path.exists(STAGE_FILE):
        os.remove(STAGE_FILE)

    print("Starting Stage 1 training...")
    train_stage1(
        encoder, symbol_decoder,
        dataloader, symbol_embeddings,
        ENCODER_DEVICE, SYMBOL_DECODER_DEVICE,
        save_checkpoint_callback=lambda ep: save_checkpoint(
            encoder, symbol_decoder,
            None, None, stage=1, epoch=ep
        )
    )
    print("Stage 1 completed.")

    del encoder, symbol_decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Models unloaded. Waiting for next command...")


def run_training_stage2():
    encoder, symbol_decoder, temporal_memory, mikri_model = create_models()
    load_models_if_exist(encoder, symbol_decoder,
                         temporal_memory, mikri_model)

    all_msgs = load_all_messages_sorted(CHAT_HISTORY_DIR)
    if not all_msgs:
        raise RuntimeError("No messages found. Cannot train Stage 2.")
    pair_dataset = UserBotPairDataset([], all_msgs, char_to_emb)
    pair_loader = DataLoader(pair_dataset, batch_size=BATCH_SIZE_MIKRI, shuffle=True,
                             collate_fn=pair_collate_fn, pin_memory=True)

    print("Starting Stage 2 training...")
    train_stage2(
        encoder, symbol_decoder,
        temporal_memory, mikri_model,
        pair_loader, symbol_embeddings,
        ENCODER_DEVICE, SYMBOL_DECODER_DEVICE,
        TEMPORAL_MEMORY_DEVICE, MIKRI_MODEL_DEVICE,
        save_checkpoint_callback=lambda ep: save_checkpoint(
            None, None,
            temporal_memory, mikri_model, stage=2, epoch=ep
        )
    )
    print("Stage 2 completed.")

    del encoder, symbol_decoder, temporal_memory, mikri_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Models unloaded. Waiting for next command...")


# ----------------------------------------------------------------------
# Сервер ZMQ
# ----------------------------------------------------------------------
def main():
    print("Training server started. ZeroMQ PULL on tcp://127.0.0.1:5555")
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://127.0.0.1:5555")

    while True:
        try:
            message = socket.recv_string()
            if message == "train":
                print("Received 'train' command. Starting full training...")
                run_training()
            elif message == "train_stage1":
                print("Received 'train_stage1' command. Starting Stage1 only...")
                run_training_stage1()
            elif message == "train_stage2":
                print("Received 'train_stage2' command. Starting Stage2 only...")
                run_training_stage2()
            else:
                print(f"Unknown command: {message}")
        except KeyboardInterrupt:
            print("Shutting down...")
            break
        except Exception as e:
            print(f"Error during training: {e}")

    socket.close()
    context.term()


if __name__ == "__main__":
    main()