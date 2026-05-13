# training_models_text_emb_to_pooled.py
"""
Обучение энкодера + SymbolDecoder на текстовых данных из CSV.
Декодер выдаёт вероятностное распределение над символами + токен конца.
Потеря: сравнение векторов как единого целого (скалярное произведение с целевым и нецелевыми one‑hot векторами).
При ошибке argmax потеря удваивается.
"""

import os
import csv
import gc
import re
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

from modelsEncoder_text_txt_emb_to_txt_emb_and_latent import Encoder
from modelsDecoder_text_txt_emb_to_txt_emb_and_latent import SymbolDecoder
from config_training_models_text_emb_to_pooled import *
from vocab import CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE

# ----------------------------------------------------------------------
# Параметры (старые не используются)
# ----------------------------------------------------------------------
NULL_INDEX = VOCAB_SIZE          # индекс токена конца последовательности


# ========== Новая функция потерь ==========
def compute_symbol_loss_vector(pred_probs, target_idx):
    """
    Сравнивает предсказанный вектор вероятностей с целевым one‑hot вектором
    КАК ЕДИНОЕ ЦЕЛОЕ (через скалярное произведение).

    pred_probs: (B, V+1) – вероятности после softmax
    target_idx: (B,) – индекс истинного класса (0..V)
    Возвращает среднюю потерю с удвоением при ошибке argmax.
    """
    V_plus_1 = pred_probs.size(-1)
    # Целевой one‑hot вектор той же размерности
    target_onehot = F.one_hot(target_idx, num_classes=V_plus_1).float()  # (B, V+1)

    # Скалярное произведение (сходство) с целевым вектором = pred_target
    pos_similarity = (pred_probs * target_onehot).sum(dim=1)            # (B,)

    # Скалярное произведение с каждым нецелевым вектором, усреднённое
    neg_similarities = pred_probs * (1.0 - target_onehot)               # (B, V+1)
    # Среднее сходство с нецелевыми векторами (делим на V, число нецелевых классов)
    neg_similarity = neg_similarities.sum(dim=1) / max(1, V_plus_1 - 1)

    # Потеря 1: чем ближе pos_similarity к 1, тем меньше потеря
    loss1 = 1.0 - pos_similarity
    # Потеря 2: чем больше среднее сходство с нецелевыми, тем больше потеря
    loss2 = neg_similarity

    loss = loss1 + loss2                     # (B,)

    # Удвоение, если argmax не совпадает с целью
    wrong_mask = (pred_probs.argmax(dim=-1) != target_idx).float()
    loss = loss * (1.0 + wrong_mask)         # x2 при ошибке

    return loss.mean()


def is_stop_probs(pred_probs: torch.Tensor) -> bool:
    """True, если предсказан NULL_INDEX (токен конца)."""
    return bool((pred_probs.argmax(dim=-1) == NULL_INDEX).item())


# ==============================================================================
# 1. Загрузка символьных эмбеддингов (нужны только для входных эмбеддингов)
# ==============================================================================
embeddings_path = "symbol_txt_emb_to_txt_emb/symbol_embeddings.pt"
if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"Symbol embeddings not found at {embeddings_path}")
symbol_embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)
symbol_embeddings = symbol_embeddings.to(ENCODER_DEVICE)
vocab_size, emb_dim = symbol_embeddings.shape
assert emb_dim == INPUT_EMB_DIM, f"Embedding dim mismatch: {emb_dim} vs {INPUT_EMB_DIM}"
char_to_emb = {ch: symbol_embeddings[idx] for ch, idx in CHAR_TO_IDX.items()}
print(f"Loaded {len(char_to_emb)} symbol embeddings, dim={emb_dim}")


# ==============================================================================
# 2. Загрузка текстовых данных из CSV (с индексами)
# ==============================================================================
def load_texts_from_csv(file_path: str, max_texts: int = None) -> List[Tuple[int, str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    texts_with_idx = []
    idx = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in ['question', 'context', 'answer']:
                text = row.get(field, '').strip()
                if text:
                    texts_with_idx.append((idx, text))
                    idx += 1
                    if max_texts is not None and len(texts_with_idx) >= max_texts:
                        return texts_with_idx
    print(f"Loaded {len(texts_with_idx)} text examples from {file_path}")
    return texts_with_idx


def load_texts_and_indices_from_chat_history(folder_path: str) -> List[Tuple[int, str]]:
    texts_with_idx = []
    if not os.path.exists(folder_path):
        print(f"Warning: folder '{folder_path}' not found.")
        return texts_with_idx
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get("edited_content") or data.get("content")
                index = data.get("index")
                if text and isinstance(text, str) and index is not None:
                    texts_with_idx.append((index, text.strip()))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    texts_with_idx.sort(key=lambda x: x[0])
    print(f"Loaded {len(texts_with_idx)} messages from chat history.")
    return texts_with_idx


# Загрузка CSV с учётом валидации
total_csv_needed = MAX_TEXTS + (NUM_VAL_TEXTS if USE_VALIDATION else 0)
texts_with_idx_csv = load_texts_from_csv(CSV_FILE, max_texts=total_csv_needed)

if USE_VALIDATION and NUM_VAL_TEXTS > 0:
    if len(texts_with_idx_csv) < total_csv_needed:
        print(f"⚠️ CSV содержит всего {len(texts_with_idx_csv)} примеров, "
              f"ожидалось {total_csv_needed}. Валидация будет урезана.")
        val_count = min(NUM_VAL_TEXTS, len(texts_with_idx_csv))
        val_texts_with_idx = texts_with_idx_csv[-val_count:]
        train_texts_with_idx_csv = texts_with_idx_csv[:min(MAX_TEXTS, len(texts_with_idx_csv) - val_count)]
    else:
        val_texts_with_idx = texts_with_idx_csv[-NUM_VAL_TEXTS:]
        train_texts_with_idx_csv = texts_with_idx_csv[:MAX_TEXTS]
else:
    val_texts_with_idx = []
    train_texts_with_idx_csv = texts_with_idx_csv[:MAX_TEXTS]

# Добавляем чат-историю только к обучающей выборке
if USE_CHAT_HISTORY:
    texts_with_idx_chat = load_texts_and_indices_from_chat_history(CHAT_HISTORY_DIR)
    if texts_with_idx_chat:
        max_csv_idx = max((idx for idx, _ in train_texts_with_idx_csv), default=-1)
        offset = max_csv_idx + 1
        texts_with_idx_chat = [(orig_idx + offset, txt) for orig_idx, txt in texts_with_idx_chat]
        train_texts_with_idx = train_texts_with_idx_csv + texts_with_idx_chat
    else:
        train_texts_with_idx = train_texts_with_idx_csv
else:
    train_texts_with_idx = train_texts_with_idx_csv

indices_train, texts_train = zip(*train_texts_with_idx) if train_texts_with_idx else ([], [])
indices_val,   texts_val   = zip(*val_texts_with_idx)   if val_texts_with_idx   else ([], [])
print(f"Обучающих примеров: {len(texts_train)}")
if texts_val:
    print(f"Валидационных примеров: {len(texts_val)}")


class TextEmbDataset(Dataset):
    def __init__(self, indices: List[int], texts: List[str]):
        self.examples = []
        for idx, t in zip(indices, texts):
            embs, char_indices = [], []
            for ch in t:
                if ch in char_to_emb:
                    embs.append(char_to_emb[ch])
                    char_indices.append(CHAR_TO_IDX[ch])
            if embs:
                emb_seq = torch.stack(embs, dim=0)
                idx_seq = torch.tensor(char_indices, dtype=torch.long)
                self.examples.append((emb_seq, idx_seq, len(emb_seq), idx))
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def collate_fn(batch):
    emb_seqs, idx_seqs, lengths, msg_indices = zip(*batch)
    new_lengths = [L + 1 for L in lengths]   # +1 для нулевого токена
    max_len = max(new_lengths)
    batch_size = len(batch)

    batch_emb = torch.zeros(batch_size, max_len, INPUT_EMB_DIM)
    batch_idx = torch.full((batch_size, max_len), -1, dtype=torch.long)
    batch_pos = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, (emb, idx, L) in enumerate(zip(emb_seqs, idx_seqs, lengths)):
        # оригинальные символы (позиции 1..L)
        batch_emb[i, :L] = emb
        batch_idx[i, :L] = idx
        batch_pos[i, :L] = torch.arange(1, L + 1)
        # нулевой токен в позиции L
        batch_emb[i, L] = torch.zeros(INPUT_EMB_DIM)
        batch_idx[i, L] = NULL_INDEX
        batch_pos[i, L] = L + 1

    return (
        batch_emb,
        batch_idx,
        batch_pos,
        torch.tensor(new_lengths, dtype=torch.long),
        torch.tensor(msg_indices, dtype=torch.long)
    )


train_dataset = TextEmbDataset(indices_train, texts_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, pin_memory=True)
print(f"Train DataLoader: {len(train_loader)} batches")

val_loader = None
if USE_VALIDATION and len(texts_val) > 0:
    val_dataset = TextEmbDataset(indices_val, texts_val)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, pin_memory=True)
    print(f"Validation DataLoader: {len(val_loader)} batches")


# ==============================================================================
# 3. Инициализация моделей
# ==============================================================================
encoder = Encoder(
    input_emb_dim=INPUT_EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=ENCODER_NUM_LAYERS,
    num_heads=ENCODER_NUM_HEADS,
    ff_multiplier=ENCODER_FF_MULTIPLIER,
    dropout=ENCODER_DROPOUT,
    max_seq_len=MAX_SEQ_LEN,
    use_checkpoint=ENCODER_USE_CHECKPOINT
).to(ENCODER_DEVICE)

symbol_decoder = SymbolDecoder(
    hidden_dim=HIDDEN_DIM,
    symbol_emb_dim=INPUT_EMB_DIM,
    num_layers=SYM_DECODER_NUM_LAYERS,
    pre_mlp_layers=SYM_DECODER_PRE_MLP_LAYERS,
    mlp_multiplier=SYM_DECODER_MLP_MULTIPLIER,
    dropout=SYM_DECODER_DROPOUT,
    max_len=MAX_SEQ_LEN,
    num_octaves=NUM_OCTAVES,
    use_checkpoint=SYM_DECODER_USE_CHECKPOINT,
    vocab_size=VOCAB_SIZE
).to(SYMBOL_DECODER_DEVICE)


# ==============================================================================
# 4. Работа с чекпоинтами
# ==============================================================================
MAX_CHECKPOINTS_PER_MODEL = 5

def get_model_path(model_name: str, epoch: int) -> str:
    return os.path.join(MODELS_DIR, f"{model_name}_epoch{epoch}.pth")

def find_latest_checkpoint(model_name: str) -> Optional[Tuple[str, int]]:
    pattern = f"{model_name}_epoch*.pth"
    files = glob.glob(os.path.join(MODELS_DIR, pattern))
    if not files:
        return None
    def extract_epoch(fname):
        match = re.search(r'epoch(\d+)', fname)
        return int(match.group(1)) if match else -1
    latest = max(files, key=extract_epoch)
    epoch = extract_epoch(latest)
    return latest, epoch

def cleanup_old_checkpoints(model_name: str, keep_last: int = MAX_CHECKPOINTS_PER_MODEL):
    pattern = f"{model_name}_epoch*.pth"
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

def save_checkpoints(epoch: int):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(encoder.state_dict(), get_model_path('encoder', epoch))
    torch.save(symbol_decoder.state_dict(), get_model_path('symbol_decoder', epoch))
    for name in ['encoder', 'symbol_decoder']:
        cleanup_old_checkpoints(name)

def load_checkpoints_if_exist() -> int:
    models_info = [
        ('encoder', encoder, ENCODER_DEVICE),
        ('symbol_decoder', symbol_decoder, SYMBOL_DECODER_DEVICE)
    ]
    loaded_epoch = 0
    for name, model, device in models_info:
        latest = find_latest_checkpoint(name)
        if latest:
            path, epoch = latest
            state_dict = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print(f"Loaded {name} from epoch {epoch} ({path})")
            if loaded_epoch == 0:
                loaded_epoch = epoch
            else:
                assert epoch == loaded_epoch, f"Mismatched epochs: {name} has epoch {epoch}, expected {loaded_epoch}"
    if loaded_epoch > 0:
        print(f"Resuming from epoch {loaded_epoch}")
    return loaded_epoch


# ==============================================================================
# 5. Метрики и генерация (адаптированы к вероятностному выходу)
# ==============================================================================
def evaluate_batch(encoder, symbol_dec, batch_emb, batch_idx, batch_pos, lengths,
                   dev_enc, dev_sym):
    was_training = [m.training for m in [encoder, symbol_dec]]
    encoder.eval(); symbol_dec.eval()
    try:
        with torch.no_grad():
            batch_emb = batch_emb.to(dev_enc)
            batch_idx = batch_idx.to(dev_enc)
            batch_pos = batch_pos.to(dev_enc)
            lengths = lengths.to(dev_enc)
            pooled = encoder(batch_emb, lengths, batch_pos)

            batch_size = batch_emb.size(0)
            total_chars = correct_chars = total_examples = correct_examples = 0
            for b in range(batch_size):
                seq_len = lengths[b].item()   # L+1
                orig_len = seq_len - 1
                generated_indices = []
                for pos in range(orig_len):
                    pos_tensor = torch.tensor([pos], device=dev_sym)
                    pooled_sym = pooled[b:b+1].to(dev_sym)
                    sym_out = symbol_dec(pooled_sym, pos_tensor)
                    probs = sym_out['symbol_probs'][0]   # (V+1,)
                    pred_idx = probs.argmax().item()
                    generated_indices.append(pred_idx)

                target_indices = batch_idx[b, :orig_len].tolist()
                for i in range(orig_len):
                    if generated_indices[i] == target_indices[i]: correct_chars += 1
                    total_chars += 1
                if generated_indices == target_indices: correct_examples += 1
                total_examples += 1
            char_acc = correct_chars / total_chars if total_chars else 0.0
            example_acc = correct_examples / total_examples if total_examples else 0.0
            return char_acc, example_acc
    finally:
        for m, was in zip([encoder, symbol_dec], was_training):
            if was: m.train()
        del batch_emb, batch_idx, batch_pos, lengths, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_sequential(encoder, symbol_dec, loader, dev_enc, dev_sym):
    encoder.eval(); symbol_dec.eval()
    total_chars = correct_chars = total_examples = correct_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch_emb, batch_idx, batch_pos, lengths, _ = batch
            batch_emb = batch_emb.to(dev_enc); batch_idx = batch_idx.to(dev_enc)
            batch_pos = batch_pos.to(dev_enc); lengths = lengths.to(dev_enc)
            pooled = encoder(batch_emb, lengths, batch_pos)
            for b in range(batch_emb.size(0)):
                seq_len = lengths[b].item()
                orig_len = seq_len - 1
                generated_indices = []
                for pos in range(orig_len):
                    pos_tensor = torch.tensor([pos], device=dev_sym)
                    pooled_sym = pooled[b:b+1].to(dev_sym)
                    sym_out = symbol_dec(pooled_sym, pos_tensor)
                    probs = sym_out['symbol_probs'][0]
                    pred_idx = probs.argmax().item()
                    generated_indices.append(pred_idx)
                target_indices = batch_idx[b, :orig_len].tolist()
                for i in range(orig_len):
                    if generated_indices[i] == target_indices[i]: correct_chars += 1
                    total_chars += 1
                if generated_indices == target_indices: correct_examples += 1
                total_examples += 1
            del batch_emb, batch_idx, batch_pos, lengths, pooled
    char_acc = correct_chars / total_chars if total_chars else 0.0
    example_acc = correct_examples / total_examples if total_examples else 0.0
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return char_acc, example_acc


def generate_text_until_stop(pooled, symbol_dec, dev_sym, max_len=MAX_SEQ_LEN):
    """Генерирует символы, пока не предсказан NULL_INDEX или не достигнута max_len."""
    generated_indices = []
    for pos in range(max_len):
        pos_tensor = torch.tensor([pos], device=dev_sym)
        pooled_sym = pooled.to(dev_sym)
        sym_out = symbol_dec(pooled_sym, pos_tensor)
        probs = sym_out['symbol_probs'][0]   # (V+1,)
        if probs.argmax().item() == NULL_INDEX:
            break
        pred_idx = probs.argmax().item()
        generated_indices.append(pred_idx)
    return ''.join([IDX_TO_CHAR[idx] for idx in generated_indices])


@torch.no_grad()
def validate(encoder, symbol_dec, loader, dev_enc, dev_sym) -> Dict[str, float]:
    encoder.eval(); symbol_dec.eval()
    total_loss_sym = total_loss_enc = 0.0
    total_batches = 0
    total_correct = 0; total_positions = 0

    for batch_emb, batch_idx, batch_pos, lengths, _ in loader:
        batch_emb = batch_emb.to(dev_enc); batch_idx = batch_idx.to(dev_enc)
        batch_pos = batch_pos.to(dev_enc); lengths = lengths.to(dev_enc)
        max_len = batch_emb.size(1)

        pooled = encoder(batch_emb, lengths, batch_pos)

        total_sym = 0.0; total_sym_pos = 0
        batch_correct = 0; total_pos = 0

        for pos in range(max_len):
            active = lengths > pos
            if not active.any():
                continue
            num_active = active.sum().item()
            target_idx = batch_idx[active, pos].to(dev_sym)
            pooled_sel = pooled[active].to(dev_sym)
            sym_out = symbol_dec(pooled_sel,
                                 torch.full((num_active,), pos, device=dev_sym))
            # Новая потеря
            pos_loss = compute_symbol_loss_vector(sym_out['symbol_probs'], target_idx)
            total_sym += pos_loss.item() * num_active
            total_sym_pos += num_active

            # Accuracy (только для символов)
            symbol_mask = target_idx != NULL_INDEX
            if symbol_mask.any():
                symbol_mask_enc = symbol_mask.to(dev_enc)
                probs = sym_out['symbol_probs'][symbol_mask]
                pred_idx = probs.argmax(dim=-1)
                correct = (pred_idx.to(dev_enc) == batch_idx[active][symbol_mask_enc, pos].to(dev_enc)).sum().item()
                batch_correct += correct
                total_pos += symbol_mask.sum().item()

        weighted_sym = SYM_LOSS_WEIGHT * (total_sym / total_sym_pos) if total_sym_pos > 0 else 0.0
        loss_enc_total = torch.tensor(weighted_sym, device=dev_enc)
        weighted_loss_enc = ENC_LOSS_WEIGHT * loss_enc_total

        total_loss_sym += weighted_sym
        total_loss_enc += weighted_loss_enc.item()
        total_batches += 1
        total_correct += batch_correct
        total_positions += total_pos

    avg_sym = total_loss_sym / total_batches if total_batches else 0.0
    avg_enc = total_loss_enc / total_batches if total_batches else 0.0
    pos_acc = total_correct / total_positions if total_positions else 0.0

    encoder.train(); symbol_dec.train()
    return {'loss_sym': avg_sym, 'loss_enc': avg_enc, 'pos_acc': pos_acc}


# ==============================================================================
# 6. Основной цикл обучения (с новой потерей)
# ==============================================================================
def train_epoch_sequential(encoder, symbol_dec, dataloader,
                           opt_sym, opt_enc,
                           epoch, dev_enc, dev_sym):
    encoder.train(); symbol_dec.train()
    total_loss_sym = 0.0; total_loss_enc = 0.0
    total_correct = 0; total_positions = 0
    num_batches = len(dataloader)

    for batch_idx, (batch_emb, batch_idx_tensor, batch_pos, lengths, msg_indices) in enumerate(dataloader):
        batch_emb = batch_emb.to(dev_enc)
        batch_idx_tensor = batch_idx_tensor.to(dev_enc)
        batch_pos = batch_pos.to(dev_enc)
        lengths_enc = lengths.to(dev_enc)
        max_len, batch_size = batch_emb.size(1), batch_emb.size(0)
        msg_idx = msg_indices[0].item()

        lengths_sym = lengths_enc.to(dev_sym)
        batch_idx_sym = batch_idx_tensor.to(dev_sym)

        # --- Фаза 1: SymbolDecoder ---
        for p in encoder.parameters(): p.requires_grad = False
        for p in symbol_dec.parameters(): p.requires_grad = True
        opt_sym.zero_grad()

        with torch.no_grad():
            pooled_detach_enc = encoder(batch_emb, lengths_enc, batch_pos)
        pooled_detach_sym = pooled_detach_enc.to(dev_sym)

        loss_sym_sum = torch.tensor(0.0, device=dev_sym)
        total_positions_sym = 0
        for pos in range(max_len):
            active = lengths_sym > pos
            if not active.any(): continue
            num_active = active.sum().item()
            target_idx = batch_idx_sym[active, pos]
            sym_out = symbol_dec(pooled_detach_sym[active],
                                 torch.full((num_active,), pos, device=dev_sym))
            # Новая потеря
            pos_loss = compute_symbol_loss_vector(sym_out['symbol_probs'], target_idx)
            loss_sym_sum = loss_sym_sum + pos_loss * num_active
            total_positions_sym += num_active

        if total_positions_sym > 0:
            weighted_sym_loss = SYM_LOSS_WEIGHT * (loss_sym_sum / total_positions_sym)
        else:
            weighted_sym_loss = torch.tensor(0.0, device=dev_sym)

        total_sym_loss = weighted_sym_loss

        if total_sym_loss.item() > 0:
            total_sym_loss.backward()
            sym_grads = {name: param.grad.clone().cpu() if param.grad is not None else None
                         for name, param in symbol_dec.named_parameters()}
        else:
            sym_grads = None
        opt_sym.zero_grad()

        # --- Фаза 2: Encoder ---
        for p in encoder.parameters(): p.requires_grad = True
        for p in symbol_dec.parameters(): p.requires_grad = False
        opt_enc.zero_grad()

        pooled = encoder(batch_emb, lengths_enc, batch_pos)

        total_pos_enc = 0
        batch_correct = 0
        sym_enc_sum = torch.tensor(0.0, device=dev_enc)
        total_sym_pos_enc = 0

        for pos in range(max_len):
            active = lengths_enc > pos
            if not active.any(): continue
            num_active = active.sum().item()
            target_idx = batch_idx_tensor[active, pos].to(dev_sym)
            pos_tensor = torch.full((num_active,), pos, device=dev_enc)
            pooled_act = pooled[active]

            pooled_act_sym = pooled_act.to(dev_sym)
            sym_out = symbol_dec(pooled_act_sym, pos_tensor.to(dev_sym))
            # Новая потеря
            pos_sym_loss = compute_symbol_loss_vector(sym_out['symbol_probs'], target_idx)
            weighted_pos_sym = SYM_LOSS_WEIGHT * pos_sym_loss
            sym_enc_sum = sym_enc_sum + weighted_pos_sym.to(dev_enc) * num_active
            total_sym_pos_enc += num_active

            # Accuracy (только символы)
            is_symbol = target_idx != NULL_INDEX
            if is_symbol.any():
                with torch.no_grad():
                    is_symbol_enc = is_symbol.to(dev_enc)
                    probs = sym_out['symbol_probs'][is_symbol]
                    pred_idx = probs.argmax(dim=-1)
                    correct = (pred_idx.to(dev_enc) == batch_idx_tensor[active][is_symbol_enc, pos].to(dev_enc)).sum().item()
                    batch_correct += correct
                    total_pos_enc += is_symbol.sum().item()

        sym_enc_avg = sym_enc_sum / total_sym_pos_enc if total_sym_pos_enc > 0 else torch.tensor(0.0, device=dev_enc)
        loss_enc_total = sym_enc_avg
        weighted_loss_enc = ENC_LOSS_WEIGHT * loss_enc_total

        if weighted_loss_enc.item() > 0:
            weighted_loss_enc.backward()
            enc_grads = {name: param.grad.clone().cpu() if param.grad is not None else None
                         for name, param in encoder.named_parameters()}
        else:
            enc_grads = None
        opt_enc.zero_grad()

        def apply_grads(model, grads_dict, optimizer, device):
            if grads_dict is None: return
            for name, param in model.named_parameters():
                if name in grads_dict and grads_dict[name] is not None:
                    param.grad = grads_dict[name].to(device)
            optimizer.step()
            optimizer.zero_grad()

        apply_grads(symbol_dec, sym_grads, opt_sym, dev_sym)
        apply_grads(encoder, enc_grads, opt_enc, dev_enc)

        sym_loss_val = total_sym_loss.item()
        pos_acc = batch_correct / total_pos_enc if total_pos_enc else 0
        char_acc, ex_acc = evaluate_batch(
            encoder, symbol_dec,
            batch_emb, batch_idx_tensor, batch_pos, lengths_enc,
            dev_enc, dev_sym
        )
        print(f"Batch {batch_idx+1}/{num_batches}  Example index {msg_idx}")
        print(f"SymL:{sym_loss_val:.4f}")
        print(f"EncL:{weighted_loss_enc:.4f}")
        print(f"PosAcc:{pos_acc:.4f} ({batch_correct}/{total_pos_enc}) | CharAcc:{char_acc:.4f} ExAcc:{ex_acc:.4f}")

        total_loss_sym += sym_loss_val
        total_loss_enc += weighted_loss_enc.item()
        total_correct += batch_correct
        total_positions += total_pos_enc

        if CLEAR_CACHE_EACH_BATCH:
            del batch_emb, batch_idx_tensor, batch_pos, lengths_enc, pooled_detach_enc, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    avg_loss_sym_epoch = total_loss_sym / num_batches
    avg_loss_enc_epoch = total_loss_enc / num_batches
    pos_accuracy = total_correct / total_positions if total_positions else 0.0
    losses = {'sym': avg_loss_sym_epoch, 'enc': avg_loss_enc_epoch}
    return losses, pos_accuracy


def train_model():
    opt_sym = optim.Adam(symbol_decoder.parameters(), lr=LEARNING_RATE)
    opt_enc = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    start_epoch = load_checkpoints_if_exist() + 1

    epoch = start_epoch
    while True:
        print(f"\n--- Epoch {epoch} ---")
        losses, pos_acc = train_epoch_sequential(
            encoder, symbol_decoder,
            train_loader, opt_sym, opt_enc,
            epoch, ENCODER_DEVICE, SYMBOL_DECODER_DEVICE
        )
        char_acc, ex_acc = evaluate_sequential(
            encoder, symbol_decoder,
            train_loader, ENCODER_DEVICE, SYMBOL_DECODER_DEVICE
        )
        print(f"Epoch {epoch:3d} TRAIN SUMMARY")
        print(f"SymL:{losses['sym']:.6f}")
        print(f"EncL:{losses['enc']:.6f}")
        print(f"PosAcc:{pos_acc:.4f} | CharAcc:{char_acc:.4f} ExAcc:{ex_acc:.4f}")

        if USE_VALIDATION and val_loader is not None and epoch % VAL_EVERY_EPOCHS == 0:
            val_metrics = validate(
                encoder, symbol_decoder,
                val_loader, ENCODER_DEVICE, SYMBOL_DECODER_DEVICE
            )
            val_char_acc, val_ex_acc = evaluate_sequential(
                encoder, symbol_decoder,
                val_loader, ENCODER_DEVICE, SYMBOL_DECODER_DEVICE
            )
            print(f"Epoch {epoch:3d} VAL")
            print(f"SymL:{val_metrics['loss_sym']:.6f}")
            print(f"EncL:{val_metrics['loss_enc']:.6f}")
            print(f"PosAcc:{val_metrics['pos_acc']:.4f} | CharAcc:{val_char_acc:.4f} ExAcc:{val_ex_acc:.4f}")

        # Генерация примера
        try:
            sample_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
            sample_batch = next(iter(sample_loader))
            batch_emb, batch_idx_tensor, batch_pos, lengths, msg_indices = sample_batch
            batch_emb = batch_emb.to(ENCODER_DEVICE)
            batch_idx_tensor = batch_idx_tensor.to(ENCODER_DEVICE)
            batch_pos = batch_pos.to(ENCODER_DEVICE)
            lengths = lengths.to(ENCODER_DEVICE)
            with torch.no_grad():
                pooled = encoder(batch_emb, lengths, batch_pos)

            true_len = lengths[0].item() - 1
            true_indices = batch_idx_tensor[0, :true_len].tolist()
            true_text = ''.join([IDX_TO_CHAR[idx] for idx in true_indices])

            pred_text = generate_text_until_stop(
                pooled[0:1], symbol_decoder, SYMBOL_DECODER_DEVICE
            )

            print(f"--- Sample (Example index {msg_indices[0].item()}) ---")
            print(f"True  (len={true_len}): {true_text}")
            print(f"Pred  (len={len(pred_text)}): {pred_text}")
            print("-" * 50)
            del batch_emb, batch_idx_tensor, batch_pos, lengths, pooled
        except Exception as e:
            print(f"Could not generate sample: {e}")

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch)

        epoch += 1


if __name__ == "__main__":
    train_model()