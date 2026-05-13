# training_mikri_models_Stage_1.py
"""
Stage 1: Encoder, SymbolDecoder.
Используется векторная потеря (сравнение вероятностей с one‑hot как единое целое).
Нулевой токен добавляется в конец каждой последовательности.
Градиенты фаз сохраняются, применяются последовательно.
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from config_training_mikri_models import *
from vocab import CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE

NULL_INDEX = VOCAB_SIZE


# ========== Вспомогательные функции ==========
def compute_symbol_loss_vector(pred_probs, target_idx):
    """
    Сравнивает предсказанный вектор вероятностей с целевым one‑hot вектором
    как единое целое (через скалярное произведение).
    При ошибке argmax потеря удваивается.
    """
    V_plus_1 = pred_probs.size(-1)
    target_onehot = F.one_hot(target_idx, num_classes=V_plus_1).float()
    pos_similarity = (pred_probs * target_onehot).sum(dim=1)
    neg_similarities = pred_probs * (1.0 - target_onehot)
    neg_similarity = neg_similarities.sum(dim=1) / max(1, V_plus_1 - 1)
    loss1 = 1.0 - pos_similarity
    loss2 = neg_similarity
    loss = loss1 + loss2
    wrong_mask = (pred_probs.argmax(dim=-1) != target_idx).float()
    loss = loss * (1.0 + wrong_mask)   # x2 при ошибке
    return loss.mean()


# --- Dataset и collate (с нулевым токеном) ---
class TextEmbDataset(Dataset):
    def __init__(self, texts, indices, char_to_emb, input_emb_dim):
        self.examples = []
        for idx, t in zip(indices, texts):
            embs, indices_list = [], []
            for ch in t:
                if ch in char_to_emb:
                    embs.append(char_to_emb[ch])
                    indices_list.append(CHAR_TO_IDX[ch])
            if embs:
                emb_seq = torch.stack(embs, dim=0)
                idx_seq = torch.tensor(indices_list, dtype=torch.long)
                self.examples.append((emb_seq, idx_seq, len(emb_seq), idx))

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def collate_fn(batch):
    emb_seqs, idx_seqs, lengths, msg_indices = zip(*batch)
    new_lengths = [L + 1 for L in lengths]
    max_len = max(new_lengths)
    batch_size = len(batch)

    batch_emb = torch.zeros(batch_size, max_len, INPUT_EMB_DIM)
    batch_idx = torch.full((batch_size, max_len), -1, dtype=torch.long)
    batch_pos = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, (emb, idx, L) in enumerate(zip(emb_seqs, idx_seqs, lengths)):
        batch_emb[i, :L] = emb
        batch_idx[i, :L] = idx
        batch_pos[i, :L] = torch.arange(1, L + 1)
        # Нулевой токен
        batch_emb[i, L] = torch.zeros(INPUT_EMB_DIM)
        batch_idx[i, L] = NULL_INDEX
        batch_pos[i, L] = L + 1

    return (batch_emb, batch_idx, batch_pos,
            torch.tensor(new_lengths, dtype=torch.long),
            torch.tensor(msg_indices, dtype=torch.long))


# --- Метрики и генерация ---
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


def generate_text_fixed_len(pooled, symbol_dec, dev_sym, length):
    """Генерирует length символов (без учёта нулевого токена)."""
    generated_indices = []
    for pos in range(length):
        pos_tensor = torch.tensor([pos], device=dev_sym)
        pooled_sym = pooled.to(dev_sym)
        sym_out = symbol_dec(pooled_sym, pos_tensor)
        probs = sym_out['symbol_probs'][0]
        if probs.argmax().item() == NULL_INDEX:
            break
        pred_idx = probs.argmax().item()
        generated_indices.append(pred_idx)
    return ''.join([IDX_TO_CHAR[idx] for idx in generated_indices])


# --- Основной цикл обучения Stage 1 ---
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

        batch_idx_sym = batch_idx_tensor.to(dev_sym)
        lengths_sym = lengths_enc.to(dev_sym)

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
        print(f"Ep {epoch:3d} Msg {msg_idx:4d} Bt {batch_idx+1:3d}/{num_batches} | "
              f"SymL:{sym_loss_val:.4f} EncL:{weighted_loss_enc:.4f} "
              f"PosAcc:{pos_acc:.4f} ({batch_correct}/{total_pos_enc}) | "
              f"CharAcc:{char_acc:.4f} ExAcc:{ex_acc:.4f}")

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


def train_stage1(encoder, symbol_decoder,
                 dataloader, symbol_embeddings,
                 dev_enc, dev_sym,
                 save_checkpoint_callback):
    opt_sym = optim.Adam(symbol_decoder.parameters(), lr=LEARNING_RATE)
    opt_enc = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    epoch = 1
    while True:
        print(f"\n--- Stage 1 Epoch {epoch} ---")
        losses, pos_acc = train_epoch_sequential(
            encoder, symbol_decoder,
            dataloader, opt_sym, opt_enc,
            epoch, dev_enc, dev_sym
        )
        char_acc, ex_acc = evaluate_sequential(
            encoder, symbol_decoder,
            dataloader, dev_enc, dev_sym
        )
        print(f"Stage1 Ep {epoch:3d} SUMMARY | SymL:{losses['sym']:.6f} "
              f"EncL:{losses['enc']:.6f} PosAcc:{pos_acc:.4f} "
              f"CharAcc:{char_acc:.4f} ExAcc:{ex_acc:.4f}")

        try:
            sample_loader = DataLoader(dataloader.dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
            sample_batch = next(iter(sample_loader))
            batch_emb, batch_idx_tensor, batch_pos, lengths, msg_indices = sample_batch
            batch_emb = batch_emb.to(dev_enc); batch_idx_tensor = batch_idx_tensor.to(dev_enc)
            batch_pos = batch_pos.to(dev_enc); lengths = lengths.to(dev_enc)
            with torch.no_grad():
                pooled = encoder(batch_emb, lengths, batch_pos)
            i = 0
            true_len = lengths[i].item() - 1
            true_indices = batch_idx_tensor[i, :true_len].tolist()
            true_text = ''.join([IDX_TO_CHAR[idx] for idx in true_indices])
            pred_text = generate_text_fixed_len(
                pooled[i:i+1], symbol_decoder, dev_sym, true_len
            )
            print(f"--- Sample (msg index {msg_indices[i].item()}) ---")
            print(f"True text: {true_text}")
            print(f"Pred text: {pred_text}")
            print("-" * 50)
            del batch_emb, batch_idx_tensor, batch_pos, lengths, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Could not generate sample: {e}")

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint_callback(epoch)

        epoch += 1