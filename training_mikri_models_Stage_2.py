# training_mikri_models_Stage_2.py
"""
Stage 2: TemporalMemoryModel + MikriModel.
Потери: символьная векторная + MSE для pooled.
Нулевой токен добавлен во все последовательности.
Градиенты фаз сохраняются, применяются последовательно.
"""

import os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import gc
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
    loss = loss * (1.0 + wrong_mask)
    return loss.mean()


def is_stop_probs(pred_probs: torch.Tensor) -> bool:
    return bool((pred_probs.argmax(dim=-1) == NULL_INDEX).item())


def generate_text_from_pooled(pooled, symbol_dec, dev_sym, max_len=1024):
    generated_indices = []
    for pos in range(max_len):
        pos_tensor = torch.tensor([pos], device=dev_sym)
        pooled_sym = pooled.to(dev_sym)
        sym_out = symbol_dec(pooled_sym, pos_tensor)
        pred_probs = sym_out['symbol_probs'][0]
        if is_stop_probs(pred_probs):
            break
        pred_idx = pred_probs.argmax().item()
        generated_indices.append(pred_idx)
    return ''.join([IDX_TO_CHAR[idx] for idx in generated_indices])


# --- Dataset и collate (без изменений) ---
class UserBotPairDataset(Dataset):
    def __init__(self, pairs, all_messages, char_to_emb):
        self.examples = []
        for i in range(len(all_messages)-1):
            if all_messages[i]['role'] == 'user' and all_messages[i+1]['role'] == 'bot':
                user_msg = all_messages[i]; bot_msg = all_messages[i+1]; history_msgs = all_messages[:i]
                user_emb, user_idx = _text_to_embs(user_msg['text'], char_to_emb)
                bot_emb, bot_idx = _text_to_embs(bot_msg['text'], char_to_emb)
                if len(user_emb)==0 or len(bot_emb)==0: continue
                history_embs = []
                history_lens = []
                for msg in history_msgs:
                    emb, _ = _text_to_embs(msg['text'], char_to_emb)
                    if len(emb)>0:
                        history_embs.append(emb)
                        history_lens.append(len(emb))
                self.examples.append({
                    'pair_index': i,
                    'user_emb': user_emb, 'user_idx': user_idx, 'user_len': len(user_emb),
                    'bot_emb': bot_emb, 'bot_idx': bot_idx, 'bot_len': len(bot_emb),
                    'history_embs': history_embs,
                    'history_lens': history_lens
                })
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def _text_to_embs(text, char_to_emb):
    embs, indices = [], []
    for ch in text:
        if ch in char_to_emb:
            embs.append(char_to_emb[ch]); indices.append(CHAR_TO_IDX[ch])
    if not embs: return torch.empty(0, INPUT_EMB_DIM), torch.empty(0, dtype=torch.long)
    return torch.stack(embs, dim=0), torch.tensor(indices, dtype=torch.long)


def pair_collate_fn(batch):
    batch_size = len(batch)
    # User
    user_max_len = max(ex['user_len'] for ex in batch) + 1
    user_emb = torch.zeros(batch_size, user_max_len, INPUT_EMB_DIM)
    user_idx = torch.full((batch_size, user_max_len), -1, dtype=torch.long)
    user_pos = torch.zeros(batch_size, user_max_len, dtype=torch.long)
    user_len = torch.zeros(batch_size, dtype=torch.long)
    # Bot
    bot_max_len = max(ex['bot_len'] for ex in batch) + 1
    bot_emb = torch.zeros(batch_size, bot_max_len, INPUT_EMB_DIM)
    bot_idx = torch.full((batch_size, bot_max_len), -1, dtype=torch.long)
    bot_pos = torch.zeros(batch_size, bot_max_len, dtype=torch.long)
    bot_len = torch.zeros(batch_size, dtype=torch.long)
    # History
    max_hist = max(len(ex['history_embs']) for ex in batch)
    max_msg_len = max((max(ex['history_lens']) if ex['history_lens'] else 0) for ex in batch) + 1
    hist_embs = torch.zeros(batch_size, max_hist, max_msg_len, INPUT_EMB_DIM)
    hist_idx = torch.full((batch_size, max_hist, max_msg_len), -1, dtype=torch.long)
    hist_pos = torch.zeros(batch_size, max_hist, max_msg_len, dtype=torch.long)
    hist_len = torch.zeros(batch_size, max_hist, dtype=torch.long)
    hist_mask = torch.zeros(batch_size, max_hist, dtype=torch.bool)
    pair_indices = torch.zeros(batch_size, dtype=torch.long)

    for i, ex in enumerate(batch):
        # User
        L = ex['user_len']
        user_emb[i, :L] = ex['user_emb']
        user_idx[i, :L] = ex['user_idx']
        user_pos[i, :L] = torch.arange(1, L+1)
        user_emb[i, L] = 0.0
        user_idx[i, L] = NULL_INDEX
        user_pos[i, L] = L + 1
        user_len[i] = L + 1

        # Bot
        Lb = ex['bot_len']
        bot_emb[i, :Lb] = ex['bot_emb']
        bot_idx[i, :Lb] = ex['bot_idx']
        bot_pos[i, :Lb] = torch.arange(1, Lb+1)
        bot_emb[i, Lb] = 0.0
        bot_idx[i, Lb] = NULL_INDEX
        bot_pos[i, Lb] = Lb + 1
        bot_len[i] = Lb + 1

        # History
        for j, (emb, length) in enumerate(zip(ex['history_embs'], ex['history_lens'])):
            Lh = length
            hist_embs[i, j, :Lh] = emb
            hist_embs[i, j, Lh] = 0.0
            hist_idx[i, j, :Lh] = torch.tensor([CHAR_TO_IDX.get(ch, 0) for ch in str(emb)], dtype=torch.long)
            hist_pos[i, j, :Lh+1] = torch.arange(1, Lh+2)
            hist_len[i, j] = Lh + 1
            hist_mask[i, j] = True
        pair_indices[i] = ex['pair_index']

    return (user_emb, user_idx, user_pos, user_len,
            bot_emb, bot_idx, bot_pos, bot_len,
            hist_embs, hist_idx, hist_pos, hist_len, hist_mask,
            pair_indices)


# --- Главная функция Stage 2 ---
def train_stage2(encoder, symbol_dec, temp_mem, mikri,
                 pair_loader, symbol_embeddings,
                 dev_enc, dev_sym, dev_temp, dev_mikri,
                 save_checkpoint_callback):
    for m in [encoder, symbol_dec]:
        m.eval()
        for p in m.parameters(): p.requires_grad = False

    opt_temp = optim.Adam(temp_mem.parameters(), lr=LEARNING_RATE_MIKRI)
    opt_mikri = optim.Adam(mikri.parameters(), lr=LEARNING_RATE_MIKRI)

    epoch = 1
    while True:
        total_loss_temp = 0.0; total_loss_mikri = 0.0; total_correct = 0; total_positions = 0
        num_batches = len(pair_loader); batches_with_hist = 0
        for batch_idx, (user_emb, user_idx, user_pos, user_len,
                        bot_emb, bot_idx, bot_pos, bot_len,
                        hist_embs, hist_idx, hist_pos, hist_len, hist_mask,
                        pair_indices) in enumerate(pair_loader):
            user_emb = user_emb.to(dev_enc); user_len = user_len.to(dev_enc); user_pos = user_pos.to(dev_enc)
            bot_emb = bot_emb.to(dev_enc); bot_idx = bot_idx.to(dev_enc); bot_pos = bot_pos.to(dev_enc); bot_len = bot_len.to(dev_enc)
            hist_embs = hist_embs.to(dev_enc); hist_mask = hist_mask.to(dev_enc); hist_len = hist_len.to(dev_enc)
            B = user_emb.size(0)
            pair_idx = pair_indices[0].item()

            with torch.no_grad():
                query_pooled = encoder(user_emb, user_len, user_pos)

            # Построение padded_pooled_seq для истории
            pooled_seq_list = []; max_hist_len = 0
            for b in range(B):
                seq = []
                for m in range(hist_embs.size(1)):
                    if hist_mask[b, m]:
                        msg_emb = hist_embs[b, m]
                        msglen = hist_len[b, m].item()
                        msg_emb = msg_emb[:msglen].unsqueeze(0)
                        msg_pos = torch.arange(1, msglen+1, device=dev_enc).unsqueeze(0)
                        with torch.no_grad():
                            p = encoder(msg_emb, torch.tensor([msglen], device=dev_enc), msg_pos).squeeze(0)
                        seq.append(p)
                if seq: pooled_seq = torch.stack(seq, dim=0)
                else: pooled_seq = torch.zeros(1, HIDDEN_DIM, device=dev_enc)
                pooled_seq_list.append(pooled_seq)
                max_hist_len = max(max_hist_len, pooled_seq.size(0))
            padded_pooled_seq = torch.zeros(B, max_hist_len, HIDDEN_DIM, device=dev_enc)
            hist_seq_mask = torch.zeros(B, max_hist_len, dtype=torch.bool, device=dev_enc)
            for b in range(B):
                L = pooled_seq_list[b].size(0)
                padded_pooled_seq[b, :L] = pooled_seq_list[b]
                hist_seq_mask[b, :L] = True

            with torch.no_grad():
                target_pooled = encoder(bot_emb, bot_len, bot_pos)
            no_hist = (hist_len.sum(dim=1) == 0); has_hist = ~no_hist

            # --- Фаза 1: TemporalMemory ---
            for p in temp_mem.parameters(): p.requires_grad = True
            for p in mikri.parameters(): p.requires_grad = False
            opt_temp.zero_grad()

            if has_hist.any():
                batches_with_hist += 1
                sel = has_hist
                padded_seq_sel = padded_pooled_seq[sel].to(dev_temp)
                mask_sel = hist_seq_mask[sel].to(dev_temp)
                query_sel = query_pooled[sel].to(dev_temp)
                retr = temp_mem(padded_seq_sel, query_sel, padding_mask=~mask_sel)

                query_mikri = query_pooled[sel].to(dev_mikri)
                retr_mikri = retr.to(dev_mikri)
                zero_mikri = torch.zeros_like(query_mikri)
                new_pooled_temp = mikri(query_mikri, retr_mikri, zero_mikri).to(dev_temp)

                loss_pooled = F.mse_loss(new_pooled_temp, target_pooled[sel].to(dev_temp))

                # Символьная векторная потеря
                total_sym_pos = 0; sym_temp_sum = 0.0
                max_bot = bot_idx.size(1)
                for pos in range(max_bot):
                    active = (bot_len[sel] > pos)
                    if not active.any(): continue
                    n_act = active.sum().item()
                    tgt_idx = bot_idx[sel][active, pos].to(dev_sym)
                    sym_out = symbol_dec(new_pooled_temp[active].to(dev_sym),
                                         torch.full((n_act,), pos, device=dev_sym))
                    pos_loss = compute_symbol_loss_vector(sym_out['symbol_probs'], tgt_idx)
                    sym_temp_sum += pos_loss.item() * n_act
                    total_sym_pos += n_act

                weighted_sym_temp = STAGE2_SYM_LOSS_WEIGHT * (sym_temp_sum / total_sym_pos) if total_sym_pos > 0 else 0.0
                total_temp_loss = STAGE2_POOLED_LOSS_WEIGHT * loss_pooled + torch.tensor(weighted_sym_temp, device=dev_temp)

                if total_temp_loss.item() > 0:
                    total_temp_loss.backward()
                    temp_grads = {name: p.grad.clone().cpu() if p.grad is not None else None
                                  for name, p in temp_mem.named_parameters()}
                else:
                    temp_grads = None
                opt_temp.zero_grad()
                retr_det = retr.detach()
                total_loss_temp += total_temp_loss.item()
            else:
                retr_det = torch.zeros_like(query_pooled)
                temp_grads = None

            # --- Фаза 2: MikriModel ---
            for p in temp_mem.parameters(): p.requires_grad = False
            for p in mikri.parameters(): p.requires_grad = True
            opt_mikri.zero_grad()

            query_mikri = query_pooled.to(dev_mikri)
            retr_det_mikri = retr_det.to(dev_mikri)
            zero_mikri = torch.zeros_like(query_mikri)
            new_pooled_mikri = mikri(query_mikri, retr_det_mikri, zero_mikri)

            loss_pooled = F.mse_loss(new_pooled_mikri, target_pooled.to(dev_mikri))

            total_sym_pos_m = 0; sym_mikri_sum = 0.0
            correct = 0
            max_bot = bot_idx.size(1)
            for pos in range(max_bot):
                active = (bot_len > pos)
                if not active.any(): continue
                n_act = active.sum().item()
                tgt_idx = bot_idx[active, pos].to(dev_sym)
                sym_out = symbol_dec(new_pooled_mikri[active].to(dev_sym),
                                     torch.full((n_act,), pos, device=dev_sym))
                pos_sym_loss = compute_symbol_loss_vector(sym_out['symbol_probs'], tgt_idx)
                sym_mikri_sum += pos_sym_loss.item() * n_act
                total_sym_pos_m += n_act

                # Accuracy (только символы)
                is_symbol = tgt_idx != NULL_INDEX
                if is_symbol.any():
                    probs = sym_out['symbol_probs'][is_symbol]
                    pred_idx = probs.argmax(dim=-1)
                    correct += (pred_idx.to(dev_mikri) == bot_idx[active][is_symbol, pos].to(dev_mikri)).sum().item()

            weighted_sym_mikri = STAGE2_SYM_LOSS_WEIGHT * (sym_mikri_sum / total_sym_pos_m) if total_sym_pos_m > 0 else 0.0
            total_mikri_loss = STAGE2_POOLED_LOSS_WEIGHT * loss_pooled + torch.tensor(weighted_sym_mikri, device=dev_mikri)

            if total_mikri_loss.item() > 0:
                total_mikri_loss.backward()
                mikri_grads = {name: p.grad.clone().cpu() if p.grad is not None else None
                               for name, p in mikri.named_parameters()}
            else:
                mikri_grads = None
            opt_mikri.zero_grad()

            def apply_grads(model, grads_dict, optimizer, device):
                if grads_dict is None: return
                for name, param in model.named_parameters():
                    if name in grads_dict and grads_dict[name] is not None:
                        param.grad = grads_dict[name].to(device)
                optimizer.step()
                optimizer.zero_grad()

            apply_grads(temp_mem, temp_grads, opt_temp, dev_temp)
            apply_grads(mikri, mikri_grads, opt_mikri, dev_mikri)

            pos_acc = correct / (max_bot * B) if max_bot > 0 else 0
            print(f"Stage2 Ep {epoch} Pair {pair_idx:4d} Bt {batch_idx+1}/{num_batches} | "
                  f"TempL:{total_temp_loss.item() if has_hist.any() else 0:.4f} MikriL:{total_mikri_loss.item():.4f} "
                  f"SymL:{weighted_sym_mikri:.4f} PosAcc:{pos_acc:.4f}")

            if CLEAR_CACHE_EACH_BATCH:
                del user_emb, user_pos, user_len, bot_emb, bot_pos, bot_len
                del hist_embs, hist_mask, hist_len, query_pooled
                del padded_pooled_seq, hist_seq_mask, target_pooled
                del new_pooled_mikri, retr_det, loss_pooled, sym_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        avg_temp = total_loss_temp / batches_with_hist if batches_with_hist else 0.0
        avg_mikri = total_loss_mikri / num_batches
        overall_acc = total_correct / total_positions if total_positions else 0.0
        print(f"Stage2 Ep {epoch} SUMMARY | AvgTempL:{avg_temp:.6f} AvgMikriL:{avg_mikri:.6f} PosAcc:{overall_acc:.4f}")

        # Генерация примера
        try:
            sample_loader = DataLoader(pair_loader.dataset, batch_size=1, shuffle=True, collate_fn=pair_collate_fn)
            sample_batch = next(iter(sample_loader))
            (user_emb_s, _, user_pos_s, user_len_s,
             bot_emb_s, bot_idx_s, bot_pos_s, bot_len_s,
             hist_embs_s, hist_idx_s, hist_pos_s, hist_len_s, hist_mask_s, pair_indices_s) = sample_batch
            user_emb_s = user_emb_s.to(dev_enc); user_len_s = user_len_s.to(dev_enc); user_pos_s = user_pos_s.to(dev_enc)
            bot_emb_s = bot_emb_s.to(dev_enc); bot_len_s = bot_len_s.to(dev_enc); bot_pos_s = bot_pos_s.to(dev_enc)
            hist_embs_s = hist_embs_s.to(dev_enc); hist_len_s = hist_len_s.to(dev_enc); hist_mask_s = hist_mask_s.to(dev_enc)

            with torch.no_grad():
                query_pooled_s = encoder(user_emb_s, user_len_s, user_pos_s)
                pooled_seq_list_s = []
                for m in range(hist_embs_s.size(1)):
                    if hist_mask_s[0, m]:
                        msg_emb = hist_embs_s[0, m]
                        msglen = hist_len_s[0, m].item()
                        msg_emb = msg_emb[:msglen].unsqueeze(0)
                        msg_pos = torch.arange(1, msglen+1, device=dev_enc).unsqueeze(0)
                        p = encoder(msg_emb, torch.tensor([msglen], device=dev_enc), msg_pos)
                        pooled_seq_list_s.append(p.squeeze(0))
                if pooled_seq_list_s:
                    pooled_seq_s = torch.stack(pooled_seq_list_s, dim=0).unsqueeze(0)
                else:
                    pooled_seq_s = torch.zeros(1, 1, HIDDEN_DIM, device=dev_enc)

                retrieved_s = temp_mem(pooled_seq_s.to(dev_temp), query_pooled_s.to(dev_temp))
                zero_s = torch.zeros_like(query_pooled_s)
                new_pooled_s = mikri(query_pooled_s.to(dev_mikri), retrieved_s.to(dev_mikri), zero_s.to(dev_mikri))

                true_bot_len = bot_len_s[0].item() - 1
                true_bot_indices = bot_idx_s[0, :true_bot_len].tolist()
                true_bot_text = ''.join([IDX_TO_CHAR[idx] for idx in true_bot_indices])

                pred_temp_text = generate_text_from_pooled(retrieved_s, symbol_dec, dev_sym)
                pred_mikri_text = generate_text_from_pooled(new_pooled_s, symbol_dec, dev_sym)

                print(f"--- Sample pair index {pair_indices_s[0].item()} ---")
                print(f"True bot answer: {true_bot_text}")
                print(f"Pred (TemporalMemory): {pred_temp_text}")
                print(f"Pred (MikriModel):     {pred_mikri_text}")
                print("-" * 60)

                del user_emb_s, user_pos_s, user_len_s, bot_emb_s, bot_idx_s, bot_pos_s, bot_len_s
                del hist_embs_s, hist_len_s, hist_mask_s, query_pooled_s
                del pooled_seq_s, retrieved_s, new_pooled_s
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Could not generate sample in Stage 2: {e}")

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint_callback(epoch)

        epoch += 1