# chat_usage_mikri_models.py
"""
Модуль инференса для чата: загружает обученные модели и генерирует ответ на сообщение пользователя.
Каждый текст дополняется завершающим нулевым эмбеддингом, как при обучении.
После использования модели выгружаются из VRAM.
"""

import os
import json
import glob
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple

from modelsEncoder_text_txt_emb_to_txt_emb_and_latent import Encoder
from modelsDecoder_text_txt_emb_to_txt_emb_and_latent import SymbolDecoder
from mikri_models import MikriModel, TemporalMemoryModel
from config_training_mikri_models import (
    DEVICE_STR,
    INPUT_EMB_DIM,
    HIDDEN_DIM,
    DROPOUT,
    SYM_DECODER_NUM_LAYERS,
    SYM_DECODER_PRE_MLP_LAYERS,
    SYM_DECODER_MLP_MULTIPLIER,
    NUM_OCTAVES,
    HIDDEN_DIM_MIKRI,
    NUM_THOUGHT_BLOCKS,
    NUM_MEMORY_SLOTS,
    NUM_HEADS_MIKRI,
    DROPOUT_MIKRI,
    MODELS_DIR,
    MAX_SEQ_LEN,
    TEMP_MEM_MAX_SEQ_LEN,
    ENCODER_DEVICE_STR, SYMBOL_DECODER_DEVICE_STR,
    TEMPORAL_MEMORY_DEVICE_STR, MIKRI_MODEL_DEVICE_STR,
)
from vocab import CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE

CHAT_HISTORY_DIR = "chat_history"
GENERATION_MAX_LEN = 1024
NULL_INDEX = VOCAB_SIZE


class ChatModel:
    def __init__(self):
        self.device_encoder = torch.device(ENCODER_DEVICE_STR) if torch.cuda.is_available() else torch.device("cpu")
        self.device_symbol = torch.device(SYMBOL_DECODER_DEVICE_STR) if torch.cuda.is_available() else torch.device("cpu")
        self.device_temp_mem = torch.device(TEMPORAL_MEMORY_DEVICE_STR) if torch.cuda.is_available() else torch.device("cpu")
        self.device_mikri = torch.device(MIKRI_MODEL_DEVICE_STR) if torch.cuda.is_available() else torch.device("cpu")

        print(f"ChatModel devices: encoder={self.device_encoder}, symbol={self.device_symbol}, "
              f"temp_mem={self.device_temp_mem}, mikri={self.device_mikri}")

        embeddings_path = "symbol_txt_emb_to_txt_emb/symbol_embeddings.pt"
        self.symbol_embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)
        vocab_size, emb_dim = self.symbol_embeddings.shape
        assert emb_dim == INPUT_EMB_DIM, f"Embedding dim mismatch: {emb_dim} vs {INPUT_EMB_DIM}"
        self.char_to_emb = {ch: self.symbol_embeddings[idx] for ch, idx in CHAR_TO_IDX.items()}
        print(f"Loaded {len(self.char_to_emb)} symbol embeddings (kept on CPU)")

        self.encoder = None
        self.symbol_decoder = None
        self.temporal_memory = None
        self.mikri_model = None
        self.is_loaded = False

        self._list_available_checkpoints()
        print("ChatModel created. Models will be loaded on first request.")

    def _init_models(self):
        self.encoder = Encoder(
            input_emb_dim=INPUT_EMB_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            max_seq_len=MAX_SEQ_LEN,
        ).to(self.device_encoder)

        self.symbol_decoder = SymbolDecoder(
            hidden_dim=HIDDEN_DIM,
            symbol_emb_dim=INPUT_EMB_DIM,
            num_layers=SYM_DECODER_NUM_LAYERS,
            pre_mlp_layers=SYM_DECODER_PRE_MLP_LAYERS,
            mlp_multiplier=SYM_DECODER_MLP_MULTIPLIER,
            dropout=DROPOUT,
            max_len=MAX_SEQ_LEN,
            num_octaves=NUM_OCTAVES,
            vocab_size=VOCAB_SIZE               # <-- добавлен параметр
        ).to(self.device_symbol)

        self.temporal_memory = TemporalMemoryModel(
            hidden_dim=HIDDEN_DIM,
            num_layers=4,
            num_heads=8,
            dropout=DROPOUT_MIKRI,
            max_seq_len=TEMP_MEM_MAX_SEQ_LEN
        ).to(self.device_temp_mem)

        self.mikri_model = MikriModel(
            hidden_dim=HIDDEN_DIM_MIKRI,
            num_thought_blocks=NUM_THOUGHT_BLOCKS,
            num_memory_slots=NUM_MEMORY_SLOTS,
            num_heads=NUM_HEADS_MIKRI,
            dropout=DROPOUT_MIKRI,
        ).to(self.device_mikri)

    def _list_available_checkpoints(self):
        pattern = os.path.join(MODELS_DIR, "*_stage*_epoch*.pth")
        files = glob.glob(pattern)
        if not files:
            print("No checkpoint files found in", MODELS_DIR)
        else:
            print(f"Found {len(files)} checkpoint file(s) in {MODELS_DIR}:")
            for f in files:
                print(f"  - {os.path.basename(f)}")

    def _find_latest_checkpoint(self, model_name: str, stage: int) -> Optional[str]:
        import re
        pattern = f"{model_name}_stage{stage}_epoch*.pth"
        files = glob.glob(os.path.join(MODELS_DIR, pattern))
        if not files:
            return None
        def extract_epoch(fname):
            match = re.search(r'epoch(\d+)', fname)
            return int(match.group(1)) if match else -1
        return max(files, key=extract_epoch)

    def _load_weights(self):
        try:
            encoder_path = self._find_latest_checkpoint("encoder", 1)
            symbol_path = self._find_latest_checkpoint("symbol_decoder", 1)
            temp_mem_path = self._find_latest_checkpoint("temporal_memory", 2)
            mikri_path = self._find_latest_checkpoint("mikri_model", 2)

            if not all([encoder_path, symbol_path, temp_mem_path, mikri_path]):
                raise FileNotFoundError("Some model checkpoints missing")

            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device_encoder, weights_only=False))
            self.symbol_decoder.load_state_dict(torch.load(symbol_path, map_location=self.device_symbol, weights_only=False))
            self.temporal_memory.load_state_dict(torch.load(temp_mem_path, map_location=self.device_temp_mem, weights_only=False))
            self.mikri_model.load_state_dict(torch.load(mikri_path, map_location=self.device_mikri, weights_only=False))

            print("All model weights loaded successfully.")
            self._set_eval_mode()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Weights loading failed: {e}")
            self.is_loaded = False
            return False

    def _set_eval_mode(self):
        self.encoder.eval()
        self.symbol_decoder.eval()
        self.temporal_memory.eval()
        self.mikri_model.eval()

    def ensure_loaded(self):
        if not self.is_loaded:
            if self.encoder is None:
                self._init_models()
            self._load_weights()

    def unload(self):
        if self.encoder is not None:
            del self.encoder
        if self.symbol_decoder is not None:
            del self.symbol_decoder
        if self.temporal_memory is not None:
            del self.temporal_memory
        if self.mikri_model is not None:
            del self.mikri_model

        self.encoder = None
        self.symbol_decoder = None
        self.temporal_memory = None
        self.mikri_model = None
        self.is_loaded = False

        self.symbol_embeddings = self.symbol_embeddings.cpu()
        self.char_to_emb = {ch: self.symbol_embeddings[idx] for ch, idx in CHAR_TO_IDX.items()}

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Models and symbol embeddings unloaded from GPU memory.")

    @torch.no_grad()
    def _is_stop_embedding(self, pred_probs: torch.Tensor) -> bool:
        """True, если предсказан NULL_INDEX (токен конца)."""
        return bool((pred_probs.argmax(dim=-1) == NULL_INDEX).item())

    def text_to_input_tensor(self, text: str):
        embs = []
        for ch in text:
            if ch in self.char_to_emb:
                embs.append(self.char_to_emb[ch])
        if not embs:
            emb_seq = torch.zeros(1, 1, INPUT_EMB_DIM).to(self.device_encoder)
            positions = torch.tensor([[1]], dtype=torch.long).to(self.device_encoder)
            length = torch.tensor([1], dtype=torch.long).to(self.device_encoder)
            return emb_seq, positions, length

        emb_seq = torch.stack(embs, dim=0).unsqueeze(0).to(self.device_encoder)
        seq_len = emb_seq.size(1)
        null_token = torch.zeros(1, 1, INPUT_EMB_DIM).to(self.device_encoder)
        emb_seq = torch.cat([emb_seq, null_token], dim=1)
        positions = torch.arange(1, seq_len + 2, dtype=torch.long).unsqueeze(0).to(self.device_encoder)
        length = torch.tensor([seq_len + 1], dtype=torch.long).to(self.device_encoder)
        return emb_seq, positions, length

    def encode_text(self, text: str) -> torch.Tensor:
        emb_seq, positions, length = self.text_to_input_tensor(text)
        if length.item() == 0:
            return torch.zeros(1, HIDDEN_DIM, device=self.device_encoder)
        return self.encoder(emb_seq, length, positions)

    def load_all_messages(self) -> List[dict]:
        messages = []
        if not os.path.exists(CHAT_HISTORY_DIR):
            return messages
        for filename in os.listdir(CHAT_HISTORY_DIR):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(CHAT_HISTORY_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    msg = json.load(f)
                    text = msg.get('edited_content') or msg.get('content')
                    messages.append({
                        'role': msg['role'],
                        'content': text,
                        'index': msg.get('index'),
                        'timestamp': msg.get('timestamp')
                    })
            except Exception:
                continue
        messages.sort(key=lambda x: x.get('index', 0))
        return messages

    @torch.no_grad()
    def generate_response(self, user_text: str) -> str:
        self.ensure_loaded()
        if not self.is_loaded:
            return "Модели ещё не обучены. Нажмите кнопку 'Обучить всё' или 'Stage 1'."

        all_messages = self.load_all_messages()
        memory_texts = []
        for msg in all_messages:
            if msg['role'] in ('user', 'bot'):
                memory_texts.append(msg['content'])
        query_text = user_text

        pooled_history = []
        for text in memory_texts:
            p = self.encode_text(text)
            if p.numel() > 0:
                pooled_history.append(p.squeeze(0))

        if pooled_history:
            pooled_seq = torch.stack(pooled_history, dim=0).unsqueeze(0)
        else:
            pooled_seq = torch.zeros(1, 1, HIDDEN_DIM, device=self.device_encoder)

        query_pooled = self.encode_text(query_text)

        pooled_seq_temp = pooled_seq.to(self.device_temp_mem)
        query_pooled_temp = query_pooled.to(self.device_temp_mem)
        retrieved_pooled = self.temporal_memory(pooled_seq_temp, query_pooled_temp)

        query_mikri = query_pooled.to(self.device_mikri)
        retrieved_mikri = retrieved_pooled.to(self.device_mikri)
        zero_mikri = torch.zeros_like(query_mikri)
        new_pooled = self.mikri_model(query_mikri, retrieved_mikri, zero_mikri)

        sym_emb = self.symbol_embeddings.to(self.device_symbol)
        generated_chars = []
        for pos in range(GENERATION_MAX_LEN):
            pos_tensor = torch.tensor([pos], device=self.device_symbol)
            pooled_sym = new_pooled.to(self.device_symbol)
            sym_out = self.symbol_decoder(pooled_sym, pos_tensor)
            pred_probs = sym_out['symbol_probs'][0]  # (V+1,)

            if self._is_stop_embedding(pred_probs):
                break

            pred_char_idx = pred_probs.argmax().item()
            pred_char = IDX_TO_CHAR[pred_char_idx]
            generated_chars.append(pred_char)

        return "".join(generated_chars)

    @torch.no_grad()
    def generate_response_from_history(self) -> str:
        self.ensure_loaded()
        if not self.is_loaded:
            return "Модели ещё не обучены. Нажмите кнопку 'Обучить всё' или 'Stage 1'."

        all_msgs = self.load_all_messages()
        last_user_text = None
        for msg in reversed(all_msgs):
            if msg['role'] == 'user':
                last_user_text = msg['content']
                break
        if last_user_text is None:
            return "Нет сообщений пользователя."
        return self.generate_response(last_user_text)