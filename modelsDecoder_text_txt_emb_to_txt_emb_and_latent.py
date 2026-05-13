# modelsDecoder_text_txt_emb_to_txt_emb_and_latent.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SymbolDecoder(nn.Module):
    """
    Предсказывает вероятностное распределение над символами (включая токен «конец»)
    по глобальному `pooled` и позиции.
    Использует разделяемый MLP-блок (weight tying) для уменьшения числа параметров.
    """
    def __init__(self,
                 hidden_dim: int = 1024,
                 symbol_emb_dim: int = 16,          # больше не используется, оставлено для совместимости
                 num_layers: int = 8,
                 pre_mlp_layers: int = 2,
                 mlp_multiplier: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 1024,
                 num_octaves: int = 4,
                 use_checkpoint: bool = False,
                 vocab_size: int = None):            # !!! новый обязательный параметр
        super().__init__()
        if vocab_size is None:
            raise ValueError("vocab_size must be provided")
        self.vocab_size = vocab_size
        self.num_outputs = vocab_size + 1            # +1 для токена конца (NULL_INDEX = vocab_size)

        self.hidden_dim = hidden_dim
        self.num_octaves = num_octaves
        self.symbol_emb_dim = symbol_emb_dim
        self.use_checkpoint = use_checkpoint

        # Предварительные полносвязные слои
        pre_layers = []
        in_dim = hidden_dim
        for i in range(pre_mlp_layers):
            pre_layers.append(nn.Linear(in_dim, hidden_dim))
            if i < pre_mlp_layers - 1:
                pre_layers.append(nn.GELU())
                pre_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.pre_mlp = nn.Sequential(*pre_layers) if pre_layers else nn.Identity()

        # Позиционное кодирование (мультирезолюционное)
        self.pos_proj = nn.Linear(hidden_dim * num_octaves, hidden_dim)

        # FiLM-генератор (параметры модуляции зависят от позиции)
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim * num_layers)
        )
        self.num_layers = num_layers

        # Единый разделяемый MLP-блок
        self.shared_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_multiplier, hidden_dim),
            nn.Dropout(dropout)
        )

        # Индивидуальные LayerNorm для каждого прохода
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Выходной линейный слой – логиты для символов + NULL
        self.fc_logits = nn.Linear(hidden_dim, self.num_outputs)

    def _multi_resolution_pe(self, pos, d_model, num_octaves):
        batch_size = pos.size(0)
        device = pos.device
        pos = pos.float().unsqueeze(1)
        pe_parts = []
        for octave in range(num_octaves):
            scale = 2 ** octave
            freq = 10000.0 ** (torch.arange(0, d_model, 2, device=device) / d_model)
            scaled_pos = pos / scale
            sin_part = torch.sin(scaled_pos * freq)
            cos_part = torch.cos(scaled_pos * freq)
            oct_pe = torch.stack([sin_part, cos_part], dim=-1).flatten(-2)
            pe_parts.append(oct_pe)
        pe = torch.cat(pe_parts, dim=-1)
        return pe

    def forward(self, pooled, position):
        """
        pooled:   (B, hidden_dim)
        position: (B,)
        Возвращает dict с ключом 'symbol_probs' – (B, vocab_size+1), softmax вероятности.
        """
        x = self.pre_mlp(pooled)                          # (B, hidden_dim)

        pos_enc_raw = self._multi_resolution_pe(position, self.hidden_dim, self.num_octaves)
        pos_enc = self.pos_proj(pos_enc_raw)              # (B, hidden_dim)

        film_params = self.film_generator(pos_enc)        # (B, 2 * hidden_dim * num_layers)
        scales, shifts = film_params.chunk(2, dim=-1)     # каждый (B, hidden_dim * num_layers)

        # Цикл с разделяемым блоком и FiLM модуляцией
        for i in range(self.num_layers):
            scale_i = scales[:, i*self.hidden_dim:(i+1)*self.hidden_dim]
            shift_i = shifts[:, i*self.hidden_dim:(i+1)*self.hidden_dim]
            modulated = x * (1 + scale_i) + shift_i
            modulated = self.layer_norms[i](modulated)

            if self.use_checkpoint:
                block_out = torch.utils.checkpoint.checkpoint(
                    self.shared_block, modulated, use_reentrant=False
                )
            else:
                block_out = self.shared_block(modulated)

            x = x + block_out

        # Выход: логиты и вероятности
        logits = self.fc_logits(x)                        # (B, vocab_size+1)
        probs = F.softmax(logits, dim=-1)                 # сумма = 1, значения ∈ [0,1]

        return {'symbol_probs': probs}                    # ключ изменён