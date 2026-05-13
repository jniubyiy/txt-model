import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AdaLN(nn.Module):
    """Adaptive Layer Normalization с предсказанием scale и shift из условия."""
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Блок трансформера с AdaLN и остаточными связями.
       key_padding_mask позволяет игнорировать паддинговые позиции.
    """
    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int = 8, 
                 ff_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.adaln1(x, cond)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out

        norm_x = self.adaln2(x, cond)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        return x


class AttentionPooling(nn.Module):
    """Обучаемый attention‑pooling: один запрос собирает информацию со всех позиций."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(query, x, x,
                              key_padding_mask=~mask if mask is not None else None)
        return pooled.squeeze(1)  # (B, D)


class Encoder(nn.Module):
    """Энкодер: на выходе один вектор `pooled` (B, hidden_dim)."""
    def __init__(self,
                 input_emb_dim: int,
                 hidden_dim: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_multiplier: int = 4,
                 dropout: float = 0.1,
                 max_seq_len: int = 1024,
                 use_checkpoint: bool = False):
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint

        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.length_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.input_proj = nn.Linear(input_emb_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, hidden_dim, num_heads, ff_multiplier, dropout)
            for _ in range(num_layers)
        ])

        self.pooling = AttentionPooling(hidden_dim)

    def forward(self,
                text_emb_seq: torch.Tensor,
                lengths: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        batch_size, max_len, _ = text_emb_seq.shape
        device = text_emb_seq.device

        # mask: True для реальных позиций, False для паддинга
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        x = self.input_proj(text_emb_seq)

        pos = positions.clamp(min=0, max=self.max_seq_len - 1)
        pos_emb = self.pos_embedding(pos)
        x = x + pos_emb

        len_idx = torch.clamp(lengths, max=self.max_seq_len - 1)
        cond = self.length_embedding(len_idx)

        # key_padding_mask: True для позиций, которые нужно игнорировать (паддинг)
        key_padding_mask = ~mask

        for layer in self.layers:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    lambda _x, _cond, _mask: layer(_x, _cond, key_padding_mask=_mask),
                    x, cond, key_padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, cond, key_padding_mask=key_padding_mask)

        pooled = self.pooling(x, mask)          # (B, hidden_dim)
        return pooled