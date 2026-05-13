import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# -------------------- ThoughtBlock (без изменений) --------------------
class ThoughtBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, memory: torch.Tensor, context_seq: torch.Tensor) -> torch.Tensor:
        norm_mem = self.norm1(memory)
        attn_out, _ = self.self_attn(norm_mem, norm_mem, norm_mem)
        memory = memory + attn_out

        norm_mem = self.norm2(memory)
        cross_out, _ = self.cross_attn(norm_mem, context_seq, context_seq)
        memory = memory + cross_out

        norm_mem = self.norm3(memory)
        ffn_out = self.ffn(norm_mem)
        memory = memory + ffn_out
        return memory


# -------------------- MikriModel (упрощённая) --------------------
class MikriModel(nn.Module):
    def __init__(self,
                 hidden_dim: int = 1024,
                 num_thought_blocks: int = 8,
                 num_memory_slots: int = 32,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint

        # Три проекции для разных ролей (вход – один pooled вектор)
        self.proj_current = nn.Linear(hidden_dim, hidden_dim)
        self.proj_temp_mem = nn.Linear(hidden_dim, hidden_dim)
        self.proj_deep_mem = nn.Linear(hidden_dim, hidden_dim)

        self.role_embeddings = nn.Parameter(torch.randn(3, hidden_dim))
        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, hidden_dim))

        self.thought_blocks = nn.ModuleList([
            ThoughtBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_thought_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)

        self.output_pooled = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.memory_slots, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)
        nn.init.normal_(self.role_embeddings, std=0.02)

    def forward(self,
                pooled_current: torch.Tensor,    # (B, hidden_dim)
                pooled_temp_mem: torch.Tensor,
                pooled_deep_mem: torch.Tensor
                ) -> torch.Tensor:
        batch_size = pooled_current.size(0)

        ctx_current = self.proj_current(pooled_current) + self.role_embeddings[0]
        ctx_temp    = self.proj_temp_mem(pooled_temp_mem) + self.role_embeddings[1]
        ctx_deep    = self.proj_deep_mem(pooled_deep_mem) + self.role_embeddings[2]

        context_seq = torch.stack([ctx_current, ctx_temp, ctx_deep], dim=1)  # (B, 3, hidden_dim)
        memory = self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)

        for block in self.thought_blocks:
            if self.use_checkpoint:
                memory = torch.utils.checkpoint.checkpoint(block, memory, context_seq, use_reentrant=False)
            else:
                memory = block(memory, context_seq)

        memory = self.final_norm(memory)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, memory, memory)
        new_pooled = self.output_pooled(pooled.squeeze(1))
        return new_pooled      # (B, hidden_dim)


# -------------------- TemporalMemoryBlock (упрощённая) --------------------
class TemporalMemoryBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_memory = nn.LayerNorm(hidden_dim)
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, query, memory, memory_padding_mask=None):
        norm_query = self.norm_query(query)
        norm_memory = self.norm_memory(memory)
        attn_out, _ = self.cross_attn(norm_query, norm_memory, norm_memory,
                                      key_padding_mask=memory_padding_mask)
        query = query + attn_out
        norm_query = self.norm_ffn(query)
        ffn_out = self.ffn(norm_query)
        query = query + ffn_out
        return query


class TemporalMemoryModel(nn.Module):
    def __init__(self,
                 hidden_dim: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 use_checkpoint: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint

        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.memory_self_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(2)
        ])
        self.memory_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])

        self.cross_attn_layers = nn.ModuleList([
            TemporalMemoryBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_pooled = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self,
                pooled_seq: torch.Tensor,          # (B, seq_len, hidden_dim)
                query_pooled: torch.Tensor,        # (B, hidden_dim)
                padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        batch_size, seq_len, _ = pooled_seq.shape

        # valid_mask: float маска, 1.0 для реальных позиций, 0.0 для паддинга
        if padding_mask is not None:
            valid_mask = (~padding_mask).float().unsqueeze(-1)  # (B, seq_len, 1)
        else:
            valid_mask = torch.ones(batch_size, seq_len, 1, device=pooled_seq.device)

        memory_hidden = self.memory_proj(pooled_seq) * valid_mask   # обнуляем паддинг

        query_hidden = self.query_proj(query_pooled).unsqueeze(1)   # (B, 1, hidden_dim)

        positions = torch.arange(seq_len, device=pooled_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions) * valid_mask.squeeze(-1).unsqueeze(-1)  # зануляем для паддинга
        memory_hidden = memory_hidden + pos_emb

        # Self-attention с последующим обнулением паддинга после каждого residual
        for attn, norm in zip(self.memory_self_attn, self.memory_layer_norms):
            if self.use_checkpoint:
                def block_fn(x, m, n):
                    normed = n(x)
                    out, _ = m(normed, normed, normed, key_padding_mask=padding_mask)
                    return x + out
                memory_hidden = torch.utils.checkpoint.checkpoint(
                    block_fn, memory_hidden, attn, norm, use_reentrant=False
                )
            else:
                normed = norm(memory_hidden)
                attn_out, _ = attn(normed, normed, normed, key_padding_mask=padding_mask)
                memory_hidden = memory_hidden + attn_out
            memory_hidden = memory_hidden * valid_mask   # обнуляем паддинг

        for layer in self.cross_attn_layers:
            if self.use_checkpoint:
                query_hidden = torch.utils.checkpoint.checkpoint(
                    layer, query_hidden, memory_hidden, padding_mask, use_reentrant=False
                )
            else:
                query_hidden = layer(query_hidden, memory_hidden, memory_padding_mask=padding_mask)

        retrieved_hidden = query_hidden.squeeze(1)
        retrieved_pooled = self.output_pooled(retrieved_hidden)
        return retrieved_pooled    # (B, hidden_dim)