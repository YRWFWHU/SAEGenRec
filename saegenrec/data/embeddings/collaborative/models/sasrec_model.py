"""SASRec (Self-Attentive Sequential Recommendation) model.

nn.Module implementation aligned with RecBole's SASRec architecture:
- Item embedding with padding_idx=0
- Learnable position embedding
- Stacked SASRecBlock (MultiHeadAttention + FFN + LayerNorm + Dropout)
- Combined causal + padding attention mask using -1e4 (not -inf) to avoid NaN
- BPR / CrossEntropy loss for training
- Tied weights for prediction (dot product with item_embedding)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MASK_VALUE = -1e4


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        self.item_embedding = nn.Embedding(
            num_items + 1, hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.attention_layers = nn.ModuleList(
            [SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """RecBole-style weight initialization for all sub-modules."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()

    def _make_attention_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create combined padding + causal attention mask (RecBole style).

        Uses _MASK_VALUE (-1e4) instead of -inf to prevent NaN in softmax
        when all keys are masked (e.g., padding-only query positions).

        Returns:
            Float mask of shape [B * num_heads, L, L].
            0.0 = attend, _MASK_VALUE = block.
        """
        B, L = seq.shape
        padding_mask = (seq != 0).float()
        key_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, L, -1)
        causal = torch.tril(torch.ones(1, 1, L, L, device=seq.device))
        combined = key_mask * causal
        attn_mask = (1.0 - combined) * _MASK_VALUE
        attn_mask = (
            attn_mask.expand(-1, self.num_heads, -1, -1)
            .reshape(B * self.num_heads, L, L)
        )
        return attn_mask

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Forward pass returning sequence representations.

        Args:
            seq: Item ID sequences, shape [B, L]. 0 means padding.

        Returns:
            Hidden states, shape [B, L, hidden_size].
        """
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)

        x = self.item_embedding(seq) + self.position_embedding(positions)
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)

        attn_mask = self._make_attention_mask(seq)

        for layer in self.attention_layers:
            x = layer(x, attn_mask=attn_mask)

        return x

    def bpr_loss(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BPR (Bayesian Personalized Ranking) loss.

        Uses RecBole-style log(sigmoid(x) + 1e-10) for numerical stability.

        Args:
            seq: Input sequences [B, L], 0-padded.
            pos: Positive (next-item) targets [B, L].
            neg: Negative samples [B, L].

        Returns:
            Scalar BPR loss.
        """
        hidden = self.forward(seq)

        pos_emb = self.item_embedding(pos)
        neg_emb = self.item_embedding(neg)

        pos_score = (hidden * pos_emb).sum(dim=-1)
        neg_score = (hidden * neg_emb).sum(dim=-1)

        valid_mask = (pos != 0).float()
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10) * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1)

    def ce_loss(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CrossEntropy loss over all items (RecBole default).

        Args:
            seq: Input sequences [B, L], 0-padded.
            pos: Positive (next-item) targets [B, L]. 0 = padding (ignored).

        Returns:
            Scalar CE loss.
        """
        hidden = self.forward(seq)
        logits = torch.matmul(hidden, self.item_embedding.weight.T)
        logits = logits.view(-1, self.num_items + 1)
        target = pos.view(-1)
        return F.cross_entropy(logits, target, ignore_index=0)

    def predict(self, seq: torch.Tensor) -> torch.Tensor:
        """Predict scores for all items using the last position.

        Args:
            seq: Input sequences [B, L].

        Returns:
            Scores [B, num_items] (excluding padding item 0).
        """
        hidden = self.forward(seq)
        last_hidden = hidden[:, -1, :]

        all_item_emb = self.item_embedding.weight[1:]
        scores = torch.matmul(last_hidden, all_item_emb.t())
        return scores

    def extract_item_embeddings(self) -> torch.Tensor:
        """Extract trained item embeddings (excluding padding index 0).

        Returns:
            Tensor of shape [num_items, hidden_size].
        """
        return self.item_embedding.weight.data[1:].clone()


class SASRecBlock(nn.Module):
    """Single SASRec transformer block: MultiHeadAttention + FFN + residuals."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn_layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = PositionwiseFeedForward(hidden_size, hidden_size * 4, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.attn_layernorm(x)
        x, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            is_causal=False,
        )
        x = x + residual

        residual = x
        x = self.ffn_layernorm(x)
        x = self.ffn(x)
        x = x + residual

        return x


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))
