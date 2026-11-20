"""
Text Pipeline Architecture: Contextual Modelling & Classifier
=============================================================
Architecture:
- Feature Extraction: Token Embedding (Vocab 205 -> 64-d) + Learnable Positional Encoding
- Contextual Modelling: 2-layer Multi-Head Transformer Encoder (d_model=64, nhead=4, dim_feedforward=128)
- Contextual Pooling: Sequence Mean Pooling -> Linear Projection to 128-d Latent Vector ht
- Classifier Head: LayerNorm -> Dropout -> Linear -> 7 Emotion Classes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TextEmotionModel(nn.Module):
    def __init__(self, vocab_size: int = 205, embed_dim: int = 64, latent_dim: int = 128, num_classes: int = 7):
        super().__init__()
        
        # 1. Feature Extraction: Token Embedding + Positional Encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=16)
        
        # 2. Contextual Modelling: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Contextual Latent Projection to 128-d
        self.latent_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, num_classes)
        )

    def extract_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extracts the 128-d contextual latent vector ht."""
        # tokens: [B, T]
        mask = (tokens == 0) # [B, T] padding mask
        x = self.embedding(tokens) * math.sqrt(64) # [B, T, embed_dim]
        x = self.pos_encoder(x)
        ctx = self.transformer_encoder(x, src_key_padding_mask=mask) # [B, T, embed_dim]
        
        # Masked Average Pooling
        expanded_mask = (~mask).unsqueeze(-1).float() # [B, T, 1]
        pooled = (ctx * expanded_mask).sum(dim=1) / (expanded_mask.sum(dim=1) + 1e-6) # [B, embed_dim]
        
        ht = self.latent_proj(pooled) # [B, latent_dim]
        return ht

    def forward(self, tokens: torch.Tensor):
        ht = self.extract_latent(tokens)
        logits = self.classifier(ht)
        return logits, ht
