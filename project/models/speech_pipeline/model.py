"""
Speech Pipeline Architecture: Temporal Modelling & Classifier
=============================================================
Architecture:
- Feature Extraction: 128-band Log Mel-Spectrogram [B, 128, T]
- Temporal Modelling: Conv1D Spectro-Temporal Extractor -> 2-Layer BiGRU -> Multi-Head Self-Attention Pooling
- Latent Space: 128-dimensional Speech Latent Vector hs
- Classifier: LayerNorm -> Dropout -> Linear Projection -> 7 Emotion Classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionPooling(nn.Module):
    """Multi-Head Temporal Self-Attention Pooling layer."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.v = nn.Linear(hidden_dim // 2, 1, bias=False)

    def forward(self, x):
        # x: [B, T, D]
        energy = torch.tanh(self.proj(x))  # [B, T, D // 2]
        scores = self.v(energy)             # [B, T, 1]
        weights = F.softmax(scores, dim=1) # [B, T, 1]
        pooled = torch.sum(x * weights, dim=1) # [B, D]
        return pooled, weights

class SpeechEmotionModel(nn.Module):
    def __init__(self, n_mels: int = 128, hidden_dim: int = 64, latent_dim: int = 128, num_classes: int = 7):
        super().__init__()
        
        # 1. Spectro-Temporal Convolutional Layers
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Temporal Modelling: Bidirectional GRU
        # Input to GRU is [B, T', 128], output is [B, T', hidden_dim * 2] = [B, T', 128]
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 3. Temporal Self-Attention Pooling
        self.attention_pool = TemporalAttentionPooling(hidden_dim * 2)
        
        # 4. Latent Projection to 128-d
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        
        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, num_classes)
        )

    def extract_latent(self, mel: torch.Tensor) -> torch.Tensor:
        """Extracts the 128-d temporal latent vector hs."""
        x = self.conv_block(mel)         # [B, 128, T']
        x = x.permute(0, 2, 1)          # [B, T', 128]
        gru_out, _ = self.gru(x)        # [B, T', 128]
        pooled, _ = self.attention_pool(gru_out) # [B, 128]
        hs = self.latent_proj(pooled)   # [B, 128]
        return hs

    def forward(self, mel: torch.Tensor):
        hs = self.extract_latent(mel)   # [B, 128]
        logits = self.classifier(hs)    # [B, num_classes]
        return logits, hs
