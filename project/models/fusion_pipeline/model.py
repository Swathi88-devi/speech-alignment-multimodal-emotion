"""
Multimodal Fusion Pipeline Architecture
=======================================
Combines Speech (Acoustic) and Text (Lexical) modalities:
- Speech Branch: Conv1D + BiGRU + Temporal Self-Attention Pooling -> hs (128-d)
- Text Branch: Transformer Contextual Encoder + Masked Average Pooling -> ht (128-d)
- Fusion Block: Gated Multimodal Unit (GMU) / Cross-Modal Gating:
      g = sigmoid(Wg * [hs; ht] + bg)
      z = g * tanh(Ws * hs) + (1 - g) * tanh(Wt * ht)
- Latent Space: 128-dimensional Unified Joint Representation z
- Classifier Head: LayerNorm -> Dropout -> Linear -> 7 Emotion Classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel

class GatedMultimodalFusion(nn.Module):
    """Gated Multimodal Unit (GMU) for dynamic cross-modal gating."""
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        self.proj_speech = nn.Linear(latent_dim, latent_dim)
        self.proj_text = nn.Linear(latent_dim, latent_dim)
        self.post_norm = nn.LayerNorm(latent_dim)

    def forward(self, hs: torch.Tensor, ht: torch.Tensor):
        # hs: [B, latent_dim], ht: [B, latent_dim]
        concat_feat = torch.cat([hs, ht], dim=1) # [B, latent_dim * 2]
        gate = self.gate_layer(concat_feat)       # [B, latent_dim] in range [0, 1]

        h_s_tilde = torch.tanh(self.proj_speech(hs))
        h_t_tilde = torch.tanh(self.proj_text(ht))

        # Dynamic Gated Fusion
        z = gate * h_s_tilde + (1.0 - gate) * h_t_tilde # [B, latent_dim]
        z = self.post_norm(z)
        return z, gate

class MultimodalEmotionModel(nn.Module):
    def __init__(self, vocab_size: int = 205, latent_dim: int = 128, num_classes: int = 7):
        super().__init__()
        self.speech_encoder = SpeechEmotionModel(n_mels=128, hidden_dim=64, latent_dim=latent_dim, num_classes=num_classes)
        self.text_encoder = TextEmotionModel(vocab_size=vocab_size, embed_dim=64, latent_dim=latent_dim, num_classes=num_classes)
        
        self.fusion = GatedMultimodalFusion(latent_dim=latent_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, num_classes)
        )

    def extract_latents(self, mel: torch.Tensor, tokens: torch.Tensor):
        """Extracts hs, ht, and unified fused representation z."""
        hs = self.speech_encoder.extract_latent(mel)     # [B, 128]
        ht = self.text_encoder.extract_latent(tokens)    # [B, 128]
        z, gate = self.fusion(hs, ht)                    # [B, 128]
        return z, hs, ht, gate

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor):
        z, hs, ht, gate = self.extract_latents(mel, tokens)
        logits = self.classifier(z)
        return logits, z, hs, ht, gate
