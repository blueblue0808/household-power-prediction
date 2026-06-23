import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return self.pe[:, :x.size(1)]


# class PatchConvBlock(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         hidden_dim: int,
#         patch_size: int = 7,
#         stride: int = 3,    # 这是 Patch 切分的步长
#         dropout: float = 0.2
#     ):
#         super().__init__()

#         self.patch_size = patch_size
#         self.stride = stride

#         self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)  # stride 默认是 1
#         self.bn1 = nn.BatchNorm1d(hidden_dim)

#         self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)

#         self.act = nn.ReLU()
#         self.drop = nn.Dropout(dropout)
#         self.pool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [batch, seq_len, in_dim]
#         B, L, C = x.shape

#         x = x.transpose(1, 2)                         
#         x = self.act(self.bn1(self.conv1(x)))         
#         x = self.drop(x)
#         x = self.act(self.bn2(self.conv2(x)))         
#         x = self.drop(x)

#         # 滑动窗口切分成 Patch
#         patch_tokens = []
#         for start in range(0, L - self.patch_size + 1, self.stride):
#             patch = x[:, :, start:start + self.patch_size]  # [B, hidden_dim, patch_size]
#             patch_emb = self.pool(patch).squeeze(-1)        # [B, hidden_dim]
#             patch_tokens.append(patch_emb)

#         # 堆叠成序列： [num_patches, B, hidden_dim]
#         patch_seq = torch.stack(patch_tokens, dim=0)
#         return patch_seq

class PatchConvBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        patch_size: int = 7,
        stride: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # 用 Linear Projection 代替 AvgPool
        self.patch_proj = nn.Linear(hidden_dim * patch_size, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        B, L, C = x.shape

        x = x.transpose(1, 2)                          # [B, C, L]
        x = self.act(self.bn1(self.conv1(x)))          # [B, hidden_dim, L]
        x = self.drop(x)
        x = self.act(self.bn2(self.conv2(x)))          # [B, hidden_dim, L]
        x = self.drop(x)

        patch_tokens = []
        for start in range(0, L - self.patch_size + 1, self.stride):
            patch = x[:, :, start:start + self.patch_size]  # [B, hidden_dim, patch_size]
            patch_flat = patch.reshape(B, -1)               # [B, hidden_dim * patch_size]
            patch_emb = self.patch_proj(patch_flat)         # [B, hidden_dim]
            patch_tokens.append(patch_emb)

        # 补上最后一个 patch，覆盖到末尾
        last_start = L - self.patch_size
        if last_start % self.stride != 0:
            patch = x[:, :, last_start:L]                  # [B, hidden_dim, patch_size]
            patch_flat = patch.reshape(B, -1)              # [B, hidden_dim * patch_size]
            patch_emb = self.patch_proj(patch_flat)        # [B, hidden_dim]
            patch_tokens.append(patch_emb)

        patch_seq = torch.stack(patch_tokens, dim=0)       # [num_patches, B, hidden_dim]
        return patch_seq


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 90,
        dropout: float = 0.2,
        nhead: int = 8,
        patch_size: int = 7,
        stride: int = 3
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Patch 卷积块
        self.patch_conv = PatchConvBlock(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            stride=stride,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len=90, input_dim=13]
        x = self.embedding(x)  # [batch, 90, hidden_dim]
        # Patch 卷积：90天 → 约30个Patch
        patch_seq = self.patch_conv(x)                       # [num_patches, batch, hidden_dim]
        patch_seq = patch_seq.transpose(0, 1)                # [batch, num_patches, hidden_dim]
        patch_seq = patch_seq + self.pos_encoder(patch_seq)  # [batch, num_patches, hidden_dim]
        patch_seq = self.encoder(patch_seq)                  # [batch, num_patches, hidden_dim]
        token = patch_seq[:, -1, :]                          # [batch, hidden_dim]
        out = self.head(token)                               # [batch, pred_len]
        return out