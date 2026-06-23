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


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 90,
        dropout: float = 0.2,
        nhead: int = 8
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim)

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
        x = self.embedding(x)              # [batch, 90, hidden_dim]
        x = x + self.pos_encoder(x)        # [batch, 90, hidden_dim] 加上位置编码
        x = self.encoder(x)                # [batch, 90, hidden_dim]
        x = x[:, -1, :]                    # [batch, hidden_dim] 取最后一个时间步
        out = self.head(x)                 # [batch, pred_len]
        return out