import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 90,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # out: [batch, seq_len, hidden_dim]
        last_hidden = out[:, -1, :]    # 取最后一个时间步的输出
        pred = self.head(last_hidden)  # [batch, pred_len]
        return pred