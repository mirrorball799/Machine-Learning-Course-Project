"""
LSTM 序列到序列预测模型

Encoder-Decoder 架构:
  - 编码器: 多层双向 LSTM 将输入序列编码为上下文向量
  - 解码器: 单向 LSTM 自回归生成预测序列
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        # 合并双向输出 → hidden_dim
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        # 将双向 hidden state 投影为单向 (num_layers, B, hidden)
        self.h_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x: (B, input_len, D)
        out, (h, c) = self.lstm(x)
        # out: (B, input_len, hidden*2)
        out = self.out_proj(out)  # (B, input_len, hidden)

        # h, c: (num_layers*2, B, hidden)
        # 重排为 (num_layers, B, hidden*2) 再投影
        B = h.size(1)
        h = h.view(self.num_layers, 2, B, self.hidden_dim)
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)  # (num_layers, B, hidden*2)
        h = self.h_proj(h)  # (num_layers, B, hidden)

        c = c.view(self.num_layers, 2, B, self.hidden_dim)
        c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
        c = self.c_proj(c)

        return out, (h, c)


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            output_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, encoder_out, h0, c0, target_len: int,
                teacher_forcing_ratio: float = 0.0, target=None):
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=device)
        outputs = []

        h, c = h0.to(device).contiguous(), c0.to(device).contiguous()

        for t in range(target_len):
            out, (h, c) = self.lstm(decoder_input, (h, c))
            pred = self.fc(out)  # (B, 1, 1)
            outputs.append(pred)

            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t : t + 1].unsqueeze(-1)
            else:
                decoder_input = pred

        return torch.cat(outputs, dim=1).squeeze(-1)  # (B, output_len)


class LSTMModel(nn.Module):
    """LSTM Seq2Seq 预测模型"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(1, hidden_dim, num_layers, dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x, target=None, teacher_forcing_ratio: float = 0.5):
        encoder_out, (h, c) = self.encoder(x)
        output_len = target.size(1) if target is not None else 90
        return self.decoder(encoder_out, h, c, output_len, teacher_forcing_ratio, target)
