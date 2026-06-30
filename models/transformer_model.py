"""
Transformer 时间序列预测模型

基于标准 Transformer Encoder-Decoder 架构:
  - 位置编码使用可学习的 Positional Embedding
  - 编码器堆叠多个 Self-Attention + FFN 层
  - 解码器使用 Cross-Attention 关注编码器输出
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerModel(nn.Module):
    """Transformer Seq2Seq 预测模型"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder 输入嵌入（目标序列的 query）
        self.query_embed = nn.Parameter(torch.randn(1, 365, d_model))  # 最大 365 天
        self.decoder_pos = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出投影
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, target=None):
        """x: (B, input_len, D)"""
        B = x.size(0)
        output_len = target.size(1) if target is not None else 90

        # 编码器
        x = self.input_proj(x)  # (B, input_len, d_model)
        x = self.pos_encoding(x)
        memory = self.encoder(x)  # (B, input_len, d_model)

        # 解码器
        query = self.query_embed[:, :output_len, :].expand(B, -1, -1)  # (B, out_len, d_model)
        query = self.decoder_pos(query)
        out = self.decoder(query, memory)  # (B, out_len, d_model)
        out = self.output_proj(out).squeeze(-1)  # (B, out_len)

        return out
