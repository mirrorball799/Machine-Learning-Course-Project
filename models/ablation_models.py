"""
SCA-Net 消融实验变体模型

五种变体:
  A1: 去除变量分组 — 所有特征单一 Linear 投影
  A2: 去除多尺度卷积 — 标准单尺度 Conv1d
  A3: 去除跨组门控 — 各组均值直接求和
  A4: 去除两阶段解码 — 直接 MLP 映射
  A5: 去除 Transformer 编码器 — 仅保留卷积+解码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.improved_model import (
    GroupedVariableEncoder,
    MultiScaleTemporalConv,
    CrossGroupGatedFusion,
    CoarseToFineDecoder,
    LearnablePositionalEncoding,
)


# ============================================================
# A1: 去除变量分组 — 单一 Linear 编码
# ============================================================
class SCANet_NoGroup(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, num_encoder_layers=4,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = LearnablePositionalEncoding(d_model, dropout=dropout)

        self.ms_conv1 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.ms_conv2 = MultiScaleTemporalConv(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.decoder = CoarseToFineDecoder(d_model, n_heads, dim_ff, dropout)

    def forward(self, x, group_ids=None, target=None):
        output_len = target.size(1) if target is not None else 90
        fused = self.input_proj(x)
        fused = self.pos_encoding(fused)
        ms_out = self.ms_conv2(self.ms_conv1(fused))
        fused = fused + ms_out
        fused = self.encoder(fused)
        return self.decoder(fused, output_len)


# ============================================================
# A2: 去除多尺度卷积 — 标准单尺度 Conv1d
# ============================================================
class SCANet_NoMSC(nn.Module):
    def __init__(self, group_sizes, d_model=256, n_heads=8, num_encoder_layers=4,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        num_groups = len(group_sizes)
        self.group_encoder = GroupedVariableEncoder(group_sizes, d_model, dropout)

        # 标准单尺度卷积替代多尺度
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.fusion = CrossGroupGatedFusion(d_model, num_groups)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.decoder = CoarseToFineDecoder(d_model, n_heads, dim_ff, dropout)

    def forward(self, x, group_ids, target=None):
        output_len = target.size(1) if target is not None else 90
        fused, group_stack = self.group_encoder(x, group_ids)
        # 单尺度卷积
        conv_out = self.conv(fused.transpose(1, 2)).transpose(1, 2)
        conv_out = self.dropout(self.norm(fused + conv_out))
        fused = self.fusion(group_stack) + conv_out
        fused = self.encoder(fused)
        return self.decoder(fused, output_len)


# ============================================================
# A3: 去除跨组门控 — 各组均值直接求和
# ============================================================
class SCANet_NoGate(nn.Module):
    def __init__(self, group_sizes, d_model=256, n_heads=8, num_encoder_layers=4,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        self.group_encoder = GroupedVariableEncoder(group_sizes, d_model, dropout)
        self.ms_conv1 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.ms_conv2 = MultiScaleTemporalConv(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = CoarseToFineDecoder(d_model, n_heads, dim_ff, dropout)

    def forward(self, x, group_ids, target=None):
        output_len = target.size(1) if target is not None else 90
        fused, group_stack = self.group_encoder(x, group_ids)
        ms_out = self.ms_conv2(self.ms_conv1(fused))
        # 各组直接取均值求和无门控
        fused = group_stack.mean(dim=1) + ms_out
        fused = self.encoder(fused)
        return self.decoder(fused, output_len)


# ============================================================
# A4: 去除两阶段解码 — 直接 MLP
# ============================================================
class SCANet_NoCoarseFine(nn.Module):
    def __init__(self, group_sizes, d_model=256, n_heads=8, num_encoder_layers=4,
                 dim_ff=512, dropout=0.1, max_out=365):
        super().__init__()
        num_groups = len(group_sizes)
        self.group_encoder = GroupedVariableEncoder(group_sizes, d_model, dropout)
        self.ms_conv1 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.ms_conv2 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.fusion = CrossGroupGatedFusion(d_model, num_groups)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 直接 MLP 替代两阶段解码
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, max_out),
        )

    def forward(self, x, group_ids, target=None):
        output_len = target.size(1) if target is not None else 90
        fused, group_stack = self.group_encoder(x, group_ids)
        ms_out = self.ms_conv2(self.ms_conv1(fused))
        fused = self.fusion(group_stack) + ms_out
        fused = self.encoder(fused)
        out = self.mlp(fused.mean(dim=1))[:, :output_len]
        return out


# ============================================================
# A5: 去除 Transformer 编码器 — 仅卷积+解码
# ============================================================
class SCANet_NoTransformer(nn.Module):
    def __init__(self, group_sizes, d_model=256, n_heads=8,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        num_groups = len(group_sizes)
        self.group_encoder = GroupedVariableEncoder(group_sizes, d_model, dropout)
        self.ms_conv1 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.ms_conv2 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.ms_conv3 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.fusion = CrossGroupGatedFusion(d_model, num_groups)
        self.decoder = CoarseToFineDecoder(d_model, n_heads, dim_ff, dropout)

    def forward(self, x, group_ids, target=None):
        output_len = target.size(1) if target is not None else 90
        fused, group_stack = self.group_encoder(x, group_ids)
        ms_out = self.ms_conv3(self.ms_conv2(self.ms_conv1(fused)))
        fused = self.fusion(group_stack) + ms_out
        return self.decoder(fused, output_len)


# ============================================================
# 模型注册表
# ============================================================
ABLATION_MODELS = {
    "full": "SCANet",
    "no_group": SCANet_NoGroup,
    "no_msc": SCANet_NoMSC,
    "no_gate": SCANet_NoGate,
}
