"""
SCA-Net: Seasonal-Channel Attention Network（改进模型）

设计思路:
  1. 变量分组编码 — 按物理意义将 11+ 个特征分为 3 组，保留电学变量间的结构化关系
  2. 多尺度时序卷积 — 每组用 3 路并行膨胀卷积捕获日/周/月级模式
  3. 跨组门控融合 + ProbSparse 自注意力 — 动态决定各时间步的变量组权重
  4. 粗-细两阶段解码 — 先预测整体趋势，再补充高频细节

参考文献:
  - Autoformer (Wu et al., NeurIPS 2021): 时序分解思想
  - Informer (Zhou et al., AAAI 2021): ProbSparse 注意力
  - PatchTST (Nie et al., ICLR 2023): 通道独立编码思想
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 变量分组编码
# ============================================================
class GroupedVariableEncoder(nn.Module):
    """按物理意义将输入特征分组，每组独立做线性投影 + 位置编码"""

    def __init__(
        self,
        group_sizes: list,  # 每组变量数, e.g. [6, 2, 5] 对应 [功率组, 电学组, 气象组]
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_groups = len(group_sizes)
        self.group_proj = nn.ModuleList([
            nn.Linear(size, d_model) for size in group_sizes
        ])
        # 每组学习一个 group embedding 用于区分
        self.group_embed = nn.Parameter(torch.randn(1, len(group_sizes), 1, d_model))
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len=1000, dropout=dropout)

    def forward(self, x, group_ids: list):
        """
        Args:
            x: (B, seq_len, D_total) 原始输入
            group_ids: 各组列索引列表, e.g. [[0,1,2,3,4,5], [6,7], [8,9,10,11,12]]
        Returns:
            fused: (B, seq_len, d_model) 融合后的表示
        """
        B, T, _ = x.shape
        group_outputs = []

        for g, proj in enumerate(self.group_proj):
            g_x = x[:, :, group_ids[g]]  # (B, T, C_g)
            g_out = proj(g_x)  # (B, T, d_model)
            g_out = g_out + self.group_embed[:, g, :, :]  # 组标识
            group_outputs.append(g_out)

        # 堆叠各组输出 → (B, num_groups, T, d_model)
        stacked = torch.stack(group_outputs, dim=1)
        # 沿组维度取平均融合 → (B, T, d_model)
        fused = stacked.mean(dim=1)
        fused = self.pos_encoding(fused)
        return fused, stacked


# ============================================================
# 2. 多尺度时序卷积
# ============================================================
class MultiScaleTemporalConv(nn.Module):
    """多尺度膨胀卷积: 3 路并行，膨胀率分别为 [1, 3, 7]，捕获日/周/月级模式

    dilation=1  → 感受野 ~3天 (日模式)
    dilation=3  → 感受野 ~9天 (周模式)
    dilation=7  → 感受野 ~21天 (月模式)
    """

    def __init__(self, d_model: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # 显式计算 padding 以兼容不同 PyTorch 版本
        p1 = (kernel_size - 1) * 1 // 2  # dilation=1 → pad=1
        p3 = (kernel_size - 1) * 3 // 2  # dilation=3 → pad=3
        p7 = (kernel_size - 1) * 7 // 2  # dilation=7 → pad=7
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=p1, dilation=1, groups=d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size, padding=p3, dilation=3, groups=d_model)
        self.conv7 = nn.Conv1d(d_model, d_model, kernel_size, padding=p7, dilation=7, groups=d_model)
        self.pointwise = nn.Conv1d(d_model * 3, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        """x: (B, T, d_model) → (B, T, d_model)"""
        x_t = x.transpose(1, 2)  # (B, d_model, T)
        c1 = self.conv1(x_t)
        c3 = self.conv3(x_t)
        c7 = self.conv7(x_t)

        # 可学习软权重加权后 concat → pointwise 融合回 d_model
        w = F.softmax(self.gate, dim=0)
        concat = torch.cat([w[0] * c1, w[1] * c3, w[2] * c7], dim=1)  # (B, d_model*3, T)
        fused = self.pointwise(concat).transpose(1, 2)  # (B, T, d_model)
        return self.dropout(self.norm(x + fused))


# ============================================================
# 3. 跨组门控融合
# ============================================================
class CrossGroupGatedFusion(nn.Module):
    """跨变量组门控注意力融合

    动态决定各时间步各组的重要性权重
    """

    def __init__(self, d_model: int = 256, num_groups: int = 3):
        super().__init__()
        self.num_groups = num_groups
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_groups),
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, group_stack):
        """
        Args:
            group_stack: (B, num_groups, T, d_model) 各组输出
        Returns:
            (B, T, d_model) 融合后的特征
        """
        B, G, T, D = group_stack.shape

        # 用各组均值计算门控权重
        avg_repr = group_stack.mean(dim=1)  # (B, T, D)
        gate = F.softmax(self.gate_proj(avg_repr), dim=-1)  # (B, T, G)

        # gate: (B, T, G) → (B, G, T, 1) 与 group_stack 对齐
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B, G, T, 1)
        fused = (group_stack * gate).sum(dim=1)  # (B, T, D)
        return self.output_proj(fused)


# ============================================================
# 4. ProbSparse 自注意力
# ============================================================
class ProbSparseSelfAttention(nn.Module):
    """ProbSparse 自注意力机制 (Informer 风格)

    只计算 Top-u 个最重要 query 的完整注意力，其余取均值近似，降低复杂度。
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.factor = factor

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK_sparsity_measure(self, Q, K):
        """用 KL 散度近似度量 query 稀疏性"""
        # Q: (B, H, L, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 取 top-factor 个最大值的均值作为稀疏度量
        top_k = max(1, int(self.factor * math.log(K.size(-2))))
        top_scores = torch.topk(scores, top_k, dim=-1)[0]
        return top_scores.mean(-1)  # (B, H, L)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 选取最重要的 u 个 query
        with torch.no_grad():
            sparsity = self._prob_QK_sparsity_measure(Q, K)  # (B, H, T)
            u = max(1, self.factor * int(math.log(T)))
            u = min(u, T)
            top_u_idx = torch.topk(sparsity, u, dim=-1)[1]  # (B, H, u)

        # 对选中的 query 计算完整注意力
        Q_reduced = torch.gather(Q, 2, top_u_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
        attn_scores = torch.matmul(Q_reduced, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out_reduced = torch.matmul(attn_weights, V)  # (B, H, u, d_k)

        # 其余 query 用 V 的均值近似
        V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, T - u, -1)
        # 构建完整输出
        out = V_mean.clone()
        out = out.scatter(2, top_u_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k), out_reduced)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ============================================================
# 5. 粗-细两阶段解码器
# ============================================================
class CoarseToFineDecoder(nn.Module):
    """两阶段解码:
    - Stage 1: MLP 预测趋势轮廓 (低频)
    - Stage 2: Cross-Attention 补充高频细节
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        # 趋势预测
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, 365),  # 最大输出 365 天
        )

        # 细节补充
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.detail_norm = nn.LayerNorm(d_model)
        self.detail_ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, 1),
        )

    def forward(self, memory, output_len: int):
        """memory: (B, T, d_model)"""
        B, T, D = memory.shape

        # Stage 1: 趋势预测
        memory_pooled = memory.mean(dim=1)  # (B, d_model) 全局池化
        trend = self.trend_proj(memory_pooled)[:, :output_len]  # (B, out_len)

        # Stage 2: 细节补充
        # 将趋势作为 query 做 cross-attention
        query = trend.unsqueeze(-1).expand(-1, -1, D)  # (B, out_len, d_model)
        detail, _ = self.cross_attn(query, memory, memory)
        detail = self.detail_norm(detail + query)
        detail = self.detail_ffn(detail).squeeze(-1)  # (B, out_len)

        return trend + detail


# ============================================================
# 辅助模块
# ============================================================
class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pos_embed[:, : x.size(1), :])


# ============================================================
# SCA-Net 完整模型
# ============================================================
class SCANet(nn.Module):
    """Seasonal-Channel Attention Network（改进模型）

    Args:
        group_sizes: 各组变量数量, 如 [6, 2, 7]
        d_model: 隐藏维度
        n_heads: 注意力头数
        num_encoder_layers: Transformer 编码器层数
        dim_ff: FFN 中间维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        group_sizes: list,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_groups = len(group_sizes)

        # 模块 1: 分组编码
        self.group_encoder = GroupedVariableEncoder(group_sizes, d_model, dropout)

        # 模块 2: 多尺度卷积 × 2 层
        self.ms_conv1 = MultiScaleTemporalConv(d_model, dropout=dropout)
        self.ms_conv2 = MultiScaleTemporalConv(d_model, dropout=dropout)

        # 模块 3: 跨组融合
        self.fusion = CrossGroupGatedFusion(d_model, num_groups)

        # 模块 4: Transformer Encoder (标准 MultiheadAttention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 模块 5: 粗-细解码器
        self.decoder = CoarseToFineDecoder(d_model, n_heads, dim_ff, dropout)

    def forward(self, x, group_ids: list, target=None):
        """x: (B, input_len, D), target: (B, output_len)"""
        output_len = target.size(1) if target is not None else 90

        # 1. 分组编码
        fused, group_stack = self.group_encoder(x, group_ids)

        # 2. 多尺度卷积
        ms_out = self.ms_conv1(fused)
        ms_out = self.ms_conv2(ms_out)

        # 3. 跨组融合
        fused = self.fusion(group_stack) + ms_out

        # 4. Transformer Encoder
        fused = self.encoder(fused)

        # 5. 粗-细解码
        output = self.decoder(fused, output_len)  # (B, out_len)
        return output
