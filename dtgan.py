from functools import partial
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from einops import rearrange
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, m: torch.Tensor = None, dummy_arg: torch.Tensor = None
    ):
        assert dummy_arg is not None, print(
            f"dummy arg should be a tensor with gradient, but get None."
        )

        if m is not None:
            x = self.module(x, m)
        else:
            x = self.module(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Union[int, None],
        out_features: Union[int, None],
        act_layer: any = nn.GELU,
        drop_prob: float = 0.0,
    ):
        """
        Transformer 中的多层感知机(前馈神经网络), 其输入张量 shape 应为 [B H W C], 输出具有和输入相同的 shape.
        :param in_features: 输入特征数, 与指定的 PatchEmbed 的输出 channel 相同
        :param hidden_features: mlp 中间隐藏特征数
        :param out_features: 输出特征数, 与 in_features 相同
        :param act_layer: mlp 中的激活函数, Transformer 用的是 nn.GELU
        :param drop_prob: 多层感知机包括两个 Linear 层, 每个 Linear 后跟一个 Dropout 层, drop_prob 为参数丢弃率
        """
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        # input x: [B H W C]
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.drop(self.fc2(x))
        # output x: [B H W C]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Union[int, float, None] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
    ):
        """
        Transformer Encoder 中的核心多头自注意力机制, 输入张量的 shape 应为 [B H W C], 输出张量的 shape 与输入相同
        :param dim: 输入张量的 channel 数, 与指定的 PatchEmbed 的输出 channel 相同
        :param num_heads: 将输入张量在 channel 维度分为 num_heads 个头, num_heads 应该被 dim 整除
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param attn_drop_prob: 计算注意力后的参数丢弃率
        :param proj_drop_prob: 最后输出层的参数丢弃率
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def forward(self, x):
        # input x: [batch_size, h, w, embed_dim]
        b, h, w, c = x.shape

        # qkv: [B, H, W, C] -> [B, H, W, C * 3]
        # rearrange: [B, H, W, 3C] -> [3, B, num_heads, H*W, dim_per_head]
        qkv = rearrange(
            self.qkv(x),
            "b h w (c num_heads dim_per_head) -> c b num_heads (h w) dim_per_head",
            c=3,
            num_heads=self.num_heads,
            dim_per_head=c // self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q: [B, num_heads, N, dim_per_head], k: [B, num_heads, N, dim_per_head]
        # k rearrange: [B, num_heads, N, dim_per_head] -> [B, num_heads, dim_per_head, N]
        # q @ k (矩阵乘法): [B, num_heads, N, N]
        attn = (
            q @ rearrange(k, "b num_heads n dim_per_head -> b num_heads dim_per_head n")
        ) * self.scale
        attn = self.attn_drop(attn.softmax(-1))

        # attn: [B, num_heads, N, N]
        # v: [B, num_heads, N, dim_per_head]
        # @: [B, num_heads, N, dim_per_head]
        # rearrange: [B, num_heads, N, dim_per_head] -> [B, N, num_heads * dim_per_head]
        x = rearrange(
            attn @ v,
            "b num_heads (h w) dim_per_head -> b h w (num_heads dim_per_head)",
            h=h,
            w=w,
        )
        x = self.proj_drop(self.proj(x))
        # output x: [B H W C(embed_dim)]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Union[int, float, None] = None,
        proj_drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.0,
        act_layer: any = nn.GELU,
        norm_layer: any = nn.LayerNorm,
    ):
        """
        构成 Transformer Encoder 的基本块, 包括一个 MultiHeadAttention 以及一个 MLP,
        输入张量的 shape 应为 [B C(embed_dim) H(grid_size[0]) W(grid_size[1])], 输出张量与输入张量 shape 相同.
        :param dim: 输入张量的 channel 数, 与指定的 PatchEmbed 的输出 channel 相同
        :param num_heads: MultiHeadAttention 中头的数量
        :param mlp_ratio: MultiHeadAttention 输出的张量 channel 将乘以指定的比例, 作为 MLP 中的中间隐藏维度
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param proj_drop_prob: MLP 中的参数丢弃率
        :param attn_drop_prob: MultiHeadAttention 中的参数丢弃率
        :param drop_path_prob: Encoder Block 整体的参数丢弃率
        :param act_layer: MLP 中的激活层, 默认为 nn.GELU
        :param norm_layer: Encoder Block 中的 norm 层, 默认为 nn.LayerNorm
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.multi_head_attention = MultiHeadAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop_prob, proj_drop_prob
        )
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, act_layer, proj_drop_prob)

    def forward(self, x):
        # input x: [batch_size, embed_dim, h, w]
        x = rearrange(x, "b c h w -> b h w c")
        x = x + self.drop_path(self.multi_head_attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, "b h w c -> b c h w")
        # output x: [batch_size, embed_dim, h, w]
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: Union[int, float, None] = None,
        proj_drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.0,
        pos_embed: bool = False,
        act_layer: any = nn.GELU,
        norm_layer: any = nn.LayerNorm,
        use_checkpoint: bool = True,
    ):
        """
        Transformer Encoder, 包含 n 个 Transformer Encoder Block 以及一个可学习的位置编码,
        输入张量的 shape 为 [B C H W], 输出张量 shape 与输入相同
        :param dim: 输入张量的 channel 数量, 等于 PatchEmbed 的输出 channel
        :param depth: Transformer Encoder 中包含的 Transformer Encoder Block 数量
        :param num_heads: MultiHeadAttention 中头的个数
        :param mlp_ratio: MultiHeadAttention 输出的张量 channel 将乘以指定的比例, 作为 MLP 中的中间隐藏维度
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param proj_drop_prob: MLP 中的参数丢弃率
        :param attn_drop_prob: MultiHeadAttention 中的参数丢弃率
        :param drop_path_prob: Encoder Block 整体的参数丢弃率
        :param pos_embed: bool 值, 是否使用位置编码
        :param act_layer: MLP 中的激活层, 默认为 nn.GELU
        :param norm_layer: Encoder Block 中的 norm 层, 默认为 nn.LayerNorm
        :param use_checkpoint: 是否使用 checkpoint 来节省训练显存占用
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ModuleWrapperIgnores2ndArg(
                    TransformerEncoderBlock(
                        dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias,
                        qk_scale,
                        proj_drop_prob,
                        attn_drop_prob,
                        drop_path_prob,
                        act_layer,
                        norm_layer,
                    )
                )
            )
        self.pos_embed = pos_embed
        if self.pos_embed:
            self.pos_embed_layer = nn.Conv2d(
                dim, dim, kernel_size=3, padding=1, groups=dim
            )

        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        # x: [batch_size, embed_dim, h, w]
        if self.pos_embed:
            x = x + self.pos_embed_layer(x)
        res = x
        # ModuleList 没有实现 forward 方法, 因此必须遍历其子模块, 直接使用 self.layers(x) 会报错
        for layer_wrapper in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer_wrapper, x, None, self.dummy_tensor)
            else:
                x = layer_wrapper(x, None, self.dummy_tensor)
        return x + res


class MultiHeadMixedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Union[int, float, None] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
    ):
        """
        Transformer Decoder 中的混合多头自注意力机制, 输入为两个张量, 在本例中张量 x 来自上一个 Decoder 的输出,
        张量 m 来自多尺度的 Encoder; 与 MultiHeadAttention 不同的地方在于计算 q, k, v 时, q 来自张量 x, k, v 来自张量 m,
        输出张量的 shape 与 x 相同.
        :param dim: 输入张量的 channel 数, 与指定的 PatchEmbed 的输出 channel 相同
        :param num_heads: 将输入张量在 channel 维度分为 num_heads 个头, num_heads 应该被 dim 整除
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param attn_drop_prob: 计算注意力后的参数丢弃率
        :param proj_drop_prob: 最后输出层的参数丢弃率
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def forward(self, x, m):
        # m 来自 encoder 的输入, 由此得到 k, v; x 来自最后一个 encoder 的输出以及前面 decoder 的输出,
        # x 和 m 只需要通道相同，尺寸可以不同
        # input x: [batch_size, h, w, embed_dim]
        b, h, w, c = x.shape
        # [q, k, v]: [b h w c] -> [b num_heads (h*w) dim_per_head]
        q = rearrange(
            self.to_q(x),
            "b h w (num_heads dim_per_head) -> b num_heads (h w) dim_per_head",
            num_heads=self.num_heads,
            dim_per_head=c // self.num_heads,
        )
        k = rearrange(
            self.to_k(m),
            "b h w (num_heads dim_per_head) -> b num_heads (h w) dim_per_head",
            num_heads=self.num_heads,
            dim_per_head=c // self.num_heads,
        )
        v = rearrange(
            self.to_v(m),
            "b h w (num_heads dim_per_head) -> b num_heads (h w) dim_per_head",
            num_heads=self.num_heads,
            dim_per_head=c // self.num_heads,
        )
        # q: [B, num_heads, N, dim_per_head]
        # k: [B, num_heads, N, dim_per_head]
        # k rearrange: [B, num_heads, N, dim_per_head] -> [B, num_heads, dim_per_head, N]
        # q @ k (矩阵乘法): [B, num_heads, N, N]
        attn = (
            q @ rearrange(k, "b num_heads n dim_per_head -> b num_heads dim_per_head n")
        ) * self.scale
        attn = self.attn_drop(attn.softmax(-1))
        # attn: [B, num_heads, N, N]
        # v: [B, num_heads, N, dim_per_head]
        # @: [B, num_heads, N, dim_per_head]
        # rearrange: [B, num_heads, N, dim_per_head] -> [B, N, num_heads * dim_per_head]
        x = rearrange(
            attn @ v,
            "b num_heads (h w) dim_per_head -> b h w (num_heads dim_per_head)",
            h=h,
            w=w,
        )
        x = self.proj_drop(self.proj(x))
        # output x shape: [B H W C]
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Union[int, float, None] = None,
        proj_drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.0,
        act_layer: any = nn.GELU,
        norm_layer: any = nn.LayerNorm,
    ):
        """
        构成 Transformer Decoder 的基本块, 包括一个 MultiHeadAttention, 一个 MultiHeadMixedAttention 以及一个 MLP,
        输入为两个张量, 在本例中张量 x 来自上一个 Decoder 的输出, 张量 m 来自多尺度的 Encoder;
        MultiHeadMixedAttention 与 MultiHeadAttention 不同的地方在于计算 q, k, v 时, q 来自张量 x, k, v 来自张量 m,
        输出一个张量, shape 与 x 相同.
        :param dim: 输入张量的 channel 数, 与指定的 PatchEmbed 的输出 channel 相同
        :param num_heads: MultiHeadMixedAttention 和 MultiHeadAttention 中头的数量
        :param mlp_ratio: MultiHeadAttention 输出的张量 channel 将乘以指定的比例, 作为 MLP 中的中间隐藏维度
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param proj_drop_prob: MLP 中的参数丢弃率
        :param attn_drop_prob: MultiHeadMixedAttention 和 MultiHeadAttention 中的参数丢弃率
        :param drop_path_prob: Decoder Block 整体的参数丢弃率
        :param act_layer: MLP 中的激活层, 默认为 nn.GELU
        :param norm_layer: Decoder Block 中的 norm 层, 默认为 nn.LayerNorm
        """
        super().__init__()
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        )
        self.norm1 = norm_layer(dim)
        self.multi_head_attention = MultiHeadAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop_prob, proj_drop_prob
        )
        self.norm2 = norm_layer(dim)
        self.multi_head_mixed_attention = MultiHeadMixedAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop_prob, proj_drop_prob
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm3 = norm_layer(dim)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, act_layer, proj_drop_prob)

    def forward(self, x, m):
        # input x: [batch_size, embed_dim, h, w]
        x = rearrange(x, "b c h w -> b h w c")
        m = rearrange(m, "b c h w -> b h w c")
        x = x + self.drop_path(self.multi_head_attention(self.norm1(x)))
        x = x + self.drop_path(
            self.multi_head_mixed_attention(self.norm2(x), self.norm2(m))
        )
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        x = rearrange(x, "b h w c -> b c h w")
        # output x: [batch_size, embed_dim, h, w]
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: Union[int, float, None] = None,
        proj_drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.0,
        pos_embed: bool = False,
        act_layer: any = nn.GELU,
        norm_layer: any = nn.LayerNorm,
        use_checkpoint: bool = True,
    ):
        """
        Transformer Decoder, 包含 n 个 Transformer Decoder Block 以及一个可学习的位置编码,
        输入为两个张量, 在本例中张量 x 来自上一个 Decoder 的输出, 张量 m 来自多尺度的 Encoder;
        MultiHeadMixedAttention 与 MultiHeadAttention 不同的地方在于计算 q, k, v 时, q 来自张量 x, k, v 来自张量 m,
        输出一个张量, shape 与 x 相同.
        :param dim: 输入张量的 channel 数, 与指定的 PatchEmbed 的输出 channel 相同
        :param depth: Transformer Decoder 中的 Transformer Decoder Block 数量
        :param num_heads: MultiHeadMixedAttention 和 MultiHeadAttention 中头的数量
        :param mlp_ratio: MultiHeadAttention 输出的张量 channel 将乘以指定的比例, 作为 MLP 中的中间隐藏维度
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param proj_drop_prob: MLP 中的参数丢弃率
        :param attn_drop_prob: MultiHeadMixedAttention 和 MultiHeadAttention 中的参数丢弃率
        :param drop_path_prob: Decoder Block 整体的参数丢弃率
        :param act_layer: MLP 中的激活层, 默认为 nn.GELU
        :param norm_layer: Decoder Block 中的 norm 层, 默认为 nn.LayerNorm
        :param use_checkpoint: 是否使用 checkpoint 来节省训练显存占用
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ModuleWrapperIgnores2ndArg(
                    TransformerDecoderBlock(
                        dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias,
                        qk_scale,
                        proj_drop_prob,
                        attn_drop_prob,
                        drop_path_prob,
                        act_layer,
                        norm_layer,
                    )
                )
            )
        self.pos_embed = pos_embed
        if self.pos_embed:
            self.pos_embed_layer = nn.Conv2d(
                dim, dim, kernel_size=3, padding=1, groups=dim
            )
        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x, m):
        # x: [batch_size, embed_dim, h, w]
        # m: [batch_size, embed_dim, h, w]
        res = x
        if self.pos_embed:
            x = x + self.pos_embed_layer(x)
        # ModuleList 没有实现 forward 方法, 因此必须遍历其子模块, 直接使用 self.layers(x) 会报错
        for layer_wrapper in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer_wrapper, x, m, self.dummy_tensor)
            else:
                x = layer_wrapper(x, m, self.dummy_tensor)
        return x + res


class DilateAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Union[int, float, None] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        dilation: int = 3,
        unfold_kernel_size: int = 3,
    ):
        """
        计算 unfold_kernel_size 大小窗口, 空洞率为 dilation 的局部单尺度多头注意力, 输入张量的 shape 为 [B H W C],
        输出张量与输入张量相同
        :param dim: 输入张量的 channel 数, 与指定的 PatchEmbed 的输出 channel 相同
        :param num_heads: 将输入张量在 channel 维度分为 num_heads 个头, num_heads 应该被 dim 整除
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param attn_drop_prob: 空洞注意力的丢弃率
        :param proj_drop_prob: 线性变换输出层的丢弃率
        :param dilation: 卷积时的空洞率
        :param unfold_kernel_size: 用于空洞卷积的卷积核的大小
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            unfold_kernel_size, dilation, dilation * (3 - 1) // 2, 1
        )
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def forward(self, x):
        # input x shape: [B H(grid_size[0]) W(grid_size[1] C(embed_dim))]
        b, h, w, c = x.shape

        # 计算 QKV
        # qkv: [B H W C] -> [B H W 3C]
        # rearrange: [B H W 3C] -> [3 B C H W]
        qkv = rearrange(self.qkv(x), "b h w (c c1) -> c b c1 h w", c=3, c1=c)
        # q, k, v: [B C H W]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # unfold(): [B, C, H, W] -> [B, C * ks * ks(unfold_kernel_size), H * W]
        k = self.unfold(k)
        # rearrange: [B, C * ks * ks, H * W] -> [B, num_heads, H * W, dim_per_head, ks * ks]
        k = rearrange(
            k,
            "b (num_heads dim_per_head k) n -> b num_heads n dim_per_head k",
            num_heads=self.num_heads,
            dim_per_head=c // self.num_heads,
        )

        # rearrange(q): [B C H W] -> [B num_heads N(H*W) 1 dim_per_head]
        q = rearrange(
            q,
            "b (num_heads dim_per_head) h w -> b num_heads (h w) 1 dim_per_head",
            num_heads=self.num_heads,
            dim_per_head=c // self.num_heads,
        )
        # q: [B num_heads N(H*W) 1 dim_per_head]
        # k: [B, num_heads, H * W, dim_per_head, ks * ks]
        # attn: [B, num_heads, H * W, 1, ks * ks]
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # unfold(): [B, C, H, W] -> [B, C * ks * ks(unfold_kernel_size), H * W]
        v = self.unfold(v)
        # rearrange: [B, C * ks * ks, H * W] -> [B, num_heads, H * W, k, dim_per_head]
        v = rearrange(
            v,
            "b (num_heads dim_per_head k) n -> b num_heads n k dim_per_head",
            num_heads=self.num_heads,
            dim_per_head=c // self.num_heads,
        )

        # attn: [B, num_heads, H * W, 1, k(ks * ks)]
        # v: [B, num_heads, H * W, k, dim_per_head]
        # attn @ v: [B, num_heads, H * W, 1, dim_per_head]
        x = attn @ v
        # rearrange: [B, num_heads, H * W, 1, dim_per_head] -> [B H W C]
        x = rearrange(
            x,
            "b num_heads (h w) new_dim dim_per_head -> b h w (num_heads new_dim dim_per_head)",
            h=h,
            w=w,
        )
        # output x shape: [B H W C]
        return x


class DilateTransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dilation: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Union[int, float, None] = None,
        unfold_kernel_size: int = 3,
        proj_drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.0,
        act_layer: any = nn.GELU,
        norm_layer: any = nn.LayerNorm,
    ):
        """
        构成 Dilate Transformer Encoder 的基本块, 包括一个 DilateAttention 以及一个 MLP,
        输入张量的 shape 应为 [B C(embed_dim) H(grid_size[0]) W(grid_size[1])], 输出张量与输入张量 shape 相同.
        :param dim: 输入张量的 channel 数, 与指定的 PatchEmbed 的输出 channel 相同
        :param num_heads: MultiHeadAttention 中头的数量
        :param dilation: 计算空洞注意力时的空洞率
        :param mlp_ratio: MultiHeadAttention 输出的张量 channel 将乘以指定的比例, 作为 MLP 中的中间隐藏维度
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param unfold_kernel_size: 计算空洞卷积时的卷积核大小
        :param proj_drop_prob: MLP 中的参数丢弃率
        :param attn_drop_prob: MultiHeadAttention 中的参数丢弃率
        :param drop_path_prob: Encoder Block 整体的参数丢弃率
        :param act_layer: MLP 中的激活层, 默认为 nn.GELU
        :param norm_layer: Encoder Block 中的 norm 层, 默认为 nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dilation = dilation
        self.norm1 = norm_layer(dim)
        self.attn = DilateAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            dilation=dilation,
            unfold_kernel_size=unfold_kernel_size,
        )
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop_prob=proj_drop_prob,
        )

    def forward(self, x):
        # x: [b c h w]
        x = rearrange(x, "b c h w -> b h w c")
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, "b h w c -> b c h w")
        # output shape: [b c h w]
        return x


class DilateTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        dilation: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: Union[int, float, None] = None,
        unfold_kernel_size: int = 3,
        proj_drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.0,
        pos_embed: bool = False,
        act_layer: any = nn.GELU,
        norm_layer: any = nn.LayerNorm,
        use_checkpoint: bool = True,
    ):
        """
        Dilate Transformer Encoder, 包含 n 个 Dilate Transformer Encoder Block 以及一个可学习的位置编码,
        输入张量的 shape 为 [B C H W], 输出张量 shape 与输入相同
        :param dim: 输入张量的 channel 数量, 等于 PatchEmbed 的输出 channel
        :param depth: Transformer Encoder 中包含的 Transformer Encoder Block 数量
        :param num_heads: MultiHeadAttention 中头的个数
        :param dilation: 计算空洞注意力时的空洞率
        :param mlp_ratio: MultiHeadAttention 输出的张量 channel 将乘以指定的比例, 作为 MLP 中的中间隐藏维度
        :param qkv_bias: 输入张量通过线性层得到 q, k, v 张量时, 线性层是否具有偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的 channel 数的开根号
        :param unfold_kernel_size: 计算空洞注意力时的卷积核大小
        :param proj_drop_prob: MLP 中的参数丢弃率
        :param attn_drop_prob: MultiHeadAttention 中的参数丢弃率
        :param drop_path_prob: Encoder Block 整体的参数丢弃率
        :param pos_embed: bool 值, 是否使用位置编码
        :param act_layer: MLP 中的激活层, 默认为 nn.GELU
        :param norm_layer: Dilate Encoder Block 中的 norm 层, 默认为 nn.LayerNorm
        :param use_checkpoint: 是否使用 checkpoint 来节省训练显存占用
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ModuleWrapperIgnores2ndArg(
                    DilateTransformerEncoderBlock(
                        dim=dim,
                        num_heads=num_heads,
                        dilation=dilation,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        unfold_kernel_size=unfold_kernel_size,
                        proj_drop_prob=proj_drop_prob,
                        attn_drop_prob=attn_drop_prob,
                        drop_path_prob=drop_path_prob,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                )
            )
        self.pos_embed = pos_embed
        if self.pos_embed:
            self.pos_embed_layer = nn.Conv2d(
                dim, dim, kernel_size=3, padding=1, groups=dim
            )

        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        # x: [batch_size, embed_dim, h, w]
        res = x
        if self.pos_embed:
            x = x + self.pos_embed_layer(x)
        # ModuleList 没有实现 forward 方法, 因此必须遍历其子模块, 直接使用 self.layers(x) 会报错
        for layer_wrapper in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer_wrapper, x, None, self.dummy_tensor)
            else:
                x = layer_wrapper(x, None, self.dummy_tensor)
        return x + res


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, tuple] = 80,
        patch_size: Union[int, tuple] = 8,
        data_channel: int = 64,
        embed_dim: int = 384,
        embed_type: str = "down_sample",
    ):
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        if (
            self.img_size[0] % self.patch_size[0] != 0
            or self.img_size[1] % self.patch_size[1] != 0
        ):
            raise ValueError(
                f"patch size {self.patch_size} is not divisible by image size {self.img_size}."
            )

        self.embed_type = embed_type
        if self.embed_type == "patch_merge":
            channel = data_channel * self.patch_size[0] * self.patch_size[1]
            self.proj = nn.Conv2d(channel, embed_dim, kernel_size=1)
        elif self.embed_type == "down_sample":
            self.proj = nn.Conv2d(
                data_channel,
                embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        else:
            raise ValueError(f"No support embed type named {self.embed_type}.")

    def forward(self, x):
        if self.embed_type == "patch_merge":
            # rearrange: [B C H W] -> [B C*patch_size[0]*patch_size[1] grid_size[0] grid_size[1]]
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (p1 p2 c) h w",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
            # proj: [B, data_channel * patch_size ** 2, grid_size[0], grid_size[1]]
            # -> [B, embed_dim, grid_size[0], grid_size[1]]
            x = self.proj(x)
        elif self.embed_type == "down_sample":
            x = self.proj(x)
        # output x shape: [B embed_dim grid_size[0] grid_size[1]]
        return x


class UpSampler(nn.Module):
    def __init__(
        self,
        dim: int,
        upscale: int = 3,
        act_layer: any = partial(nn.LeakyReLU, 0.2, True),
        norm_layer: any = nn.BatchNorm2d,
    ):
        """
        将图像一次上采样 n 倍, 输入张量的形状为 [B C H W]
        :param dim: 输入张量的特征数
        :param upscale: 将输入张量上采样的倍数
        :param act_layer: 上采样后是否使用激活层
        :param norm_layer: 上采样后是否使用 norm 层
        """
        super().__init__()
        up_body = [
            nn.Conv2d(dim, dim * upscale**2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
        ]
        if norm_layer:
            up_body.append(norm_layer(dim))
        if act_layer:
            up_body.append(act_layer())
        self.body = nn.Sequential(*up_body)

    def forward(self, x):
        return self.body(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        act_layer: any = partial(nn.LeakyReLU, 0.2, True),
        norm_layer: any = nn.BatchNorm2d,
    ):
        super().__init__()
        body = [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)]
        if norm_layer:
            body.append(norm_layer(dim))
        body.append(act_layer())
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        act_layer: any = partial(nn.LeakyReLU, 0.2, True),
        norm_layer: any = nn.BatchNorm2d,
        res_scale: float = 0.2,
    ):
        super().__init__()
        body = []
        for i in range(2):
            body.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True))
            if norm_layer:
                body.append(norm_layer(dim))
            if i == 0:
                body.append(act_layer())
        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = x
        x = self.body(x).mul(self.res_scale)
        return x + res


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.ca(x)


class CBAM(nn.Module):
    def __init__(self, dim: int = 64, reduction: int = 16):
        super(CBAM, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # spatial attention
        self.conv2d = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3
        )
        self.sigmoid = nn.Sigmoid()

    def forward_channel_attention(self, x):
        avg_pool = self.shared_MLP(self.avg_pool(x))
        max_pool = self.shared_MLP(self.max_pool(x))
        attention = self.sigmoid(avg_pool + max_pool)
        return attention * x

    def forward_spatial_attention(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pools = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv2d(pools))
        return attention * x

    def forward(self, x):
        channel_out = self.forward_channel_attention(x)
        spatial_out = self.forward_spatial_attention(channel_out)
        return spatial_out


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        act_layer: any = partial(nn.LeakyReLU, 0.2, True),
        norm_layer: any = None,
        res_scale: float = 0.2,
    ):
        super().__init__()

        reduction = 16
        body = []
        for i in range(2):
            body.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True))
            if norm_layer:
                body.append(norm_layer(dim))
            if i == 0:
                body.append(act_layer())
            if i == 1:
                # body.append(ChannelAttention(dim, reduction=reduction))
                body.append(CBAM(dim, reduction=reduction))
        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = x
        x = self.body(x).mul(self.res_scale)
        return x + res


class BackBone(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        backbone_type: str = "residual",
        backbone_depth: int = 10,
        act_layer: any = partial(nn.LeakyReLU, 0.2, True),
        norm_layer: any = nn.BatchNorm2d,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "basic":
            backbone = [
                BasicBlock(dim, act_layer, norm_layer) for _ in range(backbone_depth)
            ]
        elif backbone_type == "residual":
            backbone = [
                ResBlock(dim, act_layer, norm_layer) for _ in range(backbone_depth)
            ]
        elif backbone_type == "residual_channel_attention":
            backbone = [
                ResidualChannelAttentionBlock(dim, act_layer, norm_layer)
                for _ in range(backbone_depth)
            ]
        else:
            raise ValueError(f"No backbone type {backbone_type}.")
        self.backbone_wrapper = ModuleWrapperIgnores2ndArg(nn.Sequential(*backbone))
        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        res = x
        if self.use_checkpoint and self.training:
            x = checkpoint(self.backbone_wrapper, x, None, self.dummy_tensor)
        else:
            x = self.backbone_wrapper(x, None, self.dummy_tensor)

        return x + res


class MyNet(nn.Module):
    def __init__(
        self,
        img_size: Union[int, tuple] = (80, 80),
        patch_size: Union[int, tuple] = (4, 4),
        data_channel: int = 4,
        upscale: int = 3,
        upscale_pos: str = "tail",
        backbone_type: str = "residual_channel_attention",
        hidden_dim: int = 64,
        backbone_depth: int = 5,
        embed_dim: int = 96,
        embed_type: str = "patch_merge",
        encoder_depths: Union[tuple, list] = (1, 1, 1, 1),
        encoder_num_heads: Union[tuple, list] = (6, 6, 6, 6),
        encoder_dilate_ratios: Union[tuple, list] = (1, 2, 3, 4),
        encoder_pos_embed: bool = True,
        unfold_kernel_size: int = 3,
        decoder_depths: Union[tuple, list] = (1, 1, 1),
        decoder_num_heads: Union[tuple, list] = (6, 6, 6),
        decoder_pos_embed: bool = True,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: Union[int, float, None] = None,
        proj_drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.1,
        use_checkpoint: bool = True,
    ):
        """
        构建多尺度混合注意力 Transformer 超分辨率网络
        :param img_size: 输入的低分辨率图像尺寸
        :param patch_size: 将 backbone 的输出张量划分为 patch_size 大小的补丁
        :param data_channel: 输入图像的通道数
        :param upscale: 超分辨率上采样倍数
        :param backbone_type: 用于浅层特征提取的网络类型, "basic", "residual" or "residual_channel_attention"
        :param hidden_dim: 浅层特征主干网络中的隐层特征数
        :param backbone_depth: 浅层特征提取网络的深度
        :param embed_dim: 将浅层特征输入 Transformer Encoder 之前, 将其编码为 patch 后的输出特征数
        :param embed_type: 编码浅层特征的方式, "patch_merge" 或者 "down_sample"
        :param encoder_depths: 元组或者列表, 列表或元组的长度为 encoder 的数量, 相应的元素为 encoder 中 block 的数量
        :param encoder_num_heads: Encoder 中每一个 Block 中的头的个数, 其长度应该与 encoder_depths 相等
        :param encoder_dilate_ratios: 输入是元组或列表, 编码器计算 DilateAttention 时的空洞率
        :param encoder_pos_embed: 是否在编码器中使用位置编码
        :param unfold_kernel_size: 用于计算局部空洞注意力的卷积核的大小
        :param decoder_depths: 解码器的深度, 应该是一个元组或者列表, 相应的元素为 encoder 中 block 的数量
        :param decoder_num_heads: Encoder 中每一个 Block 中的头的个数, 其长度应该与 Decoder_depths 相等
        :param decoder_pos_embed: 是否在解码器中使用位置编码
        :param mlp_ratio: mlp 中的隐藏特征数等于, 输入 mlp 时的特征数与之相乘
        :param qkv_bias: 在计算 qkv 时的线性变换层是否使用偏置
        :param qk_scale: 计算注意力时的缩放系数, 默认为每个头的特征数开根号
        :param proj_drop_prob: mlp 中的参数丢弃率
        :param attn_drop_prob: 多头注意力中的参数丢弃率
        :param drop_path_prob: Encoder Block 和 Decoder Block 整体的参数丢弃率
        :param use_checkpoint: 是否使用 checkpoint 降低显存使用
        """
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.head = nn.Conv2d(data_channel, hidden_dim, kernel_size=3, padding=1)
        self.backbone_list = nn.ModuleList(
            BackBone(
                hidden_dim, backbone_type, backbone_depth, use_checkpoint=use_checkpoint, norm_layer=None
            )
            for _ in range(len(encoder_depths))
        )
        self.upscale = upscale
        self.upscale_pos = upscale_pos
        # self.up_sampler = UpSampler(hidden_dim, upscale=upscale)
        # 1X1 卷积，用于对输出特征降维
        reduction = 4
        patch_dim = int(hidden_dim / reduction)
        self.conv1x1_list = nn.ModuleList(
            nn.Conv2d(hidden_dim, patch_dim, kernel_size=1)
            for _ in range(len(encoder_depths))
        )
        # 降维后的数据进行补丁向量化
        self.patch_embed_list = nn.ModuleList(
            PatchEmbed(img_size, patch_size, patch_dim, embed_dim, embed_type)
            for _ in range(len(encoder_depths))
        )
        # 向量重新变成图像补丁
        self.embed_patch = nn.Conv2d(
            embed_dim,
            patch_dim * self.patch_size[0] * self.patch_size[1],
            kernel_size=1,
        )
        # 模型尾部卷积
        self.tail = nn.Sequential(
            nn.Conv2d(patch_dim, patch_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(patch_dim, patch_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(patch_dim, patch_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(patch_dim, data_channel, kernel_size=3, padding=1),
        )

        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for num_head, depth, dilate_ratio in zip(
            encoder_num_heads, encoder_depths, encoder_dilate_ratios
        ):
            if dilate_ratio:
                self.encoder_list.append(
                    DilateTransformerEncoder(
                        dim=embed_dim,
                        depth=depth,
                        num_heads=num_head,
                        dilation=dilate_ratio,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        unfold_kernel_size=unfold_kernel_size,
                        proj_drop_prob=proj_drop_prob,
                        attn_drop_prob=attn_drop_prob,
                        drop_path_prob=drop_path_prob,
                        pos_embed=encoder_pos_embed,
                        use_checkpoint=use_checkpoint,
                    )
                )
            else:
                self.encoder_list.append(
                    TransformerEncoder(
                        dim=embed_dim,
                        depth=depth,
                        num_heads=num_head,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        proj_drop_prob=proj_drop_prob,
                        attn_drop_prob=attn_drop_prob,
                        drop_path_prob=drop_path_prob,
                        pos_embed=encoder_pos_embed,
                        use_checkpoint=use_checkpoint,
                    )
                )
        for num_head, depth in zip(decoder_num_heads, decoder_depths):
            self.decoder_list.append(
                TransformerDecoder(
                    dim=embed_dim,
                    depth=depth,
                    num_heads=num_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    proj_drop_prob=proj_drop_prob,
                    attn_drop_prob=attn_drop_prob,
                    drop_path_prob=drop_path_prob,
                    pos_embed=decoder_pos_embed,
                    use_checkpoint=use_checkpoint,
                )
            )

    def forward(self, x):
        # head(x): [B C H W] -> [B hidden_dim H W]
        x = self.head(x)
        features = []
        for backbone in self.backbone_list:
            x = backbone(x)
            features.append(x)
        # features = [backbone(x) for backbone in self.backbone_list]
        if self.upscale_pos == "middle":
            features[0] = F.interpolate(
                features[0], scale_factor=self.upscale, mode="bicubic"
            )

        encoder_outputs = [
            encoder(patch_embed(conv1x1(feature)))
            for feature, conv1x1, patch_embed, encoder in zip(
                features, self.conv1x1_list, self.patch_embed_list, self.encoder_list
            )
        ]
        decoder_output = encoder_outputs[0]
        for index, decoder in enumerate(self.decoder_list):
            decoder_output = decoder(decoder_output, encoder_outputs[index + 1])
        # tail
        x = self.embed_patch(decoder_output)
        x = rearrange(
            x,
            "b (p1 p2 c) h w -> b c (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        if self.upscale_pos == "tail":
            x = nn.functional.interpolate(x, scale_factor=self.upscale, mode="bicubic")
        x = self.tail(x)

        return x


# -----------------------------------------------------Discriminator--------------------------------------
class UNetGan(nn.Module):
    def __init__(
        self,
        data_channel: int,
        hidden_dim: int,
    ) -> None:
        super(UNetGan, self).__init__()

        self.conv1 = nn.Conv2d(data_channel, 64, (3, 3), (1, 1), (1, 1))
        self.down_block1 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    hidden_dim, int(hidden_dim * 2), (4, 4), (2, 2), (1, 1), bias=False
                )
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    int(hidden_dim * 2),
                    int(hidden_dim * 4),
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    int(hidden_dim * 4),
                    int(hidden_dim * 8),
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block1 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    int(hidden_dim * 8),
                    int(hidden_dim * 4),
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    int(hidden_dim * 4),
                    int(hidden_dim * 2),
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    int(hidden_dim * 2), hidden_dim, (3, 3), (1, 1), (1, 1), bias=False
                )
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3), (1, 1), (1, 1), bias=False)
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3), (1, 1), (1, 1), bias=False)
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Conv2d(hidden_dim, 1, (3, 3), (1, 1), (1, 1))

    def forward(self, x):
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x):
        out1 = self.conv1(x)

        # Down-sampling
        down1 = self.down_block1(out1)
        down2 = self.down_block2(down1)
        down3 = self.down_block3(down2)
        # Up-sampling
        down3 = F.interpolate(
            down3, scale_factor=2, mode="bilinear", align_corners=False
        )
        up1 = self.up_block1(down3)

        up1 = torch.add(up1, down2)
        up1 = F.interpolate(up1, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = self.up_block2(up1)

        up2 = torch.add(up2, down1)
        up2 = F.interpolate(up2, scale_factor=2, mode="bilinear", align_corners=False)
        up3 = self.up_block3(up2)

        up3 = torch.add(up3, out1)

        out = self.conv2(up3)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


# if __name__ == "__main__":
#     from tensorboardX import SummaryWriter
#     from thop import profile
#
#     t = torch.rand(1, 4, 80, 80, requires_grad=True).cuda()
#     kw = {
#         "patch_size": (4, 4),
#         "backbone_type": "residual_channel_attention",
#         "encoder_depths": (8, 8, 8, 8),
#         "decoder_depths": (8, 8, 8),
#         "encoder_pos_embed": True,
#         "decoder_pos_embed": True,
#         "encoder_dilate_ratios": (False, False, False, False),
#     }
#     net = MyNet(img_size=(80, 80), **kw).cuda()
#     output = net(t)
#     print(output.shape)
#     # #
#     # with SummaryWriter(log_dir="./model_log") as sw:  # 实例化 SummaryWriter ,可以自定义数据输出路径
#     #     sw.add_graph(net, t)  # 输出网络结构图
#     #     sw.close()  # 关闭  sw
#
#     # view net by tensorboard: tensorboard --logdir=F:\Python\mycode\SRForSeg\models\model_log
#
#     flops, params = profile(net, inputs=(t,))
#     print(
#         "the flops is {}G,the params is {}M".format(
#             round(flops / (10**9), 2), round(params / (10**6), 2)
#         )
#     )
