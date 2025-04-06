from collections import deque
from dataclasses import dataclass
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import RMSNorm
from torch.nn.modules.normalization import RMSNorm
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Literal, Union, Any
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator
from syntht2i import ShapeDataset

# Original code: https://github.com/feizc/DiT-MoE/blob/main/models.py

def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach()) # .cpu())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads




@dataclass
class TokenMixerParameters:
    use_mmdit: bool = True
    use_ec: bool = False
    use_moe: bool = False
    shared_mod: bool = False
    embed_dim: int = 1152
    num_heads: int = 1152 // 64
    num_layers: int = 2
    mlp_ratio: int = 2
    num_experts: int = 8
    capacity_factor: int = 2.0
    pretraining_tp: int = 2
    num_shared_experts: int = 2
    exp_ratio: int = 4
    dropout: float = 0.1

class TokenMixer(nn.Module):
    """
    Each layer expects:
        - img:       [B, L_img, embed_dim]
        - txt:       [B, L_txt, embed_dim]
        - vec:       [B, embed_dim]            (conditioning vector for Modulation)
        - h          Height of the original image
        - w          Width of the original image
    and returns the updated (img, txt) after `num_layers` of DoubleStreamBlock.
    """
    def __init__(
        self,
        params: TokenMixerParameters,
    ):
        super().__init__()
        self.use_mmdit = params.use_mmdit
        self.shared_mod = params.shared_mod
        if self.shared_mod:
            self.mod = Modulation(params.embed_dim, True)

        if params.use_mmdit:
            self.layers = nn.ModuleList([
                DoubleStreamBlock(
                    hidden_size=params.embed_dim,
                    num_heads=params.num_heads,
                    mlp_dim=params.mlp_ratio * params.embed_dim,
                    num_experts=params.num_experts,
                    capacity_factor=params.capacity_factor,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=params.num_shared_experts,
                    exp_ratio=params.exp_ratio,
                    use_moe=params.use_moe,
                    use_expert_choice=params.use_ec,
                    shared_mod=params.shared_mod
                )
                for _ in range(params.num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                DiTBlock(
                    hidden_size=params.embed_dim,
                    num_heads=params.num_heads,
                    mlp_dim=params.mlp_ratio * params.embed_dim,
                    num_experts=params.num_experts,
                    num_experts_per_tok=params.capacity_factor,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=params.num_shared_experts,
                    use_moe=params.use_moe,
                    use_expert_choice=params.use_ec,
                    dropout=params.dropout
                )
                for _ in range(params.num_layers)
            ])

    def forward(
        self,
        img: torch.Tensor,       # [B, L_img, embed_dim]
        txt: torch.Tensor,       # [B, L_txt, embed_dim]
        vec: torch.Tensor,       # [B, embed_dim]
        pe: torch.Tensor,    # rope positional encoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_mod:
            mod = self.mod
        else:
            mod = None

        if self.use_mmdit:
            for layer in self.layers:
                img, txt = layer(img, txt, vec, pe, mod=mod)
        else:
            img = torch.cat((txt, img), 1)

            for layer in self.layers:
                img = layer(img, vec, pe, mod=mod)

            img = img[:, txt.shape[1]:, ...]
            txt = img[:, :txt.shape[1], ...]

        return img, txt





def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, dropout: float = 0.0) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def rope_ids(bs, h, w, seq_len, device, dtype):
    img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w, device=device, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt_ids = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)

    return txt_ids, img_ids

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)





def sincos_2d(embed_dim, h, w):
    """
    :param embed_dim: dimension of the embedding
    :param h: height of the grid
    :param w: width of the grid
    :return: [h*w, embed_dim] or [1+h*w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = sincos_2d_from_grid(embed_dim, grid)
    return pos_embed

def sincos_2d_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = sincos_1d(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = sincos_1d(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
    return emb

def sincos_1d(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, num_layers: int = 2):
        super().__init__()
        layers = max(num_layers, 1)

        if layers == 1:
            self.mlp = nn.Linear(in_dim, out_dim, bias=True)
        else:
            if hidden_dim is None:
                hidden_dim = max(in_dim, out_dim)
            
            mlp_layers = []
            mlp_layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            mlp_layers.append(nn.GELU())
            
            for _ in range(layers - 2):
                mlp_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                mlp_layers.append(nn.GELU())

            mlp_layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
            
            self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = MLPEmbedder(frequency_embedding_size, hidden_size, num_layers=1)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D torch.Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) torch.Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # if inference with fp16, embedding.half()
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size) 
        t_emb = self.mlp(t_freq)#.half())
        return t_emb
    
class OutputLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLPEmbedder(in_dim, out_dim, hidden_dim=in_dim*2, num_layers=2)
        self.adaLN_modulation = nn.Sequential(nn.GELU(), nn.Linear(in_dim, 2 * in_dim, bias=True))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.mlp(x)
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, E, H', W')
        return x.flatten(2).transpose(1, 2)  # (B, E, H', W') -> (B, H'*W', E)
    
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int = 10_000, axes_dim: list[int] = [16, 56, 56]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)



class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

    def modulate(self, x):
        return (1 + self.scale) * x + self.shift

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.linear(self.silu(vec)[:, None, :]).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
    
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, dropout=(self.dropout if self.training else 0.0))
        x = self.proj(x)
        return x

#################################################################################
#                                 Core DiT Modules                              #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_dim,
        num_experts=8, 
        num_experts_per_tok=2, 
        pretraining_tp=2, 
        num_shared_experts=2, 
        use_moe: bool = False, 
        use_expert_choice: bool = False, 
        dropout=0.1,
        shared_mod=False,
        shared_attn_projs=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if not shared_attn_projs:
            self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size, bias=True),
        )

        if not shared_mod:
            self.mod = Modulation(hidden_size, True)

    def forward(self, x, c, pe, mod=None, attn=None):
        if mod is not None:
            msa, mlp = mod(c)
        else:
            msa, mlp = self.mod(c)

        if attn is not None:
            attn = attn
        else:
            attn = self.attn

        # x = x + msa.gate.unsqueeze(1) * self.attn(modulate(self.norm1(x).to(x.dtype), msa.shift, msa.scale))
        x = x + msa.gate * attn(msa.modulate(self.norm1(x)), pe)
        # x = x + mlp.gate.unsqueeze(1) * self.mlp(modulate(self.norm2(x).to(x.dtype), mlp.shift, mlp.scale))
        x = x + mlp.gate * self.mlp(mlp.modulate(self.norm2(x)))
        return x

class DoubleStreamBlock(nn.Module):
    """
    A DiT block with seperate MoE for text & image
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        num_experts=8,
        capacity_factor=2.0,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1,
        exp_ratio: int = 4,
        use_moe: bool = False,
        use_expert_choice: bool = False,
        shared_mod=False,
        shared_attn_projs=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        if not shared_mod:
            self.img_mod = Modulation(hidden_size, double=True)
            self.txt_mod = Modulation(hidden_size, double=True)
        
        self.shared_attn_projs = shared_attn_projs
        if not shared_attn_projs:
            self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, dropout=dropout)
            self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, dropout=dropout)

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        text_exps = max(1, num_experts // exp_ratio)

        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size, bias=True),
        )
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        img: Tensor,          # [B, L_img, hidden_size]
        txt: Tensor,          # [B, L_txt, hidden_size]
        vec: Tensor,          # conditioning vector => Modulation
        pe: Tensor,    # rope positional encoding
        mod=None,
        img_attn=None,
        txt_attn=None,
    ) -> tuple[Tensor, Tensor]:
        dtype = img.dtype
        if mod is not None:
            img_mod1, img_mod2 = mod(vec)
            txt_mod1, txt_mod2 = mod(vec)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        if self.shared_attn_projs:
            img_attn = img_attn
            txt_attn = txt_attn
        else:
            img_attn = self.img_attn
            txt_attn = self.txt_attn

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = img_mod1.modulate(img_modulated)
        img_qkv = img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = txt_mod1.modulate(txt_modulated)
        txt_qkv = txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn_out, img_attn_out = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * img_attn.proj(img_attn_out)
        img = img + img_mod2.gate * self.img_mlp((img_mod2.modulate(self.img_norm2(img))).to(dtype))

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * txt_attn.proj(txt_attn_out)
        txt = txt + txt_mod2.gate * self.txt_mlp((txt_mod2.modulate(self.txt_norm2(txt))).to(dtype))
        
        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        num_experts=8,
        capacity_factor=2.0,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1,
        use_moe: bool = False,
        use_expert_choice: bool = False,
        shared_mod=False,
        shared_attn_projs=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        self.dropout = dropout

        # qkv and mlp_in
        self.shared_attn_projs = shared_attn_projs

        if not shared_attn_projs:
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3)


        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size, bias=True),
        )

        # proj and mlp_out
        if not shared_attn_projs:
            self.linear2 = nn.Linear(2 * hidden_size, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")

        if not shared_mod:
            self.modulation = Modulation(hidden_size, double=False)


    def forward(
        self,
        x: Tensor,   # [B, L_img + L_txt, hidden_size]
        vec: Tensor,   # conditioning vector => for Modulation
        pe: Tensor,    # rope positional encoding
        mod=None,
        linear1=None,
        linear2=None,
    ) -> tuple[Tensor, Tensor]:
        if mod is not None:
            mod1, _ = mod(vec)
        else:
            mod1, _ = self.mod(vec)

        if self.shared_attn_projs:
            linear1 = linear1
            linear2 = linear2
        else:
            linear1 = self.linear1
            linear2 = self.linear2

        x_mod = mod1.modulate(self.pre_norm(x))
        qkv = linear1(x_mod)
        mlp_out = self.mlp(x_mod)
        # qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        qkv = qkv.contiguous()
        mlp_out = mlp_out.contiguous()

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, dropout=(self.dropout if self.training else 0.0))
        # compute activation in mlp stream, cat again and run second linear layer
        output = linear2(torch.cat((attn, self.mlp_act(mlp_out)), 2))
        return x + mod1.gate * output



@dataclass
class BackboneParams:
    use_mmdit: bool = True
    use_ec: bool = False
    use_moe: bool = False
    shared_mod: bool = False
    shared_attn_projs: bool = False
    embed_dim: int = 1152
    num_layers: int = 24
    num_heads: int = 1152 // 64
    num_experts: int = 4
    capacity_factor: float = 2.0
    shared_experts: int = 2
    dropout: float = 0.1
    image_text_expert_ratio: int = 4
    pretraining_tp: int = 1

def nearest_divisor(scaled_num_heads, embed_dim):
    # Find all divisors of embed_dim
    divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
    
    # Find the nearest divisor
    nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))
    
    return nearest

class TransformerBackbone(nn.Module):
    def __init__(self, params: BackboneParams):
        super().__init__()
        mf_min, mf_max = 1.0, 3.0
        self.use_mmdit = params.use_mmdit
        self.shared_mod = params.shared_mod
        if self.shared_mod:
            self.mod = Modulation(params.embed_dim, True)
        
        self.shared_attn_projs = params.shared_attn_projs
        if self.shared_attn_projs:
            # if self.use_mmdit:
            #     self.img_attn = SelfAttention(params.embed_dim, params.num_heads, dropout=params.dropout)
            #     self.txt_attn = SelfAttention(params.embed_dim, params.num_heads, dropout=params.dropout)
            
            self.attn = SelfAttention(params.embed_dim, params.num_heads, dropout=params.dropout)

        self.double_layers = nn.ModuleList()
        self.layers = nn.ModuleList()

        for i in range(params.num_layers):
            # Calculate scaling factors for the i-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * i / (params.num_layers - 1)

            # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = (int(params.embed_dim * mf) // params.num_heads) * params.num_heads
            scaled_num_heads = params.num_heads
            mlp_dim = max(params.embed_dim, scaled_mlp_dim)

            if i % 2 == 0:  # Even layers use regular DiT (no MoE)
                n_exp = 1
                n_shared = None
                n_act = 1.0
            else:  # Odd layers use MoE DiT
                n_exp = params.num_experts
                n_shared = params.shared_experts
                n_act = min(params.capacity_factor, float(n_exp))

            if params.use_mmdit:
                if i < params.num_layers // 6: # First sixth uses DoubleStreamBlock
                    self.double_layers.append(DoubleStreamBlock(
                        hidden_size=params.embed_dim,
                        num_heads=scaled_num_heads,
                        mlp_dim=mlp_dim,
                        num_experts=n_exp,
                        capacity_factor=n_act,
                        pretraining_tp=params.pretraining_tp,
                        num_shared_experts=n_shared,
                        dropout=params.dropout,
                        exp_ratio=params.image_text_expert_ratio,
                        use_moe=params.use_moe,
                        use_expert_choice=params.use_ec,
                        shared_mod=params.shared_mod,
                        shared_attn_projs=params.shared_attn_projs
                    ))
                else:  # Rest use SingleStreamBlock
                    # self.layers.append(SingleStreamBlock(
                    #     hidden_size=params.embed_dim,
                    #     num_heads=scaled_num_heads,
                    #     mlp_dim=mlp_dim,
                    #     num_experts=n_exp,
                    #     capacity_factor=n_act,
                    #     pretraining_tp=params.pretraining_tp,
                    #     num_shared_experts=n_shared,
                    #     dropout=params.dropout,
                    #     use_moe=params.use_moe,
                    #     use_expert_choice=params.use_ec,
                    #     shared_mod=params.shared_mod,
                    #     shared_attn_projs=params.shared_attn_projs
                    # ))
                    self.layers.append(DiTBlock(
                        hidden_size=params.embed_dim,
                        num_heads=scaled_num_heads,
                        mlp_dim=mlp_dim,
                        num_experts=n_exp,
                        num_experts_per_tok=n_act,
                        pretraining_tp=params.pretraining_tp,
                        num_shared_experts=n_shared,
                        dropout=params.dropout,
                        use_moe=params.use_moe,
                        use_expert_choice=params.use_ec,
                        shared_mod=params.shared_mod,
                        shared_attn_projs=params.shared_attn_projs
                    ))
            else:
                self.layers.append(DiTBlock(
                    hidden_size=params.embed_dim,
                    num_heads=scaled_num_heads,
                    mlp_dim=mlp_dim,
                    num_experts=n_exp,
                    num_experts_per_tok=n_act,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=n_shared,
                    dropout=params.dropout,
                    use_moe=params.use_moe,
                    use_expert_choice=params.use_ec,
                    shared_mod=params.shared_mod,
                    shared_attn_projs=params.shared_attn_projs
                ))

    def forward(
            self, 
            img: torch.Tensor,
            txt: torch.Tensor,
            vec: torch.Tensor,
            pe: torch.Tensor,
            mod = None,
            img_attn = None,
            txt_attn = None,
            attn = None,
            ):
        if self.shared_mod:
            mod = self.mod

        if self.shared_attn_projs:
            # if self.use_mmdit:
            #     img_attn = self.img_attn
            #     txt_attn = self.txt_attn
            attn = self.attn

        for layer in self.double_layers:
            img, txt = layer(img, txt, vec, pe, mod=mod, img_attn=img_attn, txt_attn=txt_attn)

        img = torch.cat((txt, img), 1)

        for layer in self.layers:
            img = layer(img, vec, pe, mod=mod, attn=attn)
        
        img = img[:, txt.shape[1]:, ...]

        return img



@dataclass
class ReiMeiParameters:
    use_mmdit: bool = True
    use_ec: bool = False
    use_moe: bool = False
    shared_mod: bool = False
    shared_attn_projs: bool = False,
    channels: int = 32
    patch_size: tuple[int, int] = (1,1)
    embed_dim: int = 1152
    num_layers: int = 24
    num_heads: int = 1152 // 64
    num_experts: int = 4
    capacity_factor: float = 2.0
    shared_experts: int = 2
    dropout: float = 0.1
    token_mixer_layers: int = 0
    image_text_expert_ratio: int = 4
    # m_d: float = 1.0

class ReiMei(nn.Module):
    """
    ReiMei is a image diffusion transformer model.

        Args:
        channels (int): Number of input channels in the image data.
        patch_size (Tuple[int, int]): Size of the patch.
        embed_dim (int): Dimension of the embedding space.
        num_layers (int): Number of layers in the transformer backbone.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        mlp_dim (int): Dimension of the multi-layer perceptron.
        text_embed_dim (int): Dimension of the text embedding.
        vector_embed_dim (int): Dimension of the vector embedding.
        num_experts (int, optional): Number of experts in the transformer backbone. Default is 4.
        capacity_factor (float, optional): Average number of experts per token. Default is 2.0.
        shared_experts (int, optional): Number of shared experts in the transformer backbone. Default is 2.
        dropout (float, optional): Dropout rate. Default is 0.1.
        patch_mixer_layers (int, optional): Number of layers in the patch mixer. Default is 2.

    Attributes:
        embed_dim (int): Dimension of the embedding space.
        channels (int): Number of input channels in the image data.
        time_embedder (TimestepEmbedder): Timestep embedding layer.
        image_embedder (MLPEmbedder): Image embedding layer.
        text_embedder (MLPEmbedder): Text embedding layer.
        vector_embedder (MLPEmbedder): Vector embedding layer.
        token_mixer (TokenMixer): Token mixer layer.
        backbone (TransformerBackbone): Transformer backbone model.
        output (MLPEmbedder): Output layer.
    """
    def __init__(self, params: ReiMeiParameters):
        super().__init__()
        self.params = params
        self.embed_dim = params.embed_dim
        self.head_dim = params.embed_dim // params.num_heads
        self.channels = params.channels
        self.patch_size = params.patch_size
        self.use_mmdit = params.use_mmdit

        self.params_embedder = MLPEmbedder(27, self.embed_dim*3, hidden_dim=self.embed_dim, num_layers=4)
        self.params_pos_encoder = nn.Linear(27, 6)
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(self.embed_dim)

        # Image embedding
        self.image_embedder = PatchEmbed(self.channels, self.embed_dim, self.patch_size)

        self.rope_embedder = EmbedND(dim=self.head_dim)

        # TokenMixer
        if params.token_mixer_layers > 0:
            self.use_token_mixer = True
            token_mixer_params = TokenMixerParameters(
                use_mmdit=params.use_mmdit,
                use_ec=params.use_ec,
                # use_moe=params.use_moe,
                shared_mod=params.shared_mod,
                use_moe=False,
                embed_dim=self.embed_dim,
                num_heads=params.num_heads,
                num_layers=params.token_mixer_layers,
                num_experts=params.num_experts,
                capacity_factor=params.capacity_factor,
                num_shared_experts=params.shared_experts,
                exp_ratio=params.image_text_expert_ratio,
                dropout=params.dropout,
            )
            self.token_mixer = TokenMixer(token_mixer_params)
        else:
            self.use_token_mixer = False

        # Backbone transformer model
        backbone_params = BackboneParams(
            use_mmdit=params.use_mmdit,
            use_ec=params.use_ec,
            use_moe=params.use_moe,
            shared_mod=params.shared_mod,
            shared_attn_projs=params.shared_attn_projs,
            embed_dim=self.embed_dim,
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            num_experts=params.num_experts,
            capacity_factor=params.capacity_factor,
            shared_experts=params.shared_experts,
            dropout=params.dropout,
            image_text_expert_ratio=params.image_text_expert_ratio,
        )
        self.backbone = TransformerBackbone(backbone_params)
        
        self.output_layer = OutputLayer(self.embed_dim, self.channels * self.patch_size[0] * self.patch_size[1])

        self.initialize_weights()

    def initialize_weights(self):
        s = 1.0 / math.sqrt(self.embed_dim)

        # # Initialize all linear layers and biases
        # def _basic_init(module):
        #     if isinstance(module, nn.LayerNorm):
        #         if module.weight is not None:
        #             nn.init.constant_(module.weight, 1.0)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.Linear):
        #         nn.init.normal_(module.weight, std=s)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # # Initialize all linear layers and biases
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.Conv2d):
        #         # Initialize convolutional layers like linear layers
        #         nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.LayerNorm):
        #         # Initialize LayerNorm layers
        #         if module.weight is not None:
        #             nn.init.constant_(module.weight, 1.0)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)


        # # Apply basic initialization to all modules
        # self.apply(_basic_init)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output_layer.mlp.mlp[-1].weight, 0)
        nn.init.constant_(self.output_layer.mlp.mlp[-1].bias, 0)



    def load_weights(model: 'ReiMei', 
                     weights_path: str, 
                     map_location: Optional[Union[str, torch.device]] = None,
                     strict: bool = True) -> 'ReiMei':
        """
        Load weights for a ReiMei model from a saved state dictionary.
        
        Args:
            model (ReiMei): The ReiMei model instance to load weights into
            weights_path (str): Path to the saved weights file (.pt or .pth)
            map_location (str or torch.device, optional): Device to map the weights to.
                                                         Default is None (load to original device)
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict
                                    match the keys in model's state_dict. Default is True
                                    
        Returns:
            ReiMei: The model with loaded weights
            
        Raises:
            FileNotFoundError: If the weights file doesn't exist
            RuntimeError: If there's an issue loading the state dict with strict=True
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        
        # Load the state dictionary
        state_dict = torch.load(weights_path, map_location=map_location)
        
        # If the state dict contains more than just the model weights (e.g., optimizer state)
        # extract just the model weights
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Handle case where keys have 'module.' prefix (from DataParallel/DistributedDataParallel)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Load the weights
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            
            if not strict and (missing_keys or unexpected_keys):
                print(f"Info: Non-strict weight loading")
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
                    
        except RuntimeError as e:
            print(f"Error loading state dictionary: {e}")
            raise
        
        return model


# Example usage:
# reimei_model = ReiMei(params)
# reimei_model = load_weights(reimei_model, 'path/to/weights.pt')

    def forward(self, img, time, params):
        # img: (batch_size, channels, height, width)
        # time: (batch_size, 1)
        # params: (batch_size, 27) (27 = 3 (Background RGB), 8 (alphabet index, x1, x2, y1, y2, RGB)*3 (3 letters in each image))
        # mask: (batch_size, num_tokens)
        batch_size, channels, height, width = img.shape
        ps_h, ps_w = self.patch_size
        patched_h, patched_w = height // ps_h, width // ps_w

        params = (params / 64.0)

        # Alphabet
        # (batch_size, 27) -> (batch_size, 3, 2)
        pos_ids = self.params_pos_encoder(params).reshape(batch_size, 2, 3)
        # (batch_size, 27) -> (batch_size, 3, embed_dim)
        params = self.params_embedder(params).reshape(batch_size, 3, self.embed_dim)

        sincos_pos = torch.stack([sincos_2d_from_grid(self.embed_dim, pos_ids[i]) for i in range(batch_size)])
        params += sincos_pos

        # (batch_size, embed_dim*4) -> (batch_size, 16, embed_dim)
        # params = params.reshape(batch_size, 16, self.embed_dim)

        # Vector embedding (timestep + vector_embeddings)
        vec = self.time_embedder(time)

        # Image embedding
        img = self.image_embedder(img)

        rope_id = rope_ids(batch_size, patched_h, patched_w, params.shape[1], img.device, img.dtype)
        rope_pe = self.rope_embedder(torch.cat(rope_id, dim=1))

        # Token-mixer
        if self.use_token_mixer:
            img, params = self.token_mixer(img, params, vec, rope_pe)

        # Backbone transformer model
        img = self.backbone(img, params, vec, rope_pe)

        # Final output layer
        # (bs, unmasked_num_tokens, embed_dim) -> (bs, unmasked_num_tokens, in_channels)
        img = self.output_layer(img, vec)

        img = unpatchify(img, self.patch_size, height, width)
        
        return img
    
    @torch.no_grad()
    def sample(self, z, params, sample_steps=50, cfg=3.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device, torch.bfloat16).view([b, *([1] * len(z.shape[1:]))])


        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device, torch.bfloat16)

            vc = self(z, t, params).to(torch.bfloat16)
            if cfg != 1.0:
                neg_params = torch.zeros_like(params)
                vu = self(z, t, neg_params)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc

        return (z / 2.0) + 0.5 



def unpatchify(x, patch_size, height, width):
    """
    Reconstructs images from patches.

    Args:
        x (torch.Tensor): Tensor of shape (bs, num_patches, patch_size * patch_size * in_channels)
        patch_size (int): Size of each patch.
        height (int): Original image height.
        width (int): Original image width.

    Returns:
        torch.Tensor: Reconstructed image of shape (bs, in_channels, height, width)
    """
    bs, num_patches, patch_dim = x.shape
    H, W = patch_size
    in_channels = patch_dim // (H * W)

    # Calculate the number of patches along each dimension
    num_patches_h = height // H
    num_patches_w = width // W

    # Ensure num_patches equals num_patches_h * num_patches_w
    assert num_patches == num_patches_h * num_patches_w, "Mismatch in number of patches."

    # Reshape x to (bs, num_patches_h, num_patches_w, H, W, in_channels)
    x = x.view(bs, num_patches_h, num_patches_w, H, W, in_channels)

    # Permute x to (bs, num_patches_h, H, num_patches_w, W, in_channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape x to (bs, height, width, in_channels)
    reconstructed = x.view(bs, height, width, in_channels)

    # Permute back to (bs, in_channels, height, width)
    reconstructed = reconstructed.permute(0, 3, 1, 2).contiguous()

    return reconstructed

def random_cfg_mask(bs, x):
    """Create a tensor of shape (bs,) with (x*100)% zeros and the rest ones."""
    num_zeros = int(bs * x)  # Calculate number of zeros
    num_ones = bs - num_zeros  # Remaining values are ones

    # Create tensor with appropriate number of zeros and ones
    tensor = torch.cat([torch.zeros(num_zeros), torch.ones(num_ones)])

    # Shuffle to randomize the order
    tensor = tensor[torch.randperm(bs)]

    return tensor



TRAIN_STEPS = 100000
LR = 0.0001
DTYPE = torch.bfloat16

@torch.no_grad
def sample_images(model, noise, params, sample_steps=50, cfg=5.0):
    fifty_sampled_images = model.sample(noise, params, sample_steps=sample_steps, cfg=cfg).to(device, dtype=DTYPE)

    grid = torchvision.utils.make_grid(fifty_sampled_images, nrow=4, normalize=False)

    return grid


if __name__ == "__main__":
    # Comment this out if you havent downloaded dataset and models yet
    # datasets.config.HF_HUB_OFFLINE = 1
    # torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "true"

    embed_dim = 768
    patch_size = (4,4)

    params = ReiMeiParameters(
        use_mmdit=True,
        use_ec=True,
        use_moe=None,
        shared_mod=True,
        shared_attn_projs=True,
        channels=3,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=4,
        num_heads=(embed_dim // 128),
        num_experts=4,
        capacity_factor=2.0,
        shared_experts=1,
        dropout=0.1,
        token_mixer_layers=2,
        image_text_expert_ratio=2,
    )

    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    device = accelerator.device

    model = ReiMei(params).to(device, dtype=DTYPE)
    # model = torch.compile(ReiMei(params))

    params_count = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", params_count)

    # Create a dataset
    dataset = ShapeDataset(
        length=100_000,        # Number of images
        image_size=64,     # Image size (square)
        max_shapes=3,       # Maximum shapes per image
        seed=42,            # Random seed for reproducibility
        nocolor=True,      # White Background
        download_url="https://github.com/fal-ai-community/alphabet-dataset/raw/refs/heads/main/contest_param/2025contest_trainsetparams.pt"
    )

    ds = DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=LR, weight_decay=0.1)

    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=TRAIN_STEPS)

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, ds)

    # del checkpoint
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    noise = torch.randn(16, 3, 64, 64).to(device, dtype=DTYPE)
    # noise = torch.ones(16, 3, 64, 64).to(device, dtype=DTYPE) * 0.9725
    example_batch = next(iter(ds))

    example_images, example_params = example_batch
    example_params = example_params.to(device=device, dtype=DTYPE)[:16]

    grid = torchvision.utils.make_grid(example_images[:16], nrow=4, normalize=True, scale_each=True)
    torchvision.utils.save_image(grid, f"logs/example_images.png")

    del grid, example_images

    print("Starting training...")
    progress_bar = tqdm(leave=False, total=TRAIN_STEPS)
    i = 0
    grads = None
    while i < TRAIN_STEPS:
        if i == TRAIN_STEPS:
            break
        for batch in train_dataloader:
            images, image_params = batch

            images = ((images - 0.5) * 2.0).to(device=device, dtype=DTYPE)
            image_params = image_params.to(device=device, dtype=DTYPE)
            bs = images.shape[0]

            # cfg_mask = random_cfg_mask(bs, CFG_RATIO).to(device, dtype=DTYPE)
            z = torch.randn_like(images).to(device, dtype=DTYPE)
            # z = torch.ones_like(images).to(device, dtype=DTYPE) * 0.9725

            # nt = torch.randn((bs,), device=device, dtype=DTYPE)
            # t = torch.sigmoid(nt)
            t = torch.ones((bs,), device=device, dtype=DTYPE)
            # t = torch.nn.functional.sigmoid((nt*2+0.4)*1.02).clip(0,1)

            # texp = t.view([bs, 1, 1, 1]).to(device, dtype=DTYPE)

            # x_t = (1 - texp) * images + texp * z

            vtheta = model(z, t, image_params)

            v = z - images

            mse = (((v - vtheta) ** 2)).mean(dim=(1,2))

            loss = mse.mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            grads = gradfilter_ema(model, grads=grads)

            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)

            del mse, loss, v, vtheta

            if i % 500 == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    with torch.no_grad():
                        model.eval()

                        grid = sample_images(model, noise, example_params, sample_steps=1, cfg=1.0)
                        torchvision.utils.save_image(grid, f"logs/sampled_images_step_{i}.png")

                        del grid

                        model.train()

            if ((i % (TRAIN_STEPS//5)) == 0) and i != 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                    model_save_path = f"models/reimei_model_and_optimizer_{i//(TRAIN_STEPS//5)}_f32.pt"
                    torch.save({
                        'global_step': i,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, model_save_path)
                    print(f"Model saved to {model_save_path}.")
            
            if i == TRAIN_STEPS - 1:
                print("Training complete.")

                # Save model in /models
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                    model_save_path = "models/pretrained_reimei_model_and_optimizer.pt"
                    torch.save(
                        {
                            'global_step': i,
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        },
                        model_save_path,
                    )
                    print(f"Model saved to {model_save_path}.")

            progress_bar.update(1)
            i += 1
