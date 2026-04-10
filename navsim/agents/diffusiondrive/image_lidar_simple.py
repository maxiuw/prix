######################## minimal TransFuser (Image + LiDAR) 
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Transformer utilities with size debugging
# ---------------------------------------------------------------------------
class _SelfAttention(nn.Module):
    """ Multi‑head self‑attention block (no masking) with size debugging """

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float, debug=True):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dim must be divisible by n_head"
        self.n_head = n_head
        self.debug = debug

        self.key   = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop  = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj       = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"    SelfAttention输入: {x.shape}")
        
        B, T, C = x.shape
        
        # 计算 K, Q, V
        k_raw = self.key(x)
        q_raw = self.query(x)
        v_raw = self.value(x)
        
        if self.debug:
            print(f"    K/Q/V原始输出: {k_raw.shape}")
        
        k = k_raw.contiguous() \
                .reshape(B, T, self.n_head, C // self.n_head) \
                .transpose(1, 2)                       # (B, nh, T, hs)

        q = q_raw.contiguous() \
                .reshape(B, T, self.n_head, C // self.n_head) \
                .transpose(1, 2)

        v = v_raw.contiguous() \
                .reshape(B, T, self.n_head, C // self.n_head) \
                .transpose(1, 2)

        if self.debug:
            print(f"    K reshape后: {k.shape} (B, n_head, T, head_size)")
            print(f"    Q reshape后: {q.shape}")
            print(f"    V reshape后: {v.shape}")

        # 注意力计算
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.debug:
            print(f"    注意力分数: {att.shape} (B, n_head, T, T)")
        
        att = self.attn_drop(torch.softmax(att, dim=-1))
        if self.debug:
            print(f"    softmax后: {att.shape}")
        
        y = att @ v                          # (B, nh, T, hs)
        if self.debug:
            print(f"    注意力×V: {y.shape}")
        
        y = y.transpose(1, 2).reshape(B, T, C)
        if self.debug:
            print(f"    转置重塑后: {y.shape}")
        
        output = self.resid_drop(self.proj(y))
        if self.debug:
            print(f"    SelfAttention输出: {output.shape}")
            print()
        
        return output


class _TransformerBlock(nn.Module):
    """ A single Transformer encoder block with size debugging """

    def __init__(self, n_embd: int, n_head: int = 4, mlp_ratio: int = 4, pdrop: float = 0.1, debug=True):
        super().__init__()
        self.debug = debug
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = _SelfAttention(n_embd, n_head, pdrop, pdrop, debug)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.ReLU(True),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"  TransformerBlock输入: {x.shape}")
        
        # 第一个残差连接: LayerNorm + SelfAttention
        ln1_out = self.ln1(x)
        if self.debug:
            print(f"  LayerNorm1输出: {ln1_out.shape}")
        
        attn_out = self.attn(ln1_out)
        x_after_attn = x + attn_out
        if self.debug:
            print(f"  残差连接1后: {x_after_attn.shape}")
        
        # 第二个残差连接: LayerNorm + MLP
        ln2_out = self.ln2(x_after_attn)
        if self.debug:
            print(f"  LayerNorm2输出: {ln2_out.shape}")
        
        mlp_out = self.mlp(ln2_out)
        if self.debug:
            print(f"  MLP输出: {mlp_out.shape}")
        
        x_final = x_after_attn + mlp_out
        if self.debug:
            print(f"  TransformerBlock最终输出: {x_final.shape}")
            print("  " + "="*50)
        
        return x_final


class TinyGPT(nn.Module):
    """Token mixer: fuse two 8×32 grids (256 token each) via small Transformer."""

    def __init__(self, n_embd: int, num_layers: int = 2, debug=True):
        super().__init__()
        self.debug = debug
        self.num_layers = num_layers
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, n_embd))
        self.drop    = nn.Dropout(0.1)
        self.blocks  = nn.Sequential(*[_TransformerBlock(n_embd, debug=debug) for _ in range(num_layers)])
        self.ln_f    = nn.LayerNorm(n_embd)

    def forward(self, feat_img: torch.Tensor, feat_lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both features must be (B, C, 8, 32) with same C."""
        if self.debug:
            print("="*60)
            print("TinyGPT 前向传播开始")
            print("="*60)
        
        B, C, H, W = feat_img.shape  # H=8, W=32
        if self.debug:
            print(f"输入特征图尺寸: feat_img={feat_img.shape}, feat_lidar={feat_lidar.shape}")
        
        # 特征图转token
        tok_img   = feat_img.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tok_lidar = feat_lidar.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        if self.debug:
            print(f"转换为token后:")
            print(f"  tok_img: {tok_img.shape}")
            print(f"  tok_lidar: {tok_lidar.shape}")

        # 拼接两个模态
        x = torch.cat([tok_img, tok_lidar], dim=1)  # (B, 512, C)
        if self.debug:
            print(f"拼接后: {x.shape}")
        
        # 添加位置编码
        x_with_pos = x + self.pos_emb
        if self.debug:
            print(f"添加位置编码后: {x_with_pos.shape}")
        
        # Dropout
        x_dropped = self.drop(x_with_pos)
        if self.debug:
            print(f"Dropout后: {x_dropped.shape}")
            print()

        # 通过Transformer层
        if self.debug:
            print(f"开始通过 {self.num_layers} 个Transformer层:")
        
        x_transformed = self.blocks(x_dropped)
        
        if self.debug:
            print(f"所有Transformer层输出: {x_transformed.shape}")
        
        # 最终LayerNorm
        x_final = self.ln_f(x_transformed)
        if self.debug:
            print(f"最终LayerNorm后: {x_final.shape}")

        # 分离两个模态
        tok_img_out, tok_lidar_out = x_final.split(H * W, dim=1)
        if self.debug:
            print(f"分离后:")
            print(f"  tok_img_out: {tok_img_out.shape}")
            print(f"  tok_lidar_out: {tok_lidar_out.shape}")
        
        # 转换回特征图格式
        img_out   = tok_img_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        lidar_out = tok_lidar_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        if self.debug:
            print(f"转换回特征图格式:")
            print(f"  img_out: {img_out.shape}")
            print(f"  lidar_out: {lidar_out.shape}")
            print("="*60)
            print("TinyGPT 前向传播结束")
            print("="*60)
            print()
        
        return img_out, lidar_out


# ---------------------------------------------------------------------------
# Backbone – minimal TransFuser (Image + LiDAR) with debugging
# ---------------------------------------------------------------------------
class TransfuserBackbone(nn.Module):
    """Takes RGB image & LiDAR BEV, outputs FPN-like P3 and global C5."""

    def __init__(self, debug=True):
        super().__init__()
        self.debug = debug
        embed_dim = 256

        # 1×1 convs to project raw inputs into common embed_dim
        self.img_proj   = nn.Conv2d(3,   embed_dim, kernel_size=1)
        self.lidar_proj = nn.Conv2d(1,   embed_dim, kernel_size=1)

        # Average‑pool to 8×32 grid
        self.pool_img   = nn.AdaptiveAvgPool2d((8, 32))
        self.pool_lidar = nn.AdaptiveAvgPool2d((8, 32))

        # TinyGPT fusion
        self.fuser = TinyGPT(embed_dim, debug=debug)

        # C5 → P3 top‑down
        self.top_down = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 1),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
        )

    def forward(self, img, lidar):
        """Forward pass with detailed size debugging
        Parameters
        ----------
        img   : (B,3,256,1024)
        lidar : (B,1,256,256)
        Returns (P3, C5, None)
        """
        if self.debug:
            print("="*80)
            print("TRANSFUSER BACKBONE 前向传播开始")
            print("="*80)
        
        assert lidar is not None, "LiDAR input must be provided"

        if self.debug:
            print("1. 原始输入:")
            print(f"   img: {img.shape}")
            print(f"   lidar: {lidar.shape}")
            print()

        # 投影到共同嵌入维度
        img_proj = self.img_proj(img)
        lidar_proj = self.lidar_proj(lidar)
        
        if self.debug:
            print("2. 1×1卷积投影后:")
            print(f"   img_proj: {img_proj.shape}")
            print(f"   lidar_proj: {lidar_proj.shape}")
            print()

        # 池化到统一尺寸
        img_emb   = self.pool_img(img_proj)        # (B,embed,8,32)
        lidar_emb = self.pool_lidar(lidar_proj)    # (B,embed,8,32)
        
        if self.debug:
            print("3. 自适应池化后:")
            print(f"   img_emb: {img_emb.shape}")
            print(f"   lidar_emb: {lidar_emb.shape}")
            print()

        # TinyGPT融合
        if self.debug:
            print("4. 开始TinyGPT融合:")
        
        img_fused, lidar_fused = self.fuser(img_emb, lidar_emb)
        
        if self.debug:
            print("5. TinyGPT融合完成:")
            print(f"   img_fused: {img_fused.shape}")
            print(f"   lidar_fused: {lidar_fused.shape}")
            print()

        # 选择图像分支作为C5
        c5 = img_fused                                   # (B,embed,8,32)
        if self.debug:
            print("6. 选择C5特征:")
            print(f"   c5: {c5.shape}")
        
        # 上采样生成P3
        p3 = self.top_down(c5)                           # (B,64,64,64)
        if self.debug:
            print("7. 上采样生成P3:")
            print(f"   p3: {p3.shape}")
            print()
            print("="*80)
            print("TRANSFUSER BACKBONE 前向传播完成")
            print("="*80)
            print()
        
        return p3, c5, None


# ---------------------------------------------------------------------------
# Quick self‑test with full debugging
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("创建模型...")
    model = TransfuserBackbone(debug=True).cuda()
    
    print("创建输入数据...")
    img   = torch.randn(1, 3, 256, 1024).cuda()
    lidar = torch.randn(1, 1, 256, 256).cuda()
    
    print("开始前向传播...\n")
    p3, c5, _ = model(img, lidar)
    
    print("="*80)
    print("最终结果:")
    print(f"P3 shape: {p3.shape}   # 期望: (1, 64, 64, 64)")
    print(f"C5 shape: {c5.shape}   # 期望: (1, 256, 8, 32)")
    print("="*80)
