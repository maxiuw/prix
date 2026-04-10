
"""
Single-image TransFuser backbone
输出:
    p3 : (B,  64, 64, 64)  – FPN 浅层特征
    c5 : (B, C5,  8, 32)   – 深层语义特征 (C5，通道自动读取, ResNet-34 为 512)
    None: 占位
"""

import math
import timm
import torch
import torch.nn.functional as F
from torch import nn

# ───────────────────── Transformer 基元 ─────────────────────
class _SelfAttn(nn.Module):
    def __init__(self, dim, n_head=4, pdrop=0.1):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = self.attn_drop(att.softmax(dim=-1))
        y   = att @ v
        y   = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class _TransformerBlock(nn.Module):
    def __init__(self, dim, n_head=4, mlp_ratio=4, pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _SelfAttn(dim, n_head, pdrop)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.ReLU(True),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """自注意力重标定 (B,C,8,32) → (B,C,8,32)"""
    def __init__(self, dim, n_layer=2):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 256, dim))
        self.drop    = nn.Dropout(0.1)
        self.blocks  = nn.Sequential(*[_TransformerBlock(dim) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(dim)

    def forward(self, feat):
        B, C, H, W = feat.shape                # H=8, W=32
        tok = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x   = self.ln_f(self.blocks(self.drop(tok + self.pos_emb)))
        x   = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

# ───────────────────── Backbone ─────────────────────
class TransfuserBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # ── ResNet-34，强制 output_stride=32 并加载自定义权重 ──
        self.encoder = timm.create_model(
            'resnet34',
            pretrained=True,                    # 打开 timm 的权重加载逻辑
            features_only=True,
            out_indices=(1, 2, 3, 4),
            output_stride=32,                   # 确保 C5 → 8×32
            pretrained_cfg_overlay=dict(
                file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/pytorch_model.bin"
            )
        )


        # 自动读取 layer1-4 通道
        ch_list = [fi["num_chs"] for fi in self.encoder.feature_info.info[1:5]]
        # e.g. [64, 128, 256, 512] for ResNet-34

        self.avgpool = nn.AdaptiveAvgPool2d((8, 32))
        self.gpts = nn.ModuleList([TinyGPT(c) for c in ch_list])
        self.channel_proj = nn.ModuleList([nn.Identity() for _ in ch_list])

        # FPN 顶-下: C5 → P3
        self.c5_to_64 = nn.Conv2d(ch_list[-1], 64, 1)
        self.relu = nn.ReLU(True)
        self.up2x   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2fix = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)
        self.up_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.up_conv2 = nn.Conv2d(64, 64, 3, padding=1)

    # helper: 跑到下一个 return layer
    @staticmethod
    def _run_to_next_return(it_layers, return_layers, x):
        for name, mod in it_layers:
            x = mod(x)
            if name in return_layers:
                break
        return x

    def _top_down(self, c5):
        p5 = self.relu(self.c5_to_64(c5))         # (B,64, 8,32)
        p4 = self.relu(self.up_conv1(self.up2x(p5)))   # (B,64,16,64)
        p3 = self.relu(self.up_conv2(self.up2fix(p4))) # (B,64,64,64)
        return p3

    # ── forward ──
    def forward(self, img, *_):
        # print("extracting features foc camera only ")
        feat = img
        it_layers = iter(self.encoder.items())

        for i in range(4):
            feat = self._run_to_next_return(it_layers, self.encoder.return_layers, feat)
            emb  = self.avgpool(feat)            # (B,C,8,32)
            new  = self.gpts[i](emb)
            new  = self.channel_proj[i](new)
            feat = feat + F.interpolate(new, size=feat.shape[2:], mode="bilinear", align_corners=False)

        c5 = feat                                # (B,C5,8,32)
        p3 = self._top_down(c5)                  # (B,64,64,64)
        return p3, c5, None


# ───────────────────────── quick test ─────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransfuserBackbone().to(device).eval()
    dummy  = torch.randn(1, 3, 256, 1024, device=device)

    with torch.no_grad():
        p3, c5, _ = model(dummy)

    print("P3 :", p3.shape)  # torch.Size([1, 64, 64, 64])
    print("C5 :", c5.shape)  # torch.Size([1, 512, 8, 32])  (C5 通道随骨干而变)
