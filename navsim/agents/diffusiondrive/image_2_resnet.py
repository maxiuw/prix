
"""
Image-Image TransFuser backbone (no LiDAR branch)
"""
import math
import timm
import torch
import torch.nn.functional as F
from torch import nn


# ─────────── Transformer block ───────────
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key   = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop  = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1/math.sqrt(k.size(-1)))
        att = self.attn_drop(torch.softmax(att, dim=-1))
        y   = att @ v
        y   = y.transpose(1, 2).reshape(B, T, C)      # (B,T,C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head=4, mlp_ratio=4, pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, pdrop, pdrop)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio*n_embd),
            nn.ReLU(True),
            nn.Linear(mlp_ratio*n_embd, n_embd),
            nn.Dropout(pdrop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Fusion token mixer: 8×32 + 8×32 = 512 tokens"""
    def __init__(self, n_embd, num_layers=2):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, n_embd))  # learnable
        self.drop    = nn.Dropout(0.1)
        self.blocks  = nn.Sequential(*[TransformerBlock(n_embd) for _ in range(num_layers)])
        self.ln_f    = nn.LayerNorm(n_embd)

    def forward(self, featA, featB):          # feats: (B,C,8,32)
        B, C, H, W = featA.shape              # H=8,W=32
        tokA = featA.permute(0,2,3,1).reshape(B, H*W, C)
        tokB = featB.permute(0,2,3,1).reshape(B, H*W, C)
        x = torch.cat([tokA, tokB], dim=1)    # (B,512,C)
        x = self.ln_f(self.blocks(self.drop(x + self.pos_emb)))
        tokA_out, tokB_out = x.split(H*W, dim=1)  # each (B,256,C)
        tokA_out = tokA_out.reshape(B,H,W,C).permute(0,3,1,2).contiguous()
        tokB_out = tokB_out.reshape(B,H,W,C).permute(0,3,1,2).contiguous()
        return tokA_out, tokB_out


class TransfuserBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoderA = timm.create_model('resnet34', pretrained=True, features_only=True,pretrained_cfg_overlay=dict(file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/pytorch_model.bin"),out_indices=(1, 2, 3, 4)   )

        self.image_encoderB = timm.create_model('resnet34', pretrained=True, features_only=True,pretrained_cfg_overlay=dict(file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/pytorch_model.bin"),out_indices=(1, 2, 3, 4)   )

        start_idx = 1  
        self.avgpoolA = nn.AdaptiveAvgPool2d((8,32))
        self.avgpoolB = nn.AdaptiveAvgPool2d((8,32))

        self.transformers = nn.ModuleList([
            TinyGPT(self.image_encoderA.feature_info.info[start_idx+i]["num_chs"])
            for i in range(4)
        ])
        self.B2A = nn.ModuleList([
            nn.Conv2d(self.image_encoderB.feature_info.info[start_idx+i]["num_chs"],
                      self.image_encoderA.feature_info.info[start_idx+i]["num_chs"], 1)
            for i in range(4)
        ])
        self.A2B = nn.ModuleList([
            nn.Conv2d(self.image_encoderA.feature_info.info[start_idx+i]["num_chs"],
                      self.image_encoderB.feature_info.info[start_idx+i]["num_chs"], 1)
            for i in range(4)
        ])
        c5_ch = self.image_encoderA.feature_info.info[start_idx+3]["num_chs"]
        self.c5_conv = nn.Conv2d(c5_ch, 64, 1)
        self.relu = nn.ReLU(True)
        self.up2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2size = nn.Upsample(size=(64,64), mode="bilinear", align_corners=False)
        self.up_conv5 = nn.Conv2d(64,64,3,padding=1)
        self.up_conv4 = nn.Conv2d(64,64,3,padding=1)

    @staticmethod
    def _fwd_block(layer_iter, return_layers, feat):
        for name, mod in layer_iter:
            feat = mod(feat)
            if name in return_layers:
                break
        return feat

    def _top_down(self, c5):
        p5 = self.relu(self.c5_conv(c5))
        p4 = self.relu(self.up_conv5(self.up2x(p5)))
        p3 = self.relu(self.up_conv4(self.up2size(p4)))  # 64×64
        return p3

    def forward(self, imgA, lidar):
        featA, featB = imgA, imgA
        iterA = iter(self.image_encoderA.items())
        iterB = iter(self.image_encoderB.items())

        for i in range(4):
            featA = self._fwd_block(iterA, self.image_encoderA.return_layers, featA)#3x256x1024---------256x32x8
            featB = self._fwd_block(iterB, self.image_encoderB.return_layers, featB)
            embA = self.avgpoolA(featA)
            embB = self.A2B[i](self.avgpoolB(featB))     
            newA, newB = self.transformers[i](embA, embB)
            newB = self.B2A[i](newB)                     

            featA = featA + F.interpolate(newA, size=featA.shape[2:], mode="bilinear", align_corners=False)
            featB = featB + F.interpolate(newB, size=featB.shape[2:], mode="bilinear", align_corners=False)

        fused_c5 = featA            
        feature_up = self._top_down(fused_c5)
        return feature_up, fused_c5,None



if __name__ == "__main__":
    model = TransfuserBackbone()
    imgA  = torch.randn(1, 3, 256, 1024)   
    imgB  = torch.randn(1, 3, 256, 1024)   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    imgA, imgB = imgA.to(device), imgB.to(device)

    p3, c5,_= model(imgA, imgB)
    print("FPN P3 :", p3.shape)   # torch.Size([1, 64, 64, 64])
    print("C5 fused:", c5.shape)  # torch.Size([1, 512, 8, 32])  