import torch
import torch.nn.functional as F
from torch import nn

class TransfuserBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # ① 把 (3,256,1024) → (512,8,32)
        self.simple_resize = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 32)),   # 先缩到固定网格
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1)
        )

        # ② 把 (512,8,32) → (64,64,64)
        self.top_down = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)
        )

    def forward(self, imgA, lidar=None):      # lidar 无用，放占位
        feat_c5  = self.simple_resize(imgA)   # (B,512,8,32)
        feat_p3  = self.top_down(feat_c5)     # (B,64,64,64)
        return feat_p3, feat_c5, None


# ─────────── quick test ───────────
if __name__ == "__main__":
    model = TransfuserBackbone().cuda()
    imgA  = torch.randn(1, 3, 256, 1024).cuda()

    p3, c5, _ = model(imgA, None)
    print("P3 shape :", p3.shape)   # torch.Size([1, 64, 64, 64])
    print("C5 shape :", c5.shape)   # torch.Size([1, 512, 8, 32])
