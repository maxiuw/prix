
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
import matplotlib.pylab as plt 
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
            new  = self.avgpool(feat)            # (B,C,8,32)
            new  = self.gpts[i](new)
            new  = self.channel_proj[i](new)
            feat = feat + F.interpolate(new, size=feat.shape[2:], mode="bilinear", align_corners=False)

        c5 = feat                                # (B,C5,8,32)
        p3 = self._top_down(c5)                  # (B,64,64,64)
        return p3, c5, None # [64, 2048, 8, 32] -> shoud have 512 dims c5, 

    def visualize_features(self, features, save_path, original_size=(256, 1024)):
        """
        Visualizes a batch of feature maps by averaging across channels and resizing.

        Args:
            features (torch.Tensor): Feature map tensor of shape (B, C, H, W).
            save_path (str): Path to save the output image.
            original_size (tuple): The (height, width) to resize the map to for viewing.
        """
        # Detach from graph and move to CPU
        features = features.detach().cpu()
        
        # Average across the channel dimension
        heatmap = features.mean(dim=1) # Shape: (B, H, W)
        
        # Resize the heatmap to a more viewable size
        heatmap_resized = F.interpolate(
            heatmap.unsqueeze(1),       # Add channel dim: (B, 1, H, W)
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)                    # Remove channel dim: (B, H, W)

        # Plot each heatmap in the batch
        batch_size = features.shape[0]
        fig, axes = plt.subplots(batch_size, 1, figsize=(15, 5 * batch_size), squeeze=False)

        for i in range(batch_size):
            ax = axes[i, 0]
            im = ax.imshow(heatmap_resized[i], cmap='viridis')
            ax.set_title(f'Feature Heatmap (Batch Item {i})')
            ax.axis('off')
            fig.colorbar(im, ax=ax)

        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Feature visualization saved to '{save_path}'")

    def visualize_features_on_image(self, image_tensor, features, save_path, alpha=0.6):
        """
        Visualizes feature maps as a heatmap overlaid on the original image.

        Args:
            image_tensor (torch.Tensor): Original input image tensor (B, 3, H, W).
            features (torch.Tensor): Feature map tensor (B, C, H, W).
            save_path (str): Path to save the output image.
            alpha (float): Transparency of the heatmap overlay.
        """
        # --- Data Preparation ---
        features = features.detach().cpu()
        image_tensor = image_tensor.detach().cpu()
        
        original_size = image_tensor.shape[2:]
        
        # Create and resize heatmap
        heatmap = features.mean(dim=1)
        heatmap_resized = F.interpolate(
            heatmap.unsqueeze(1), size=original_size, mode='bilinear', align_corners=False
        ).squeeze(1)

        # --- Plotting ---
        batch_size = features.shape[0]
        fig, axes = plt.subplots(batch_size, 1, figsize=(15, 5 * batch_size), squeeze=False)

        for i in range(batch_size):
            # Prepare image for plotting (C, H, W) -> (H, W, C)
            img = image_tensor[i].permute(1, 2, 0).numpy()
            # Normalize to [0, 1] for display, in case it's not already
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            # Plot the original image and the heatmap overlay
            ax = axes[i, 0]
            ax.imshow(img)
            ax.imshow(heatmap_resized[i], cmap='viridis', alpha=alpha)
            ax.set_title(f'Feature Overlay (Batch Item {i})')
            ax.axis('off')

        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Feature overlay saved to '{save_path}'")
# ───────────────────────── quick test ─────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransfuserBackbone().to(device).eval()
    dummy  = torch.randn(1, 3, 256, 1024, device=device)

    with torch.no_grad():
        p3, c5, _ = model(dummy)

    print("P3 :", p3.shape)  # torch.Size([1, 64, 64, 64])
    print("C5 :", c5.shape)  # torch.Size([1, 512, 8, 32])  (C5 通道随骨干而变)

