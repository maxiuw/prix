import math
import timm
import torch
import torch.nn.functional as F
from torch import nn

# ───────────────────── Transformer Primitives ─────────────────────
class _SelfAttn(nn.Module):
    """A simple self-attention layer."""
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
    """A standard transformer block with pre-normalization."""
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

class CaRT(nn.Module):
    """Context-Aware Recalibration Transformer (CaRT)"""
    def __init__(self, dim, n_layer=2):
        super().__init__()
        # Positional embedding for the flattened 8x32 feature map
        self.pos_emb = nn.Parameter(torch.zeros(1, 8 * 32, dim))
        self.drop    = nn.Dropout(0.1)
        self.blocks  = nn.Sequential(*[_TransformerBlock(dim) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(dim)

    def forward(self, feat):
        B, C, H, W = feat.shape # Expected H=8, W=32
        tok = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x   = self.ln_f(self.blocks(self.drop(tok + self.pos_emb)))
        x   = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

# ───────────────────── Backbone with SHARED CaRT ─────────────────────
class TransfuserBackboneShared(nn.Module):
    def __init__(self, shared_cart_dim=512):
        super().__init__()
        print(f"Initializing TransfuserBackboneShared with shared CaRT dimension: {shared_cart_dim}")
        # ── ResNet-34 Encoder ──
        self.encoder = timm.create_model(
            'resnet34',#'resnet50', #
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            output_stride=32,
            # This part for custom weights is kept from your original code
            # pretrained_cfg_overlay=dict(
            #     file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/pytorch_model.bin"
            # )
        )
        # self.encoder = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)

        # Get the channel dimensions from the ResNet layers (e.g., [64, 128, 256, 512])
        ch_list = [fi["num_chs"] for fi in self.encoder.feature_info.info[1:5]]

        self.avgpool = nn.AdaptiveAvgPool2d((8, 32))

        # ─── SHARED CaRT MODULE AND PROJECTIONS ───
        # 1. A single, shared CaRT module
        self.shared_cart = CaRT(shared_cart_dim)

        # 2. Projection layers to map each scale TO the shared dimension
        self.projs_in = nn.ModuleList([
            nn.Conv2d(in_ch, shared_cart_dim, kernel_size=1) for in_ch in ch_list
        ])

        # 3. Projection layers to map each scale FROM the shared dimension
        self.projs_out = nn.ModuleList([
            nn.Conv2d(shared_cart_dim, out_ch, kernel_size=1) for out_ch in ch_list
        ])
        # ──────────────────────────────────────────

        # FPN Top-Down Path
        self.c5_to_64 = nn.Conv2d(ch_list[-1], 64, 1)
        self.relu = nn.ReLU(True)
        self.up2x     = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2fix = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)
        self.up_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.up_conv2 = nn.Conv2d(64, 64, 3, padding=1)

    @staticmethod
    def _run_to_next_return(it_layers, return_layers, x):
        """Helper to run the ResNet encoder to the next feature map."""
        for name, mod in it_layers:
            x = mod(x)
            if name in return_layers:
                break
        return x

    def _top_down(self, c5):
        """Standard FPN top-down path to generate P3."""
        p5 = self.relu(self.c5_to_64(c5))       # (B,64, 8,32)
        p4 = self.relu(self.up_conv1(self.up2x(p5)))   # (B,64,16,64) -> This is not used, just an intermediate step
        p3 = self.relu(self.up_conv2(self.up2fix(p4))) # (B,64,64,64)
        return p3

    def forward(self, img, *_):
        feat = img
        it_layers = iter(self.encoder.items())

        # Loop through ResNet stages 2, 3, 4, 5
        for i in range(4):
            # Get feature map from the current ResNet stage
            feat = self._run_to_next_return(it_layers, self.encoder.return_layers, feat)

            # Recalibrate features using the SHARED CaRT module
            emb = self.avgpool(feat)                 # (B, C_in, 8, 32)
            projected_in = self.projs_in[i](emb)     # (B, shared_dim, 8, 32)
            recalibrated = self.shared_cart(projected_in) # (B, shared_dim, 8, 32)
            projected_out = self.projs_out[i](recalibrated) # (B, C_in, 8, 32)

            # Apply residual connection
            feat = feat + F.interpolate(projected_out, size=feat.shape[2:], mode="bilinear", align_corners=False)

        c5 = feat # Final feature map from stage 5
        p3 = self._top_down(c5)
        # self.pcaviz(p3,img)
        return p3, c5, None
    
    def pcaviz(self, feature_map, img):
        from sklearn.decomposition import PCA
        import matplotlib.pylab as plt 
        import numpy as np
        # --- Method 1: Average Activation (GPU) ---
        # This operation is fast on the GPU
        activation_map = -torch.mean(feature_map, dim=1).squeeze()


        # --- Method 2: PCA Activation (GPU -> CPU -> GPU) ---
        # Reshape on GPU
        # b, c, h, w = feature_map.shape
        # feature_map_reshaped = feature_map.permute(0, 2, 3, 1).reshape(b * h * w, c)

        # # Apply PCA on CPU
        # pca = PCA(n_components=1)
        # # Move tensor to CPU before converting to numpy for scikit-learn
        # principal_components = pca.fit_transform(feature_map_reshaped.detach().cpu().numpy())

        # # Reshape back to a 2D map. This is now a numpy array on the CPU.
        # activation_map = principal_components.reshape(h, w)
      
        # Move tensor to CPU for processing and plotting
        activation_map = activation_map.detach().cpu().numpy()

        activation_map_tensor = torch.from_numpy(activation_map.astype(np.float32))

        upsampled_map = F.interpolate(
            activation_map_tensor.unsqueeze(0).unsqueeze(0),
            size=(256, 1024),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Move image tensor to CPU for plotting with matplotlib
        image_to_plot = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(image_to_plot)
        ax.imshow(upsampled_map, cmap='viridis', alpha=0.6)
        ax.axis('off')
        fig.savefig('testfeatures_c5_1avg.png')
        
