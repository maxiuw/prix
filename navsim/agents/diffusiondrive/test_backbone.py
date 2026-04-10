
# """
# Implements the TransFuser vision backbone.
# """

# import copy
# import math

# import timm
# import torch
# import torch.nn.functional as F
# from torch import nn



# class TransfuserBackbone(nn.Module):
#     """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

#     def __init__(self):

#         super().__init__()
        
#         self.image_encoder = timm.create_model('resnet34', pretrained=True, features_only=True,pretrained_cfg_overlay=dict(file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/vit/pytorch_model.bin"),out_indices=(1, 2, 3, 4)   )
#         in_channels = 1


#         self.avgpool_img = nn.AdaptiveAvgPool2d((8, 32))

#         self.lidar_encoder = timm.create_model('resnet34', pretrained=True, features_only=True,pretrained_cfg_overlay=dict(file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/vit/pytorch_model.bin"),out_indices=(1, 2, 3, 4)   )

#         self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
#         self.avgpool_lidar = nn.AdaptiveAvgPool2d((8,8))
#         lidar_time_frames = [1, 1, 1, 1]

#         self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
#         start_index = 1
#         # # # Some networks have a stem layer
#         # if len(self.image_encoder.return_layers) > 4:
#         #     start_index += 1

#         self.transformers = nn.ModuleList(
#             [
#                 GPT(
#                     n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
#                     lidar_time_frames=1,         
#                 )
#                 for i in range(4)
#             ]
#         )

#         self.lidar_channel_to_img = nn.ModuleList(
#             [
#                 nn.Conv2d(
#                     self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
#                     self.image_encoder.feature_info.info[start_index + i]["num_chs"],
#                     kernel_size=1,
#                 )
#                 for i in range(4)
#             ]
#         )
#         self.img_channel_to_lidar = nn.ModuleList(
#             [
#                 nn.Conv2d(
#                     self.image_encoder.feature_info.info[start_index + i]["num_chs"],
#                     self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
#                     kernel_size=1,
#                 )
#                 for i in range(4)
#             ]
#         )

#         self.num_image_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
#         # Typical encoders down-sample by a factor of 32
#         self.perspective_upsample_factor = (
#             self.image_encoder.feature_info.info[start_index + 3]["reduction"]
#         )

#         self.num_features = self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]


#         # FPN fusion
#         channel = 64
#         self.relu = nn.ReLU(inplace=True)
#         # top down
#         self.upsample = nn.Upsample(
#             scale_factor=2, mode="bilinear", align_corners=False
#         )
#         self.upsample2 = nn.Upsample(
#             size=(64,64),
#             mode="bilinear",
#             align_corners=False,
#         )

#         self.up_conv5 = nn.Conv2d(64, 64, (3, 3), padding=1)
#         self.up_conv4 = nn.Conv2d(64, 64, (3, 3), padding=1)

#         # lateral
#         self.c5_conv = nn.Conv2d(self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"], 64, (1, 1))
#         print("img return_layers:", self.image_encoder.return_layers)
#         print("lidar return_layers:", self.lidar_encoder.return_layers)
#         for i, fi in enumerate(self.lidar_encoder.feature_info.info):
#             print(i, fi["num_chs"])

#     def top_down(self, x):

#         p5 = self.relu(self.c5_conv(x))
#         p4 = self.relu(self.up_conv5(self.upsample(p5)))
#         p3 = self.relu(self.up_conv4(self.upsample2(p4)))

#         return p3

#     def forward(self, image, lidar):
#         """
#         Image + LiDAR feature fusion using transformers
#         Args:
#             image_list (list): list of input images
#             lidar_list (list): list of input LiDAR BEV
#         """
#         image_features, lidar_features = image, image

#         # Generate an iterator for all the layers in the network that one can loop through.
#         image_layers = iter(self.image_encoder.items())
#         lidar_layers = iter(self.lidar_encoder.items())
    
#         for i in range(4):
#             image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
#             lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
#             image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)

#         image_feature_grid = None
#         fused_features = lidar_features
#         features = self.top_down(lidar_features)

#         return features, fused_features, image_feature_grid

#     def forward_layer_block(self, layers, return_layers, features):
#         """
#         Run one forward pass to a block of layers from a TIMM neural network and returns the result.
#         Advances the whole network by just one block
#         :param layers: Iterator starting at the current layer block
#         :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
#         :param features: Input features
#         :return: Processed features
#         """
#         for name, module in layers:
#             features = module(features)
#             if name in return_layers:
#                 break
#         return features

#     def fuse_features(self, image_features, lidar_features, layer_idx):
#         """
#         Perform a TransFuser feature fusion block using a Transformer module.
#         :param image_features: Features from the image branch
#         :param lidar_features: Features from the LiDAR branch
#         :param layer_idx: Transformer layer index.
#         :return: image_features and lidar_features with added features from the other branch.
#         """
#         image_embd_layer = self.avgpool_img(image_features)
#         lidar_embd_layer = self.avgpool_lidar(lidar_features)
#         lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

#         image_features_layer, lidar_features_layer = self.transformers[layer_idx](image_embd_layer, lidar_embd_layer)
#         lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

#         image_features_layer = F.interpolate(
#             image_features_layer,
#             size=(image_features.shape[2], image_features.shape[3]),
#             mode="bilinear",
#             align_corners=False,
#         )
#         lidar_features_layer = F.interpolate(
#             lidar_features_layer,
#             size=(lidar_features.shape[2], lidar_features.shape[3]),
#             mode="bilinear",
#             align_corners=False,
#         )

#         image_features = image_features + image_features_layer
#         lidar_features = lidar_features + lidar_features_layer

#         return image_features, lidar_features


# class GPT(nn.Module):
#     """The full GPT language backbone, with a context size of block_size."""

#     # def __init__(self, n_embd, config, lidar_video, lidar_time_frames):
#     def __init__(self, n_embd, lidar_time_frames):
#         super().__init__()
#         self.n_embd = n_embd
#         # We currently only support seq len 1
#         self.seq_len = 1
#         self.lidar_seq_len = 1
#         self.lidar_time_frames = 1

#         # positional embedding parameter (learnable), image + lidar
#         self.pos_emb = nn.Parameter(
#             torch.zeros(
#                 1,
#                 1 * 8 * 32 + 1 * 8 * 8,
#                 self.n_embd,
#             )
#         )

#         self.drop = nn.Dropout(0.1)

#         # transformer
#         self.blocks = nn.Sequential(
#             *[
#                 Block(n_embd, 4, 4, 0.1, 0.1)
#                 for layer in range(2)
#             ]
#         )

#         # decoder head
#         self.ln_f = nn.LayerNorm(n_embd)

#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(
#                 mean=0.0,
#                 std=0.02,
#             )
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(self, image_tensor, lidar_tensor):
#         """
#         Args:
#             image_tensor (tensor): B*4*seq_len, C, H, W
#             lidar_tensor (tensor): B*seq_len, C, H, W
#         """

#         bz = lidar_tensor.shape[0]
#         lidar_h, lidar_w = lidar_tensor.shape[2:4]

#         img_h, img_w = image_tensor.shape[2:4]

#         assert self.seq_len == 1
#         image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
#         lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

#         token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

#         x = self.drop(self.pos_emb + token_embeddings)
#         x = self.blocks(x)  # (B, an * T, C)
#         x = self.ln_f(x)  # (B, an * T, C)

#         image_tensor_out = (
#             x[:, : 1 * 32 * 8, :]
#             .view(bz * 1, img_h, img_w, -1)
#             .permute(0, 3, 1, 2)
#             .contiguous()
#         )
#         lidar_tensor_out = (
#             x[
#                 :,
#                 1 * 8 * 32 :,
#                 :,
#             ]
#             .view(bz, lidar_h, lidar_w, -1)
#             .permute(0, 3, 1, 2)
#             .contiguous()
#         )

#         return image_tensor_out, lidar_tensor_out


# class SelfAttention(nn.Module):
#     """
#     A vanilla multi-head masked self-attention layer with a projection at the
#     end.
#     """

#     def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
#         super().__init__()
#         assert n_embd % n_head == 0
#         # key, query, value projections for all heads
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         # output projection
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head

#     def forward(self, x):
#         b, t, c = x.size()

#         # calculate query, key, values for all heads in batch and move head
#         # forward to be the batch dim
#         k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
#         q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
#         v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

#         # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
#         y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_drop(self.proj(y))
#         return y


# class Block(nn.Module):
#     """an unassuming Transformer block"""

#     def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)
#         self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
#         self.mlp = nn.Sequential(
#             nn.Linear(n_embd, block_exp * n_embd),
#             nn.ReLU(True),  # changed from GELU
#             nn.Linear(block_exp * n_embd, n_embd),
#             nn.Dropout(resid_pdrop),
#         )

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))

#         return x

# if __name__ == "__main__":
#     # 1) 实例化骨干网络
#     model = TransfuserBackbone()

#     # 2) 准备输入：RGB 图像 3×256×256；LiDAR BEV 1×256×1024
#     image = torch.randn(1, 3, 256, 1024)      # (B, C, H, W)
#     lidar = torch.randn(1, 1, 256, 256)     # (B, C, H, W)



#     # ─────────── 可选：使用 GPU ───────────
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     image, lidar = image.to(device), lidar.to(device)
#     # ─────────────────────────────────────

#     # 3) 前向推理
#     bev_feature_upscale, bev_feature, _ = model(image, lidar)

#     # 4) 打印输出特征图尺寸
#     print("bev_feature_upscale:", bev_feature_upscale.size())
#     print("bev_feature:", bev_feature.size())


"""
Image-Image TransFuser backbone (no LiDAR branch)
"""
import math
import timm
import torch
import torch.nn.functional as F
from torch import nn


# ─────────── Transformer block（保持不变） ───────────
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


# ─────────── 主干网络 ───────────
class TransfuserBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # 两个 ResNet34（可共享权重或独立）
        self.image_encoderA = timm.create_model('resnet34', pretrained=True, features_only=True,pretrained_cfg_overlay=dict(file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/vit/pytorch_model.bin"),out_indices=(1, 2, 3, 4)   )

        self.image_encoderB = timm.create_model('resnet34', pretrained=True, features_only=True,pretrained_cfg_overlay=dict(file="/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/pretrained/vit/pytorch_model.bin"),out_indices=(1, 2, 3, 4)   )

        start_idx = 1  # 对应 C2
        self.avgpoolA = nn.AdaptiveAvgPool2d((8,32))
        self.avgpoolB = nn.AdaptiveAvgPool2d((8,32))

        self.transformers = nn.ModuleList([
            TinyGPT(self.image_encoderA.feature_info.info[start_idx+i]["num_chs"])
            for i in range(4)
        ])

        # 通道映射（可直接 1×1 Conv，也可 Identity）
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

        # FPN：只对 A 分支做上采样，可按需要改
        c5_ch = self.image_encoderA.feature_info.info[start_idx+3]["num_chs"]
        self.c5_conv = nn.Conv2d(c5_ch, 64, 1)
        self.relu = nn.ReLU(True)
        self.up2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2size = nn.Upsample(size=(64,64), mode="bilinear", align_corners=False)
        self.up_conv5 = nn.Conv2d(64,64,3,padding=1)
        self.up_conv4 = nn.Conv2d(64,64,3,padding=1)

    # helper：推进一个 block
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

    def forward(self, imgA, imgB):
        featA, featB = imgA, imgB
        iterA = iter(self.image_encoderA.items())
        iterB = iter(self.image_encoderB.items())

        for i in range(4):
            featA = self._fwd_block(iterA, self.image_encoderA.return_layers, featA)
            featB = self._fwd_block(iterB, self.image_encoderB.return_layers, featB)

            # 池化到 8×32，送 Transformer
            embA = self.avgpoolA(featA)
            embB = self.A2B[i](self.avgpoolB(featB))     # 通道对齐到 A
            newA, newB = self.transformers[i](embA, embB)
            newB = self.B2A[i](newB)                     # 再映射回 B

            # 残差融合
            featA = featA + F.interpolate(newA, size=featA.shape[2:], mode="bilinear", align_corners=False)
            featB = featB + F.interpolate(newB, size=featB.shape[2:], mode="bilinear", align_corners=False)

        fused_c5 = featA            # 这里以 A 分支的 C5 作为输出
        feature_up = self._top_down(fused_c5)
        return feature_up, fused_c5


# ─────────── quick test ───────────
if __name__ == "__main__":
    model = TransfuserBackbone()
    imgA  = torch.randn(1, 3, 256, 1024)   # front view
    imgB  = torch.randn(1, 3, 256, 1024)   # rear / side / 任意

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    imgA, imgB = imgA.to(device), imgB.to(device)

    p3, c5 = model(imgA, imgB)
    print("FPN P3 :", p3.shape)   # torch.Size([1, 64, 64, 64])
    print("C5 fused:", c5.shape)  # torch.Size([1, 512, 8, 32])  (512 只是举例)
