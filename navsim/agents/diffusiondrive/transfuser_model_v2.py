from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union
import json, pathlib

# from navsim.visualization.plots import plot_bev_with_agent
# from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

# agent = ConstantVelocityAgent()
# fig, ax = plot_bev_with_agent(scene, agent)
# plt.show()
class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone()
        ##############################################################
        H, W = 8, 32                     
        num_kv_tokens = H * W + 1         # 256 (grid) + 1 (status/cls) = 257
        d_model = config.tf_d_model             # 例如 64 / 128 / 256 / 512 之一
        self._keyval_embedding = nn.Embedding(num_kv_tokens, d_model)
        #############################################################
        #self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )
         # 冻结未用权重
        unused_file = pathlib.Path(
            "/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/unused_param_list.json"
        )
        if unused_file.is_file():
            unused = set(json.load(open(unused_file)))
            for name, param in self.named_parameters():
                if name in unused:
                    param.requires_grad = False

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        
        # 提取输入特征
        camera_feature: torch.Tensor = features["camera_feature"]
        #print(f"1. camera_feature shape: {camera_feature.shape}")
        
        lidar_feature: torch.Tensor = features["lidar_feature"]
        #print(f"2. lidar_feature shape: {lidar_feature.shape}")
        
        status_feature: torch.Tensor = features["status_feature"]
        #print(f"3. status_feature shape: {status_feature.shape}")
        # ===== 可视化代码开始 =====
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime
        
        # 创建保存目录
        save_dir = "/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/exp/visualizations"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳作为文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        
        try:
            # 1. 保存 Camera 特征 (1, 3, 256, 1024)
            if camera_feature.shape == torch.Size([1, 3, 256, 1024]):
                # 取第一个batch，转换为 (256, 1024, 3) RGB格式
                camera_img = camera_feature[0].permute(1, 2, 0).cpu().numpy()
                
                # 归一化到 [0, 1] 范围
                camera_img = (camera_img - camera_img.min()) / (camera_img.max() - camera_img.min() + 1e-8)
                
                # 保存 camera 图像
                plt.figure(figsize=(15, 6))
                plt.imshow(camera_img)
                plt.title(f'Camera Feature - {camera_feature.shape}')
                plt.axis('off')
                plt.tight_layout()
                camera_path = os.path.join(save_dir, f"camera_{timestamp}.png")
                plt.savefig(camera_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Camera feature saved: {camera_path}")
            
            # 2. 保存 LiDAR 特征 (1, 1, 256, 256)
            if lidar_feature.shape == torch.Size([1, 1, 256, 256]):
                # 取第一个batch和第一个channel
                lidar_img = lidar_feature[0, 0].cpu().numpy()
                
                # 保存 lidar 图像
                plt.figure(figsize=(8, 8))
                plt.imshow(lidar_img, cmap='viridis')  # 使用viridis色彩映射更适合深度数据
                plt.title(f'LiDAR Feature - {lidar_feature.shape}')
                plt.colorbar()
                plt.axis('off')
                plt.tight_layout()
                lidar_path = os.path.join(save_dir, f"lidar_{timestamp}.png")
                plt.savefig(lidar_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"LiDAR feature saved: {lidar_path}")
            
            # 3. 保存组合图像
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Camera 子图
            if camera_feature.shape == torch.Size([1, 3, 256, 1024]):
                camera_img = camera_feature[0].permute(1, 2, 0).cpu().numpy()
                camera_img = (camera_img - camera_img.min()) / (camera_img.max() - camera_img.min() + 1e-8)
                ax1.imshow(camera_img)
                ax1.set_title(f'Camera Feature\nShape: {camera_feature.shape}')
                ax1.axis('off')
            
            # LiDAR 子图
            if lidar_feature.shape == torch.Size([1, 1, 256, 256]):
                lidar_img = lidar_feature[0, 0].cpu().numpy()
                im = ax2.imshow(lidar_img, cmap='viridis')
                ax2.set_title(f'LiDAR Feature\nShape: {lidar_feature.shape}')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2)
            
            plt.tight_layout()
            combined_path = os.path.join(save_dir, f"combined_{timestamp}.png")
            plt.savefig(combined_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Combined visualization saved: {combined_path}")
            
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # ===== 可视化代码结束 =====
    
        
        batch_size = status_feature.shape[0]
        #print(f"4. batch_size: {batch_size}")
        
        # Backbone处理
        #print("--- BACKBONE 处理 ---")
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        # print(f"5. bev_feature_upscale shape: {bev_feature_upscale.shape}")
        # print(f"6. bev_feature shape: {bev_feature.shape}")
        # print()
        
        # BEV特征处理
        #print("--- BEV 特征处理 ---")
        cross_bev_feature = bev_feature_upscale
        #print(f"7. cross_bev_feature shape: {cross_bev_feature.shape}")
        
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        #print(f"8. bev_spatial_shape: {bev_spatial_shape}")
        
        concat_cross_bev_shape = bev_feature.shape[2:]
        #print(f"9. concat_cross_bev_shape: {concat_cross_bev_shape}")
        
        bev_feature_downscaled = self._bev_downscale(bev_feature)
        #print(f"10. bev_feature after downscale: {bev_feature_downscaled.shape}")
        
        bev_feature_flattened = bev_feature_downscaled.flatten(-2, -1)
        #print(f"11. bev_feature after flatten: {bev_feature_flattened.shape}")
        
        bev_feature = bev_feature_flattened.permute(0, 2, 1)
        #print(f"12. bev_feature after permute: {bev_feature.shape}")
        #print()
        
        # Status编码
        #print("--- STATUS 编码 ---")
        status_encoding = self._status_encoding(status_feature)
        #print(f"13. status_encoding shape: {status_encoding.shape}")
        #print()
        
        # KeyVal构建
        #print("--- KEYVAL 构建 ---")
        status_encoding_expanded = status_encoding[:, None]
        #rint(f"14. status_encoding expanded shape: {status_encoding_expanded.shape}")
        
        keyval = torch.concatenate([bev_feature, status_encoding_expanded], dim=1)
        #print(f"15. keyval after concatenate: {keyval.shape}")
        
        keyval_embedding_weight = self._keyval_embedding.weight[None, ...]
        #print(f"16. keyval_embedding weight shape: {keyval_embedding_weight.shape}")
        
        keyval += keyval_embedding_weight
        
        # 交叉BEV特征重构
        #print("--- 交叉BEV特征重构 ---")
        keyval_without_last = keyval[:,:-1]
        #print(f"18. keyval without last shape: {keyval_without_last.shape}")
        
        keyval_permuted = keyval_without_last.permute(0,2,1)
        #print(f"19. keyval after permute: {keyval_permuted.shape}")
        
        keyval_contiguous = keyval_permuted.contiguous()
        #print(f"20. keyval after contiguous: {keyval_contiguous.shape}")
        
        concat_cross_bev = keyval_contiguous.view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        #print(f"21. concat_cross_bev after view: {concat_cross_bev.shape}")
        #print(f"    Target view shape: ({batch_size}, -1, {concat_cross_bev_shape[0]}, {concat_cross_bev_shape[1]})")
        #print()
        
        # 上采样到相同形状
        #print("--- 上采样处理 ---")
        concat_cross_bev_upsampled = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # print(f"22. concat_cross_bev after interpolate: {concat_cross_bev_upsampled.shape}")
        # print(f"    Target interpolate size: {bev_spatial_shape}")
        
        # 拼接特征
        #print("--- 特征拼接 ---")
        cross_bev_feature_cat = torch.cat([concat_cross_bev_upsampled, cross_bev_feature], dim=1)
        #print(f"23. cross_bev_feature after cat: {cross_bev_feature_cat.shape}")
        
        cross_bev_feature_flattened = cross_bev_feature_cat.flatten(-2,-1)
        #print(f"24. cross_bev_feature after flatten: {cross_bev_feature_flattened.shape}")
        
        cross_bev_feature_permuted = cross_bev_feature_flattened.permute(0,2,1)
        #print(f"25. cross_bev_feature after permute: {cross_bev_feature_permuted.shape}")
        
        cross_bev_feature_projected = self.bev_proj(cross_bev_feature_permuted)
        #print(f"26. cross_bev_feature after projection: {cross_bev_feature_projected.shape}")
        
        cross_bev_feature_final_permute = cross_bev_feature_projected.permute(0,2,1)
        #print(f"27. cross_bev_feature after final permute: {cross_bev_feature_final_permute.shape}")
        
        cross_bev_feature_final_contiguous = cross_bev_feature_final_permute.contiguous()
        #print(f"28. cross_bev_feature after contiguous: {cross_bev_feature_final_contiguous.shape}")
        
        cross_bev_feature = cross_bev_feature_final_contiguous.view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
 #
        
        # Query处理
        # print("--- QUERY 处理 ---")
        query_embedding_weight = self._query_embedding.weight[None, ...]
        # print(f"30. query_embedding weight shape: {query_embedding_weight.shape}")
        
        query = query_embedding_weight.repeat(batch_size, 1, 1)
        # print(f"31. query after repeat: {query.shape}")
        
        query_out = self._tf_decoder(query, keyval)
        # print(f"32. query_out from transformer decoder: {query_out.shape}")
        # print()
        
        # 语义分割头
        # print("--- 语义分割头 ---")
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        # print(f"33. bev_semantic_map shape: {bev_semantic_map.shape}")
        
        # Query分割
        # print("--- QUERY 分割 ---")
        # print(f"34. query_splits: {self._query_splits}")
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
        # print(f"35. trajectory_query shape: {trajectory_query.shape}")
        # print(f"36. agents_query shape: {agents_query.shape}")

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=None)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )


        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img)


    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)


        # 4. begin the stacked decoder
        poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)

        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg,"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict}

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg}




# from typing import Dict
# import numpy as np
# import torch
# import torch.nn as nn
# import copy
# from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
# from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
# from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
# from navsim.common.enums import StateSE2Index
# from diffusers.schedulers import DDIMScheduler
# from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
# import torch.nn.functional as F
# from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
# from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
# from torch.nn import TransformerDecoder,TransformerDecoderLayer
# from typing import Any, List, Dict, Optional, Union
# class V2TransfuserModel(nn.Module):
#     """Torch module for Transfuser."""

#     def __init__(self, config: TransfuserConfig):
#         """
#         Initializes TransFuser torch module.
#         :param config: global config dataclass of TransFuser.
#         """

#         super().__init__()

#         self._query_splits = [
#             1,
#             config.num_bounding_boxes,
#         ]

#         self._config = config
#         self._backbone = TransfuserBackbone(config)

#         self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
#         self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

#         # usually, the BEV features are variable in size.
#         self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
#         self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

#         self._bev_semantic_head = nn.Sequential(
#             nn.Conv2d(
#                 config.bev_features_channels,
#                 config.bev_features_channels,
#                 kernel_size=(3, 3),
#                 stride=1,
#                 padding=(1, 1),
#                 bias=True,
#             ),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 config.bev_features_channels,
#                 config.num_bev_classes,
#                 kernel_size=(1, 1),
#                 stride=1,
#                 padding=0,
#                 bias=True,
#             ),
#             nn.Upsample(
#                 size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
#                 mode="bilinear",
#                 align_corners=False,
#             ),
#         )

#         tf_decoder_layer = nn.TransformerDecoderLayer(
#             d_model=config.tf_d_model,
#             nhead=config.tf_num_head,
#             dim_feedforward=config.tf_d_ffn,
#             dropout=config.tf_dropout,
#             batch_first=True,
#         )

#         self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
#         self._agent_head = AgentHead(
#             num_agents=config.num_bounding_boxes,
#             d_ffn=config.tf_d_ffn,
#             d_model=config.tf_d_model,
#         )

#         self._trajectory_head = TrajectoryHead(
#             num_poses=config.trajectory_sampling.num_poses,
#             d_ffn=config.tf_d_ffn,
#             d_model=config.tf_d_model,
#             plan_anchor_path=config.plan_anchor_path,
#             config=config,
#         )
#         self.bev_proj = nn.Sequential(
#             *linear_relu_ln(256, 1, 1,320),
#         )


#     def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
#         """Torch module forward pass."""

#         camera_feature: torch.Tensor = features["camera_feature"]
#         lidar_feature: torch.Tensor = features["lidar_feature"]
#         status_feature: torch.Tensor = features["status_feature"]

#         batch_size = status_feature.shape[0]

#         bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
#         cross_bev_feature = bev_feature_upscale
#         bev_spatial_shape = bev_feature_upscale.shape[2:]
#         concat_cross_bev_shape = bev_feature.shape[2:]
#         bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
#         bev_feature = bev_feature.permute(0, 2, 1)
#         status_encoding = self._status_encoding(status_feature)

#         keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
#         keyval += self._keyval_embedding.weight[None, ...]

#         concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
#         # upsample to the same shape as bev_feature_upscale

#         concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
#         # concat concat_cross_bev and cross_bev_feature
#         cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

#         cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
#         cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
#         query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
#         query_out = self._tf_decoder(query, keyval)

#         bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
#         trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

#         output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

#         trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=None)
#         output.update(trajectory)

#         agents = self._agent_head(agents_query)
#         output.update(agents)

#         return output

# class AgentHead(nn.Module):
#     """Bounding box prediction head."""

#     def __init__(
#         self,
#         num_agents: int,
#         d_ffn: int,
#         d_model: int,
#     ):
#         """
#         Initializes prediction head.
#         :param num_agents: maximum number of agents to predict
#         :param d_ffn: dimensionality of feed-forward network
#         :param d_model: input dimensionality
#         """
#         super(AgentHead, self).__init__()

#         self._num_objects = num_agents
#         self._d_model = d_model
#         self._d_ffn = d_ffn

#         self._mlp_states = nn.Sequential(
#             nn.Linear(self._d_model, self._d_ffn),
#             nn.ReLU(),
#             nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
#         )

#         self._mlp_label = nn.Sequential(
#             nn.Linear(self._d_model, 1),
#         )

#     def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
#         """Torch module forward pass."""

#         agent_states = self._mlp_states(agent_queries)
#         agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
#         agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

#         agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

#         return {"agent_states": agent_states, "agent_labels": agent_labels}

# class DiffMotionPlanningRefinementModule(nn.Module):
#     def __init__(
#         self,
#         embed_dims=256,
#         ego_fut_ts=8,
#         ego_fut_mode=20,
#         if_zeroinit_reg=True,
#     ):
#         super(DiffMotionPlanningRefinementModule, self).__init__()
#         self.embed_dims = embed_dims
#         self.ego_fut_ts = ego_fut_ts
#         self.ego_fut_mode = ego_fut_mode
#         self.plan_cls_branch = nn.Sequential(
#             *linear_relu_ln(embed_dims, 1, 2),
#             nn.Linear(embed_dims, 1),
#         )
#         self.plan_reg_branch = nn.Sequential(
#             nn.Linear(embed_dims, embed_dims),
#             nn.ReLU(),
#             nn.Linear(embed_dims, embed_dims),
#             nn.ReLU(),
#             nn.Linear(embed_dims, ego_fut_ts * 3),
#         )
#         self.if_zeroinit_reg = False

#         self.init_weight()

#     def init_weight(self):
#         if self.if_zeroinit_reg:
#             nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
#             nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

#         bias_init = bias_init_with_prob(0.01)
#         nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
#     def forward(
#         self,
#         traj_feature,
#     ):
#         bs, ego_fut_mode, _ = traj_feature.shape

#         # 6. get final prediction
#         traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
#         plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
#         traj_delta = self.plan_reg_branch(traj_feature)
#         plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

#         return plan_reg, plan_cls
# class ModulationLayer(nn.Module):

#     def __init__(self, embed_dims: int, condition_dims: int):
#         super(ModulationLayer, self).__init__()
#         self.if_zeroinit_scale=False
#         self.embed_dims = embed_dims
#         self.scale_shift_mlp = nn.Sequential(
#             nn.Mish(),
#             nn.Linear(condition_dims, embed_dims*2),
#         )
#         self.init_weight()

#     def init_weight(self):
#         if self.if_zeroinit_scale:
#             nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
#             nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

#     def forward(
#         self,
#         traj_feature,
#         time_embed,
#         global_cond=None,
#         global_img=None,
#     ):
#         if global_cond is not None:
#             global_feature = torch.cat([
#                     global_cond, time_embed
#                 ], axis=-1)
#         else:
#             global_feature = time_embed
#         if global_img is not None:
#             global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
#             global_feature = torch.cat([
#                     global_img, global_feature
#                 ], axis=-1)
        
#         scale_shift = self.scale_shift_mlp(global_feature)
#         scale,shift = scale_shift.chunk(2,dim=-1)
#         traj_feature = traj_feature * (1 + scale) + shift
#         return traj_feature

# class CustomTransformerDecoderLayer(nn.Module):
#     def __init__(self, 
#                  num_poses,
#                  d_model,
#                  d_ffn,
#                  config,
#                  ):
#         super().__init__()
#         self.dropout = nn.Dropout(0.1)
#         self.dropout1 = nn.Dropout(0.1)
#         self.cross_bev_attention = GridSampleCrossBEVAttention(
#             config.tf_d_model,
#             config.tf_num_head,
#             num_points=num_poses,
#             config=config,
#             in_bev_dims=256,
#         )
#         self.cross_agent_attention = nn.MultiheadAttention(
#             config.tf_d_model,
#             config.tf_num_head,
#             dropout=config.tf_dropout,
#             batch_first=True,
#         )
#         self.cross_ego_attention = nn.MultiheadAttention(
#             config.tf_d_model,
#             config.tf_num_head,
#             dropout=config.tf_dropout,
#             batch_first=True,
#         )
#         self.ffn = nn.Sequential(
#             nn.Linear(config.tf_d_model, config.tf_d_ffn),
#             nn.ReLU(),
#             nn.Linear(config.tf_d_ffn, config.tf_d_model),
#         )
#         self.norm1 = nn.LayerNorm(config.tf_d_model)
#         self.norm2 = nn.LayerNorm(config.tf_d_model)
#         self.norm3 = nn.LayerNorm(config.tf_d_model)
#         self.time_modulation = ModulationLayer(config.tf_d_model,256)
#         self.task_decoder = DiffMotionPlanningRefinementModule(
#             embed_dims=config.tf_d_model,
#             ego_fut_ts=num_poses,
#             ego_fut_mode=20,
#         )

#     def forward(self, 
#                 traj_feature, 
#                 noisy_traj_points, 
#                 bev_feature, 
#                 bev_spatial_shape, 
#                 agents_query, 
#                 ego_query, 
#                 time_embed, 
#                 status_encoding,
#                 global_img=None):
#         traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
#         traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
#         traj_feature = self.norm1(traj_feature)
        
#         # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

#         # 4.5 cross attention with  ego query
#         traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
#         traj_feature = self.norm2(traj_feature)
        
#         # 4.6 feedforward network
#         traj_feature = self.norm3(self.ffn(traj_feature))
#         # 4.8 modulate with time steps
#         traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
#         # 4.9 predict the offset & heading
#         poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
#         poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
#         poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

#         return poses_reg, poses_cls
# def _get_clones(module, N):
#     # FIXME: copy.deepcopy() is not defined on nn.module
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# class CustomTransformerDecoder(nn.Module):
#     def __init__(
#         self, 
#         decoder_layer, 
#         num_layers,
#         norm=None,
#     ):
#         super().__init__()
#         torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
    
#     def forward(self, 
#                 traj_feature, 
#                 noisy_traj_points, 
#                 bev_feature, 
#                 bev_spatial_shape, 
#                 agents_query, 
#                 ego_query, 
#                 time_embed, 
#                 status_encoding,
#                 global_img=None):
#         poses_reg_list = []
#         poses_cls_list = []
#         traj_points = noisy_traj_points
#         for mod in self.layers:
#             poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
#             poses_reg_list.append(poses_reg)
#             poses_cls_list.append(poses_cls)
#             traj_points = poses_reg[...,:2].clone().detach()
#         return poses_reg_list, poses_cls_list

# class TrajectoryHead(nn.Module):
#     """Trajectory prediction head."""

#     def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
#         """
#         Initializes trajectory head.
#         :param num_poses: number of (x,y,θ) poses to predict
#         :param d_ffn: dimensionality of feed-forward network
#         :param d_model: input dimensionality
#         """
#         super(TrajectoryHead, self).__init__()

#         self._num_poses = num_poses
#         self._d_model = d_model
#         self._d_ffn = d_ffn
#         self.diff_loss_weight = 2.0
#         self.ego_fut_mode = 20

#         self.diffusion_scheduler = DDIMScheduler(
#             num_train_timesteps=1000,
#             beta_schedule="scaled_linear",
#             prediction_type="sample",
#         )


#         plan_anchor = np.load(plan_anchor_path)

#         self.plan_anchor = nn.Parameter(
#             torch.tensor(plan_anchor, dtype=torch.float32),
#             requires_grad=False,
#         ) # 20,8,2
#         self.plan_anchor_encoder = nn.Sequential(
#             *linear_relu_ln(d_model, 1, 1,512),
#             nn.Linear(d_model, d_model),
#         )
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(d_model),
#             nn.Linear(d_model, d_model * 4),
#             nn.Mish(),
#             nn.Linear(d_model * 4, d_model),
#         )

#         diff_decoder_layer = CustomTransformerDecoderLayer(
#             num_poses=num_poses,
#             d_model=d_model,
#             d_ffn=d_ffn,
#             config=config,
#         )
#         self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

#         self.loss_computer = LossComputer(config)
#     def norm_odo(self, odo_info_fut):
#         odo_info_fut_x = odo_info_fut[..., 0:1]
#         odo_info_fut_y = odo_info_fut[..., 1:2]
#         odo_info_fut_head = odo_info_fut[..., 2:3]

#         odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
#         odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
#         odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
#         return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
#     def denorm_odo(self, odo_info_fut):
#         odo_info_fut_x = odo_info_fut[..., 0:1]
#         odo_info_fut_y = odo_info_fut[..., 1:2]
#         odo_info_fut_head = odo_info_fut[..., 2:3]

#         odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
#         odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
#         odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
#         return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
#     def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
#         """Torch module forward pass."""
#         if self.training:
#             return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img)
#         else:
#             return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img)


#     def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
#         bs = ego_query.shape[0]
#         device = ego_query.device
#         # 1. add truncated noise to the plan anchor
#         plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
#         odo_info_fut = self.norm_odo(plan_anchor)
#         timesteps = torch.randint(
#             0, 50,
#             (bs,), device=device
#         )
#         noise = torch.randn(odo_info_fut.shape, device=device)
#         noisy_traj_points = self.diffusion_scheduler.add_noise(
#             original_samples=odo_info_fut,
#             noise=noise,
#             timesteps=timesteps,
#         ).float()
#         noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
#         noisy_traj_points = self.denorm_odo(noisy_traj_points)

#         ego_fut_mode = noisy_traj_points.shape[1]
#         # 2. proj noisy_traj_points to the query
#         traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
#         traj_pos_embed = traj_pos_embed.flatten(-2)
#         traj_feature = self.plan_anchor_encoder(traj_pos_embed)
#         traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
#         # 3. embed the timesteps
#         time_embed = self.time_mlp(timesteps)
#         time_embed = time_embed.view(bs,1,-1)


#         # 4. begin the stacked decoder
#         poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)

#         trajectory_loss_dict = {}
#         ret_traj_loss = 0
#         for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
#             trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
#             trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
#             ret_traj_loss += trajectory_loss

#         mode_idx = poses_cls_list[-1].argmax(dim=-1)
#         mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
#         best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
#         return {"trajectory": best_reg,"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict}

#     def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img) -> Dict[str, torch.Tensor]:
#         step_num = 2
#         bs = ego_query.shape[0]
#         device = ego_query.device
#         self.diffusion_scheduler.set_timesteps(1000, device)
#         step_ratio = 20 / step_num
#         roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
#         roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


#         # 1. add truncated noise to the plan anchor
#         plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
#         img = self.norm_odo(plan_anchor)
#         noise = torch.randn(img.shape, device=device)
#         trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
#         img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
#         noisy_trajs = self.denorm_odo(img)
#         ego_fut_mode = img.shape[1]
#         for k in roll_timesteps[:]:
#             x_boxes = torch.clamp(img, min=-1, max=1)
#             noisy_traj_points = self.denorm_odo(x_boxes)

#             # 2. proj noisy_traj_points to the query
#             traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
#             traj_pos_embed = traj_pos_embed.flatten(-2)
#             traj_feature = self.plan_anchor_encoder(traj_pos_embed)
#             traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

#             timesteps = k
#             if not torch.is_tensor(timesteps):
#                 # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
#                 timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
#             elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
#                 timesteps = timesteps[None].to(img.device)
            
#             # 3. embed the timesteps
#             timesteps = timesteps.expand(img.shape[0])
#             time_embed = self.time_mlp(timesteps)
#             time_embed = time_embed.view(bs,1,-1)

#             # 4. begin the stacked decoder
#             poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
#             poses_reg = poses_reg_list[-1]
#             poses_cls = poses_cls_list[-1]
#             x_start = poses_reg[...,:2]
#             x_start = self.norm_odo(x_start)
#             img = self.diffusion_scheduler.step(
#                 model_output=x_start,
#                 timestep=k,
#                 sample=img
#             ).prev_sample
#         mode_idx = poses_cls.argmax(dim=-1)
#         mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
#         best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
#         return {"trajectory": best_reg}