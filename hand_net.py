import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import timm
from timm.models.layers import DropPath, Mlp
import hiera

from models.modules import MeshHead, AttentionBlock, IdentityBlock, SepConvBlock, MarkerBranch
from models.losses import mesh_to_joints
from models.losses import l1_loss

class HandNet(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["MODEL"]
        backbone_cfg = model_cfg["BACKBONE"]

        self.loss_cfg = model_cfg["LOSSES"]

        if pretrained is None:
            pretrained = backbone_cfg['pretrain']

        if "hiera" in backbone_cfg['model_name']:
            self.backbone = hiera.__dict__[backbone_cfg['model_name']](
                pretrained=True,
                checkpoint="mae_in1k",
                drop_path_rate=backbone_cfg['drop_path_rate']
            )
            self.is_hiera = True
        else:
            self.backbone = timm.create_model(
                backbone_cfg['model_name'],
                pretrained=pretrained,
                drop_path_rate=backbone_cfg['drop_path_rate']
            )
            self.is_hiera = False

        self.avg_pool = nn.AvgPool2d((7, 7), 1)

        uv_cfg = model_cfg['UV_HEAD']
        depth_cfg = model_cfg['DEPTH_HEAD']

        self.keypoints_2d_head = nn.Linear(uv_cfg['in_features'], uv_cfg['out_features'])
        # self.depth_head = nn.Linear(depth_cfg['in_features'], depth_cfg['out_features'])

        mesh_head_cfg = model_cfg["MESH_HEAD"].copy()
        block_types_name = mesh_head_cfg['block_types']
        block_types = []
        block_map = {
            "attention": AttentionBlock,
            "identity": IdentityBlock,
            "conv": SepConvBlock,
        }
        for name in block_types_name:
            block_types.append(block_map[name])
        mesh_head_cfg['block_types'] = block_types

        self.mesh_head = MeshHead(**mesh_head_cfg)

        # ---------------------------
        # Add the marker branch.
        # We assume marker heatmaps have 5 channels.
        # After initializing self.marker_branch...
        self.marker_branch = MarkerBranch(in_channels=5, out_channels=64)

        if hasattr(self.backbone, "num_features"):
            self.rgb_feature_channels = self.backbone.num_features
        else:
            self.rgb_feature_channels = 1216

        # Add a projection layer to expand marker features to the same channel size as the RGB features.
        self.marker_proj = nn.Conv2d(64, self.rgb_feature_channels, kernel_size=1)

        # Instead of the concatenation-based fusion, use a learnable scalar for weighted addition:
        self.marker_weight = nn.Parameter(torch.tensor(0.1))  # Start small.

        # Optionally, add a BatchNorm for stabilization after fusion.
        self.fusion_norm = nn.BatchNorm2d(self.rgb_feature_channels)

        # ---------------------------
    
    def infer(self, image, marker_heatmaps=None):
        # Extract RGB features from the backbone.
        if self.is_hiera:
            x, intermediates = self.backbone(image, return_intermediates=True)
            features = intermediates[-1]
            features = features.permute(0, 3, 1, 2).contiguous()
        else:
            features = self.backbone.forward_features(image)
    
        # Process marker heatmaps if provided.
        if marker_heatmaps is not None:
            # Downsample marker heatmaps to match the spatial dimensions of features.
            marker_heatmaps_ds = F.adaptive_avg_pool2d(marker_heatmaps, (features.shape[2], features.shape[3]))
            marker_features = self.marker_branch(marker_heatmaps_ds)  # Shape: [B, 64, H, W]
            # Project marker features to match the RGB feature channels.
            marker_features = self.marker_proj(marker_features)      # Now shape: [B, 1216, H, W]
            # Apply the learnable scalar weight.
            marker_features = self.marker_weight * marker_features
            # Fuse by adding marker features to the RGB features.
            fused_features = features + marker_features
            # Optionally, apply normalization after fusion.
            fused_features = self.fusion_norm(fused_features)
        else:
            fused_features = features
            print("else")
    
        global_feature = self.avg_pool(fused_features).squeeze(-1).squeeze(-1)
        uv = self.keypoints_2d_head(global_feature)
        vertices = self.mesh_head(fused_features, uv)
        joints = mesh_to_joints(vertices)

        return {
            "uv": uv,
            "joints": joints,
            "vertices": vertices,
        }
    
    def forward(self, image, target=None, marker_heatmaps=None):
        """
        Forward pass. Expects:
            - image: Tensor of shape [B, 3, H, W]
            - marker_heatmaps (optional): Tensor of shape [B, 5, H, W]
            - target: Ground truth dictionary for training.
        """
        image = image / 255 - 0.5
        output_dict = self.infer(image, marker_heatmaps=marker_heatmaps)
        if self.training:
            assert target is not None
            loss_dict = self._cal_single_hand_losses(output_dict, target)
            return loss_dict
        return output_dict

    def _cal_single_hand_losses(self, pred_hand_dict, gt_hand_dict):
        uv_pred = pred_hand_dict['uv']
        joints_pred = pred_hand_dict["joints"]
        vertices_pred = pred_hand_dict['vertices']

        uv_pred = uv_pred.reshape(-1, 21, 2).contiguous()
        joints_pred = joints_pred.reshape(-1, 21, 3).contiguous()

        uv_gt = gt_hand_dict['uv']
        joints_gt = gt_hand_dict['xyz']
        hand_uv_valid = gt_hand_dict['uv_valid']
        hand_xyz_valid = gt_hand_dict['xyz_valid']
        vertices_gt = gt_hand_dict['vertices']

        uv_loss = l1_loss(uv_pred, uv_gt, hand_uv_valid)
        joints_loss = l1_loss(joints_pred, joints_gt, valid=hand_xyz_valid)
        vertices_loss = l1_loss(vertices_pred, vertices_gt, valid=hand_xyz_valid)

        loss_dict = {
            "uv_loss": uv_loss * self.loss_cfg["UV_LOSS_WEIGHT"],
            "joints_loss": joints_loss * self.loss_cfg["JOINTS_LOSS_WEIGHT"],
            "vertices_loss": vertices_loss * self.loss_cfg["VERTICES_LOSS_WEIGHT"],
        }
        total_loss = sum(loss for loss in loss_dict.values())
        loss_dict['total_loss'] = total_loss
        
        return loss_dict

if __name__ == "__main__":
    import pickle
    import numpy as np
    from cfg import _CONFIG

    print('test forward')
    x = np.random.uniform(0, 255, (1, 3, 224, 224)).astype(np.float32)
    x = Tensor(x)

    net = HandNet(_CONFIG)
    print(net)
    print("get losses")

    path = 'batch_data.pkl'
    with open(path, 'rb') as f:
        batch_data = pickle.load(f)
        for k in batch_data:
            batch_data[k] = Tensor(batch_data[k]).float()
            print(k, batch_data[k].shape, batch_data[k].max(), batch_data[k].min())

    # When testing, pass marker_heatmaps along with image and target.
    losses_dict = net(batch_data['img'], batch_data, marker_heatmaps=batch_data['marker_heatmaps'])
    for key in losses_dict:
        print(key, losses_dict[key].item())
