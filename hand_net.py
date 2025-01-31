import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import timm
from timm.models.layers import DropPath, Mlp
import hiera

from models.modules import MeshHead, AttentionBlock, IdentityBlock, SepConvBlock
from models.losses import mesh_to_joints
from models.losses import l1_loss

INDICES = [0, 4, 8, 12, 16]

class HandNet(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["MODEL"]
        backbone_cfg = model_cfg["BACKBONE"]

        self.loss_cfg = model_cfg["LOSSES"]

        if pretrained is None:
            pretrained=backbone_cfg['pretrain']            

        if "hiera" in backbone_cfg['model_name']:
            self.backbone = hiera.__dict__[backbone_cfg['model_name']](pretrained=True, checkpoint="mae_in1k",  drop_path_rate=backbone_cfg['drop_path_rate'])
            self.is_hiera = True
        else:
            self.backbone = timm.create_model(backbone_cfg['model_name'], pretrained=pretrained, drop_path_rate=backbone_cfg['drop_path_rate'])
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

        # Add new layers for marker integration
        marker_cfg = model_cfg.get('MARKER_HEAD', {
            'in_features': uv_cfg['in_features'],
            'num_markers': 5  #toFix
        })
        self.marker_head = nn.Linear(marker_cfg['in_features'], marker_cfg['num_markers'] * 3)
        
        # Add fusion layer to combine marker and regular features
        self.fusion_layer = nn.Linear(uv_cfg['in_features'] + marker_cfg['num_markers'] * 3, uv_cfg['in_features'])
              
        
    def compute_relative_markers(self, markers3d, wrist_joint):
        """Convert camera space markers to wrist-relative coordinates"""
        # markers3d: [B, N, 3], wrist_joint: [B, 3]
        wrist_joint = wrist_joint.unsqueeze(1)  # [B, 1, 3]
        relative_markers = markers3d - wrist_joint
        return relative_markers


    def infer(self, image, markers3d = None):
        if self.is_hiera:
            x, intermediates = self.backbone(image, return_intermediates=True)
            features = intermediates[-1]
            features = features.permute(0, 3, 1, 2).contiguous()
        else:
            features = self.backbone.forward_features(image)
        
        global_feature = self.avg_pool(features).squeeze(-1).squeeze(-1)
        
        # Initial UV prediction
        uv = self.keypoints_2d_head(global_feature)
        
        # Predict initial joints to get wrist position
        initial_vertices = self.mesh_head(features, uv)
        initial_joints = mesh_to_joints(initial_vertices)
        wrist_position = initial_joints[:, 0].detach()  # Assuming wrist is the first joint
        
        if markers3d is not None:
            # PREDICT markers using initial features
            pred_markers = self.marker_head(global_feature).reshape(-1, len(INDICES), 3)
    
            # Fuse features using PREDICTED markers
            marker_features = pred_markers.reshape(pred_markers.size(0), -1)
            fused_features = torch.cat([global_feature, marker_features], dim=1)
            enhanced_features = self.fusion_layer(fused_features)
    
            # Make final predictions
            uv = self.keypoints_2d_head(enhanced_features)
            vertices = self.mesh_head(features, uv)
            joints = mesh_to_joints(vertices)
        else:
            # Inference path (no markers)
            pred_markers = None
            vertices = initial_vertices
            joints = initial_joints

        return {
            "uv": uv,
            "joints": joints,
            "vertices": vertices,
            "pred_markers": pred_markers,
            "relative_markers" : relative_markers if markers3d is not None else None,
        }


    def forward(self, image, target=None):
        """get training loss

        Args:
            inputs (dict): {
                'img': (B, 1, H, W), 
                "uv": [B, 21, 2],
                "xyz": [B,  21, 3],
                "hand_uv_valid": [B, 21],
                "gamma": [B, 1],    

                "vertices": [B, 778, 3],
                "xyz_valid": [B,  21],
                "verts_valid": [B, 1],
                "hand_valid": [B, 1],
            }     
        """
        image = image / 255 - 0.5
        markers3d = target.get('markers3d') if target is not None else None
        output_dict = self.infer(image, markers3d)

        if self.training:
            assert target is not None
            loss_dict = self._cal_single_hand_losses(output_dict, target)
            return loss_dict

        return output_dict


    def _cal_single_hand_losses(self, pred_hand_dict, gt_hand_dict):
        """get training loss

        Args:
            pred_hand_dict (dict): {
                'uv': [B, 21, 2],
                'root_depth': [B, 1],
                'joints': [B, 21, 3],
                # 'vertices': [B, 778, 2],
            },
            gt_hand_dict (dict): {
                'uv': [B, 21, 2],
                'xyz': [B, 21, 3],
                'gamma': [B, 1],
                'uv_valid': [B, 21],
                # 'vertices': [B, 778, 3],
                # 'xyz_valid': [B, 21],
                # 'verts_valid': [B, 1],
            },            

        """
        

        uv_pred = pred_hand_dict['uv']
        # root_depth_pred = pred_hand_dict['root_depth']
        joints_pred = pred_hand_dict["joints"]
        vertices_pred = pred_hand_dict['vertices']


        uv_pred = uv_pred.reshape(-1, 21, 2).contiguous()
        joints_pred = joints_pred.reshape(-1, 21, 3).contiguous()
        # root_depth_pred = root_depth_pred.reshape(-1, 1).contiguous()

        uv_gt = gt_hand_dict['uv']
        joints_gt = gt_hand_dict['xyz']
        # root_depth_gt = gt_hand_dict['gamma'].reshape(-1, 1).contiguous()
        hand_uv_valid = gt_hand_dict['uv_valid']
        hand_xyz_valid = gt_hand_dict['xyz_valid'] # N, 1
        vertices_gt = gt_hand_dict['vertices']

        uv_loss = l1_loss(uv_pred, uv_gt, hand_uv_valid)
        joints_loss = l1_loss(joints_pred, joints_gt, valid=hand_xyz_valid)
        vertices_loss = l1_loss(vertices_pred, vertices_gt, valid=hand_xyz_valid)


        # root_depth_loss = (torch.abs(root_depth_pred- root_depth_gt)).mean()
        # root_depth_loss = root_depth_loss.mean()

        if 'markers3d' in gt_hand_dict and pred_hand_dict['pred_markers'] is not None:
            abs_markers_gt = gt_hand_dict['markers3d']  # [B, 5, 3]
    
            # Use PREDICTED wrist position (detached)
            wrist_pred = pred_hand_dict["joints"][:, 0].detach()  # [B, 3]
            relative_markers_gt = abs_markers_gt - wrist_pred.unsqueeze(1)  # [B, 5, 3]
    
            markers_pred = pred_hand_dict['pred_markers']  # [B, 5, 3]
    
            # Handle validity mask
            markers_valid = gt_hand_dict.get(
                'markers_valid', 
                torch.ones_like(abs_markers_gt[..., 0])  # [B, 5]
            )
            if markers_valid.shape[1] == 21:  # Subset if needed
                markers_valid = markers_valid[:, INDICES]
    
            marker_loss = l1_loss(markers_pred, relative_markers_gt, valid=markers_valid)
            loss_dict["marker_loss"] = marker_loss * self.loss_cfg.get("MARKER_LOSS_WEIGHT", 1.0)

        loss_dict = {
            "uv_loss": uv_loss * self.loss_cfg["UV_LOSS_WEIGHT"],
            "joints_loss": joints_loss * self.loss_cfg["JOINTS_LOSS_WEIGHT"],
            # "root_depth_loss": root_depth_loss * self.loss_cfg["DEPTH_LOSS_WEIGHT"],
            "vertices_loss": vertices_loss * self.loss_cfg["VERTICES_LOSS_WEIGHT"],            
        }

        total_loss = 0
        for k in loss_dict:
            total_loss += loss_dict[k]

        if 'markers3d' in gt_hand_dict and pred_hand_dict['pred_markers'] is not None:
            markers_pred = pred_hand_dict['pred_markers']
            markers_valid = torch.ones_like(markers_gt[..., 0])
            
            marker_loss = l1_loss(markers_pred, relative_markers_gt, valid=markers_valid)
            loss_dict["marker_loss"] = marker_loss * self.loss_cfg.get("MARKER_LOSS_WEIGHT", 1.0)
        
        total_loss = sum(loss_dict.values())
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


if __name__ == "__main__":
    import pickle
    import numpy as np
    from cfg import _CONFIG




    print('test forward')
    x = np.random.uniform(0, 255, (1, 3, 224, 224)).astype(np.float32)
    x = Tensor(x)

    print(x.shape)

    # model = timm.create_model("convnext_tiny", pretrained=True)
    # print(model)

    # out = model.forward_features(x)
    # print(out.shape)

    net = HandNet(_CONFIG)
    print(net)

    print("get losses")


    path = 'batch_data.pkl'
    with open(path, 'rb') as f:
        batch_data = pickle.load(f)
        for k in batch_data:
            batch_data[k] = Tensor(batch_data[k]).float()
            print(k, batch_data[k].shape, batch_data[k].max(), batch_data[k].min())

    losses_dict = net(batch_data['img'],batch_data)
    for key in losses_dict:
        print(key, losses_dict[key].item())


    # loss = losses_dict['total_loss']
    # loss.backward()