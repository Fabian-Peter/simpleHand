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

        # Get backbone output features
        backbone_out_features = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1216
    
        # Define dimensions
        self.backbone_dim = backbone_out_features  # 1216
        self.root_dim = 3
        self.markers_dim = 15  # 5 markers * 3 coordinates
        self.total_input_dim = self.backbone_dim + self.root_dim + self.markers_dim  # 1234
    
        # Initialize layers with correct dimensions
        self.keypoints_2d_head = nn.Linear(uv_cfg['in_features'], uv_cfg['out_features'])
        self.root_head = nn.Linear(backbone_out_features, self.root_dim)
    
        # marker_normalizer should map from total_input_dim to uv_cfg['in_features']
        self.marker_normalizer = nn.Linear(self.total_input_dim, uv_cfg['in_features'])
    
        # Rest of your initialization code...
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

    def infer(self, image, target=None):
        if self.is_hiera:
            x, intermediates = self.backbone(image, return_intermediates=True)
            features = intermediates[-1]
            features = features.permute(0, 3, 1, 2).contiguous()
        else:
            features = self.backbone.forward_features(image)
    
        global_feature = self.avg_pool(features).squeeze(-1).squeeze(-1)  # [B, C]
        uv = self.keypoints_2d_head(global_feature)     
    
        vertices = self.mesh_head(features, uv)
        joints = mesh_to_joints(vertices)

        root_pred = self.root_head(global_feature)  # [B, 3]
        device = global_feature.device

        batch_size = global_feature.shape[0]

        if self.training and target is not None:
            markers_gt = target['markers3d']  # [B, 5, 3]
            normalized_markers = markers_gt - root_pred.unsqueeze(1)  # [B, 5, 3]
        else:
            normalized_markers = torch.zeros(batch_size, 5, 3, device=device)  # [B, 5, 3]

        marker_features = normalized_markers.reshape(batch_size, -1)  # [B, 15]
          
        # Ensure concatenation is done correctly
        fused_features = torch.cat([
            global_feature,  # [B, C]
            root_pred,      # [B, 3]
            marker_features # [B, 15]
        ], dim=1)
           
        enhanced_features = self.marker_normalizer(fused_features)
    
        uv = self.keypoints_2d_head(enhanced_features)
        vertices = self.mesh_head(features, uv)
        joints = mesh_to_joints(vertices)

        return {
            "uv": uv,
            "joints": joints,
            "vertices": vertices,  
            "root_pred": root_pred          
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
        output_dict = self.infer(image, target)
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

        root_pred = pred_hand_dict["joints"][:, 0]  # [B,3]
        root_gt = gt_hand_dict['joints'][:, 0]  # [B,3]
        root_loss = F.l1_loss(root_pred, root_gt)

        # root_depth_loss = (torch.abs(root_depth_pred- root_depth_gt)).mean()
        # root_depth_loss = root_depth_loss.mean()


        loss_dict = {
            "uv_loss": uv_loss * self.loss_cfg["UV_LOSS_WEIGHT"],
            "joints_loss": joints_loss * self.loss_cfg["JOINTS_LOSS_WEIGHT"],
            # "root_depth_loss": root_depth_loss * self.loss_cfg["DEPTH_LOSS_WEIGHT"],
            "vertices_loss": vertices_loss * self.loss_cfg["VERTICES_LOSS_WEIGHT"], 
            "root_loss": root_loss * 2.0 #todo           
        }

        total_loss = 0
        for k in loss_dict:
            total_loss += loss_dict[k]

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