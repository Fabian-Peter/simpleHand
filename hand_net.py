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

import matplotlib.pyplot as plt
import numpy as np
import os

#-------------
#debug backbone feature extraction map display  
def visualize_feature_maps(features, num_channels_to_show=16, figsize=(15, 8), save_path='./debug_img/feature_maps.png'):
    """
    Visualize feature maps from the backbone network and save to file
    
    Args:
        features (torch.Tensor): Feature maps tensor of shape (B, C, H, W)
        num_channels_to_show (int): Number of channels to display
        figsize (tuple): Figure size for matplotlib
        save_path (str): Path to save the visualization
    """
    print(f"Starting feature map visualization...")
    print(f"Feature tensor shape: {features.shape}")

    # Create debug_img directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created/verified directory: {save_dir}")

    # Convert features to numpy and take first batch
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
        print("Converted features to numpy array")

    if len(features.shape) == 4:
        features = features[0]  # Take first batch
        print(f"Selected first batch, shape now: {features.shape}")

    # Select subset of channels to display
    num_channels = min(num_channels_to_show, features.shape[0])
    selected_channels = features[:num_channels]
    print(f"Selected {num_channels} channels for visualization")

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    print(f"Grid size: {grid_size}x{grid_size}")

    try:
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        fig.suptitle('Feature Maps from Backbone', fontsize=16)

        # Normalize each feature map independently
        for idx, feature_map in enumerate(selected_channels):
            if idx >= num_channels:
                break

            row = idx // grid_size
            col = idx % grid_size

            # Normalize feature map
            feature_map = feature_map - feature_map.min()
            feature_map = feature_map / (feature_map.max() + 1e-8)

            # Plot
            if grid_size == 1:
                axes.imshow(feature_map, cmap='viridis')
                axes.axis('off')
                axes.set_title(f'Channel {idx}')
            else:
                if isinstance(axes, np.ndarray):
                    axes[row, col].imshow(feature_map, cmap='viridis')
                    axes[row, col].axis('off')
                    axes[row, col].set_title(f'Channel {idx}')
                else:
                    axes.imshow(feature_map, cmap='viridis')
                    axes.axis('off')
                    axes.set_title(f'Channel {idx}')

        # Turn off remaining empty subplots
        if grid_size > 1 and isinstance(axes, np.ndarray):
            for idx in range(num_channels, grid_size * grid_size):
                row = idx // grid_size
                col = idx % grid_size
                axes[row, col].axis('off')

        plt.tight_layout()

        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved feature maps to: {save_path}")
        plt.close()  # Close the figure to free memory

        return save_path

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise
#-------------

def visualize_global_feature_vector(global_feature, save_path='./debug_img/global_feature.png'):
    """
    Visualize a global feature vector.
    Args:
        global_feature (torch.Tensor): Tensor of shape [D] (for one image)
        save_path (str): Path where to save the plot
    """
    import matplotlib.pyplot as plt
    import os

    # Convert to numpy
    global_feature_np = global_feature.detach().cpu().numpy()

    # Create the debug_img directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 5))
    plt.plot(global_feature_np, marker='o', linestyle='-', markersize=2)
    plt.title("Global Feature Vector")
    plt.xlabel("Dimension")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Global feature vector saved to: {save_path}")

#-------------
def visualize_uv_keypoints(image, uv, save_path='./debug_img/uv_keypoints.png'):
    """
    Overlay predicted uv keypoints on the cropped and resized image.
    
    Args:
        image (torch.Tensor or np.ndarray): Cropped image in shape [C, H, W] in range [0,1].
        uv (torch.Tensor or np.ndarray): Keypoints tensor, shape [N, 2].
        save_path (str): Path to save the visualization.
    """
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        image_np = image

    H, W, _ = image_np.shape

    # Convert uv to numpy array and reshape if needed
    if isinstance(uv, torch.Tensor):
        uv = uv.detach().cpu().view(-1, 2).numpy()

    # Scale UV coordinates from normalized [0, 1] to pixel space [0, W] and [0, H]
    uv_pixel = uv * np.array([W, H])

    # Define the hand skeleton as connections (edges) between keypoint indices.
    # Note: 0 is the wrist.
    skeleton = [
        # Thumb: wrist -> 1 -> 2 -> 3 -> 4
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger: wrist -> 5 -> 6 -> 7 -> 8
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger: wrist -> 9 -> 10 -> 11 -> 12
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger: wrist -> 13 -> 14 -> 15 -> 16
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Little finger: wrist -> 17 -> 18 -> 19 -> 20
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)

    # Draw skeleton lines
    for edge in skeleton:
        pt1, pt2 = edge
        x_vals = [uv_pixel[pt1, 0], uv_pixel[pt2, 0]]
        y_vals = [uv_pixel[pt1, 1], uv_pixel[pt2, 1]]
        plt.plot(x_vals, y_vals, 'b-', linewidth=2)  # Blue lines

    # Overlay keypoints
    plt.scatter(uv_pixel[:, 0], uv_pixel[:, 1], s=100, c='red', marker='o')
    plt.title("Predicted 2D Keypoints with Hand Skeleton")
    plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"UV keypoints visualization saved to: {save_path}")

#-------------
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
        


    def infer(self, image):
        if self.is_hiera:
            x, intermediates = self.backbone(image, return_intermediates=True)
            features = intermediates[-1]
            features = features.permute(0, 3, 1, 2).contiguous()
        else:
            features = self.backbone.forward_features(image)
        
        #-------------
        #debug backbone feature extraction map display    
        #save_path = './debug_img/feature_maps.png'
        #visualize_feature_maps(features, save_path=save_path)
        #-------------
        global_feature = self.avg_pool(features).squeeze(-1).squeeze(-1)
        #-------------
        #debug global feature vector
        #visualize_global_feature_vector(global_feature[0], save_path='./debug_img/global_feature.png')
        #-------------
        uv = self.keypoints_2d_head(global_feature)     
        # depth = self.depth_head(global_feature)
        #-------------
        #debug predicted uv keypoints     
        visualize_uv_keypoints(image[0], uv[0], save_path='./debug_img/uv_keypoints.png')
        #-------------
        vertices = self.mesh_head(features, uv)
        joints = mesh_to_joints(vertices)

        return {
            "uv": uv,
            # "root_depth": depth,
            "joints": joints,
            "vertices": vertices,            
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
        output_dict = self.infer(image)
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


        loss_dict = {
            "uv_loss": uv_loss * self.loss_cfg["UV_LOSS_WEIGHT"],
            "joints_loss": joints_loss * self.loss_cfg["JOINTS_LOSS_WEIGHT"],
            # "root_depth_loss": root_depth_loss * self.loss_cfg["DEPTH_LOSS_WEIGHT"],
            "vertices_loss": vertices_loss * self.loss_cfg["VERTICES_LOSS_WEIGHT"],            
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