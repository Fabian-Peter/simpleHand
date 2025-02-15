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
#debug global feature vector map display  
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
#debug uv keypoint visualization
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_uv_keypoints(image, uv_pred, uv_gt, save_path='./debug_img/uv_keypoints.png'):
    """
    Overlay both predicted and ground truth UV keypoints on the image with skeleton lines.
    
    Args:
        image (torch.Tensor or np.ndarray): Cropped image in shape [C, H, W] in range [0,1].
        uv_pred (torch.Tensor or np.ndarray): Predicted keypoints, shape [N, 2] (normalized).
        uv_gt (torch.Tensor or np.ndarray): Ground truth keypoints, shape [N, 2] (normalized).
        save_path (str): Path to save the visualization.
    """
    # Convert image to numpy (H, W, C)
    if hasattr(image, 'detach'):
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        image_np = image

    H, W, _ = image_np.shape

    # Convert keypoints to numpy arrays and reshape if needed
    if hasattr(uv_pred, 'detach'):
        uv_pred = uv_pred.detach().cpu().view(-1, 2).numpy()
    if hasattr(uv_gt, 'detach'):
        uv_gt = uv_gt.detach().cpu().view(-1, 2).numpy()

    # Scale UV coordinates from normalized [0, 1] to pixel space [0, W] and [0, H]
    uv_pred_pixel = uv_pred * np.array([W, H])
    uv_gt_pixel = uv_gt * np.array([W, H])

    # Define the hand skeleton as connections (edges) between keypoint indices.
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
    
    # Draw ground truth skeleton in green
    for edge in skeleton:
        pt1, pt2 = edge
        x_vals = [uv_gt_pixel[pt1, 0], uv_gt_pixel[pt2, 0]]
        y_vals = [uv_gt_pixel[pt1, 1], uv_gt_pixel[pt2, 1]]
        plt.plot(x_vals, y_vals, 'g-', linewidth=2)
    
    # Draw predicted skeleton in red
    for edge in skeleton:
        pt1, pt2 = edge
        x_vals = [uv_pred_pixel[pt1, 0], uv_pred_pixel[pt2, 0]]
        y_vals = [uv_pred_pixel[pt1, 1], uv_pred_pixel[pt2, 1]]
        plt.plot(x_vals, y_vals, 'r-', linewidth=2)
    
    # Overlay ground truth keypoints with green circles
    plt.scatter(uv_gt_pixel[:, 0], uv_gt_pixel[:, 1], s=100, c='green', marker='o', label='Ground Truth')
    # Overlay predicted keypoints with red crosses
    plt.scatter(uv_pred_pixel[:, 0], uv_pred_pixel[:, 1], s=100, c='red', marker='x', label='Predicted')

    plt.title("Predicted vs Ground Truth 2D Keypoints with Hand Skeleton")
    plt.axis('off')
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"UV keypoints visualization saved to: {save_path}")

#-------------

def visualize_joints_comparison(joints_pred, joints_gt, save_path='./debug_img/joints_comparison.png'):
    """
    Visualize predicted vs ground truth joints in a 3D scatter plot with skeleton lines.
    
    Args:
        joints_pred (torch.Tensor or np.ndarray): Predicted joints for one image, shape [21, 3].
        joints_gt (torch.Tensor or np.ndarray): Ground truth joints for one image, shape [21, 3].
        save_path (str): Path where to save the visualization.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
    import os

    # Convert to numpy arrays if necessary
    if isinstance(joints_pred, torch.Tensor):
        joints_pred = joints_pred.detach().cpu().numpy()
    if isinstance(joints_gt, torch.Tensor):
        joints_gt = joints_gt.detach().cpu().numpy()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot predicted joints (red crosses) and ground truth joints (green circles)
    ax.scatter(joints_pred[:, 0], joints_pred[:, 1], joints_pred[:, 2],
               c='r', marker='x', s=50, label='Predicted')
    ax.scatter(joints_gt[:, 0], joints_gt[:, 1], joints_gt[:, 2],
               c='g', marker='o', s=50, label='Ground Truth')

    # Define the hand skeleton connections (the same as in your 2D keypoint visualization)
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # Draw skeleton lines for predicted joints in red
    for (i, j) in skeleton:
        ax.plot([joints_pred[i, 0], joints_pred[j, 0]],
                [joints_pred[i, 1], joints_pred[j, 1]],
                [joints_pred[i, 2], joints_pred[j, 2]],
                c='r', linewidth=2, alpha=0.5)
    
    # Draw skeleton lines for ground truth joints in green
    for (i, j) in skeleton:
        ax.plot([joints_gt[i, 0], joints_gt[j, 0]],
                [joints_gt[i, 1], joints_gt[j, 1]],
                [joints_gt[i, 2], joints_gt[j, 2]],
                c='g', linewidth=2, alpha=0.5)

    ax.set_title("Predicted vs Ground Truth Joints")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Predicted vs Ground Truth joints visualization saved to: {save_path}")

def visualize_predicted_joints(joints, save_path='./debug_img/joints.png'):
    """
    Visualize predicted joints in a 3D scatter plot along with skeleton connections.
    
    Args:
        joints (torch.Tensor or np.ndarray): Predicted joints for one image with shape [21, 3].
        save_path (str): Path where to save the visualization.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
    import os

    # Convert joints to numpy if needed.
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy()

    # Create the debug directory if it doesn't exist.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a 3D scatter plot.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints as blue scatter points.
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c='b', marker='o', s=30, label='Joints')
    
    # Define hand skeleton connections (same as in your 2D keypoints).
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    # Draw lines for each connection.
    for start, end in skeleton:
        xs = [joints[start, 0], joints[end, 0]]
        ys = [joints[start, 1], joints[end, 1]]
        zs = [joints[start, 2], joints[end, 2]]
        ax.plot(xs, ys, zs, c='r', linewidth=2)
    
    ax.set_title("Predicted Joints")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Predicted joints visualization saved to: {save_path}")


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
        


    def infer(self, image, gt_uv=None, gt_joints=None):
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
        #visualize_uv_keypoints(image[0], uv_pred=uv[0], uv_gt=gt_uv[0])
        #-------------
        vertices = self.mesh_head(features, uv)

        #debug predicted vertices
        #visualize_predicted_vertices(vertices[0], save_path='./debug_img/vertices.png')


        joints = mesh_to_joints(vertices)
        #debug
        #visualize_joints_comparison(joints_pred[0], gt_xyz=gt_joints[0], save_path='./debug_img/joints_comparison.png')

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
        if target is not None:
            # Pass both the predicted and ground truth UV/Joints to the infer method.
            output_dict = self.infer(image, gt_uv=target["uv"], gt_joints=target["xyz"])
        else:
            output_dict = self.infer(image)
        if self.training:
            # In training mode, we require target to compute losses.
            assert target is not None
            loss_dict = self._cal_single_hand_losses(output_dict, target)
            return loss_dict
        else:
            # In eval mode, if a target is provided, you can compute evaluation metrics (e.g., losses) 
            # and return both the predictions and those metrics.
            if target is not None:
                eval_metrics = self._cal_single_hand_losses(output_dict, target)
                return output_dict, eval_metrics
            else:
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