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

def visualize_hand_predictions(pred_coords, gt_coords, root_index=0, debug=False):
    """
    Visualize and compare predicted and ground truth hand coordinates.
    Additionally, adjust the predicted values for certain joints (indexes 4, 8, 12, 16, 20)
    by replacing them with the ground truth values (in root-relative coordinates).
    Finally, undo the root-relative transformation and return the adjusted predicted values.
    
    Args:
        pred_coords (np.ndarray or torch.Tensor): Shape (N, 3) array of predicted 3D coordinates in absolute space.
        gt_coords (np.ndarray or torch.Tensor): Shape (N, 3) array of ground truth 3D coordinates in absolute space.
        root_index (int): Index of the root joint for making coordinates root-relative.
        debug (bool): If True, generate debug plots.
        
    Returns:
        torch.Tensor: Adjusted predicted 3D coordinates in absolute space.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Save device if the inputs are tensors.
    device = torch.device("cpu")
    if isinstance(pred_coords, torch.Tensor):
        device = pred_coords.device
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    # Define skeleton connections (example for a 21-joint hand model)
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),     # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),# Ring
        (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]

    # Convert to root-relative coordinates.
    pred_root = pred_coords[root_index]
    gt_root = gt_coords[root_index]
    pred_rel = pred_coords - pred_root  # Predicted in root-relative space.
    gt_rel = gt_coords - gt_root        # Ground truth in root-relative space.

    # Optional debug plot for original root-relative predictions.
    if debug:
        error_original = np.linalg.norm(pred_rel - gt_rel, axis=1)
        mean_error_original = np.mean(error_original)
        max_error_original = np.max(error_original)

        fig1 = plt.figure(figsize=(12, 6))
        ax1 = fig1.add_subplot(121, projection='3d')
        ax1.scatter(pred_rel[:, 0], pred_rel[:, 1], pred_rel[:, 2],
                    c='r', marker='x', s=50, label='Predictions')
        for start, end in skeleton:
            ax1.plot([pred_rel[start, 0], pred_rel[end, 0]],
                     [pred_rel[start, 1], pred_rel[end, 1]],
                     [pred_rel[start, 2], pred_rel[end, 2]],
                     c='r', linestyle='--', linewidth=2)
        ax1.scatter(gt_rel[:, 0], gt_rel[:, 1], gt_rel[:, 2],
                    c='g', marker='o', s=50, label='Ground Truth')
        for start, end in skeleton:
            ax1.plot([gt_rel[start, 0], gt_rel[end, 0]],
                     [gt_rel[start, 1], gt_rel[end, 1]],
                     [gt_rel[start, 2], gt_rel[end, 2]],
                     c='g', linestyle='-', linewidth=2)
        ax1.set_title("Original Root-Relative Predictions\n"
                      "Mean Error: {:.4f}, Max Error: {:.4f}".format(mean_error_original, max_error_original))
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()

    # ---------------------------
    # Adjust predictions: Replace specified joints in the predicted values with ground truth values.
    adjust_idxs = [4, 8, 12, 16, 20]
    adjusted_rel = pred_rel.copy()
    for idx in adjust_idxs:
        adjusted_rel[idx] = gt_rel[idx]

    if debug:
        error_adjusted = np.linalg.norm(adjusted_rel - gt_rel, axis=1)
        mean_error_adjusted = np.mean(error_adjusted)
        max_error_adjusted = np.max(error_adjusted)

        ax2 = fig1.add_subplot(122, projection='3d')
        ax2.scatter(adjusted_rel[:, 0], adjusted_rel[:, 1], adjusted_rel[:, 2],
                    c='r', marker='x', s=50, label='Adjusted Predictions')
        for start, end in skeleton:
            ax2.plot([adjusted_rel[start, 0], adjusted_rel[end, 0]],
                     [adjusted_rel[start, 1], adjusted_rel[end, 1]],
                     [adjusted_rel[start, 2], adjusted_rel[end, 2]],
                     c='r', linestyle='--', linewidth=2)
        ax2.scatter(gt_rel[:, 0], gt_rel[:, 1], gt_rel[:, 2],
                    c='g', marker='o', s=50, label='Ground Truth')
        for start, end in skeleton:
            ax2.plot([gt_rel[start, 0], gt_rel[end, 0]],
                     [gt_rel[start, 1], gt_rel[end, 1]],
                     [gt_rel[start, 2], gt_rel[end, 2]],
                     c='g', linestyle='-', linewidth=2)
        ax2.set_title("Adjusted Root-Relative Predictions\n"
                      "Mean Error: {:.4f}, Max Error: {:.4f}".format(mean_error_adjusted, max_error_adjusted))
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Undo the root-relative transformation by adding back the predicted root.
    adjusted_pred_abs = adjusted_rel + pred_root

    # Optional debug plot for absolute coordinates.
    if debug:
        gt_abs = gt_rel + gt_root
        fig2 = plt.figure(figsize=(8, 6))
        ax3 = fig2.add_subplot(111, projection='3d')
        ax3.scatter(adjusted_pred_abs[:, 0], adjusted_pred_abs[:, 1], adjusted_pred_abs[:, 2],
                    c='r', marker='x', s=50, label='Adjusted Predictions (Absolute)')
        for start, end in skeleton:
            ax3.plot([adjusted_pred_abs[start, 0], adjusted_pred_abs[end, 0]],
                     [adjusted_pred_abs[start, 1], adjusted_pred_abs[end, 1]],
                     [adjusted_pred_abs[start, 2], adjusted_pred_abs[end, 2]],
                     c='r', linestyle='--', linewidth=2)
        ax3.scatter(gt_abs[:, 0], gt_abs[:, 1], gt_abs[:, 2],
                    c='g', marker='o', s=50, label='Ground Truth (Absolute)')
        for start, end in skeleton:
            ax3.plot([gt_abs[start, 0], gt_abs[end, 0]],
                     [gt_abs[start, 1], gt_abs[end, 1]],
                     [gt_abs[start, 2], gt_abs[end, 2]],
                     c='g', linestyle='-', linewidth=2)
        ax3.set_title("Adjusted Predictions in Absolute Coordinates")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.legend()
        plt.tight_layout()
        plt.show()

    # Convert the adjusted prediction back to a torch tensor on the original device.
    adjusted_pred_abs_tensor = torch.tensor(adjusted_pred_abs).to(device)
    return adjusted_pred_abs_tensor



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
        #visualize_joints_comparison(joints[0], joints_gt=gt_joints[0], save_path='./debug_img/joints_comparison.png')
        
        
        #marker information addition
        if gt_joints is not None:
            adjusted_joints_list = []
            for i in range(joints.shape[0]):
                adjusted = visualize_hand_predictions(joints[i], gt_joints[i], root_index=0)
                adjusted_joints_list.append(adjusted)
            # Replace joints with the adjusted version
            joints = torch.stack(adjusted_joints_list, dim=0)
        

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
            output_dict = self.infer(image, gt_uv=target["uv"], gt_joints=target["xyz"])
        else:
            output_dict = self.infer(image)
            
        if self.training:
            # In training mode, require a target to compute losses.
            assert target is not None, "Target is required for training"
            loss_dict = self._cal_single_hand_losses(output_dict, target)
            return loss_dict
        else:
            # In evaluation mode, if target is provided and has the necessary keys, you can compute eval metrics.
            # Otherwise, simply return the output dictionary.
            if target is not None and 'uv_valid' in target:
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