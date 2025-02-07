import json
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import albumentations as A
from typing import List, Dict
from itertools import cycle
from cfg import _CONFIG
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transforms import GetRandomScaleRotation, MeshAffine, RandomHorizontalFlip, \
            get_points_center_scale, RandomChannelNoise, BBoxCenterJitter, MeshPerspectiveTransform
import torch

DATA_CFG = _CONFIG["DATA"]
IMAGE_SHAPE: List = DATA_CFG["IMAGE_SHAPE"][:2]
NORMALIZE_3D_GT = DATA_CFG['NORMALIZE_3D_GT']
AUG_CFG: Dict = DATA_CFG["AUG"]
ROOT_INDEX = DATA_CFG['ROOT_INDEX']


def read_info(img_path):
    info_path = img_path.replace('.jpg', '.json')
    with open(info_path) as f:
        info = json.load(f)
    return info

with open(DATA_CFG['JSON_DIR']) as f:
    all_image_info = json.load(f)
all_info = []

dataset_dir = DATA_CFG['DATASET_DIR']

# Loop through the images and update paths
for image_path in tqdm(all_image_info):
    # Update image path to have the correct base directory
    updated_image_path = image_path.replace('/data/myHAND/training/rgb/', dataset_dir)

    # Read the corresponding JSON file using the updated path
    info = read_info(updated_image_path)
    info['image_path'] = updated_image_path
    all_info.append(info)

class HandDataset(Dataset):
    def __init__(self, all_info):
        super().__init__()

        self.init_aug_funcs()
        self.all_info = all_info

    def __len__(self):
        return len(self.all_info)
    
    def init_aug_funcs(self):
        self.random_channel_noise = RandomChannelNoise(**AUG_CFG['RandomChannelNoise'])
        self.random_bright = A.RandomBrightnessContrast(**AUG_CFG["RandomBrightnessContrastMap"])            
        self.random_flip = RandomHorizontalFlip(**AUG_CFG["RandomHorizontalFlip"])
        self.bbox_center_jitter = BBoxCenterJitter(**AUG_CFG["BBoxCenterJitter"])
        self.get_random_scale_rotation = GetRandomScaleRotation(**AUG_CFG["GetRandomScaleRotation"])
        self.mesh_affine = MeshAffine(IMAGE_SHAPE[0])
        self.mesh_perspective_trans = MeshPerspectiveTransform(IMAGE_SHAPE[0])
        
        self.root_index = ROOT_INDEX

    def generate_heatmap(self, height, width, normalized_coord, sigma=2):
        """
        Generates a Gaussian heatmap for a single normalized coordinate.
    
        Args:
            height (int): Height of the heatmap (typically the image height).
            width (int): Width of the heatmap (typically the image width).
            normalized_coord (np.array): [x, y] in [0,1] representing the fingertip position.
            sigma (float): Standard deviation of the Gaussian.
        
        Returns:
            heatmap (np.array): A heatmap of shape [height, width] with values in [0,1].
        """
        # Convert normalized coordinates to pixel coordinates
        cx = normalized_coord[0] * width
        cy = normalized_coord[1] * height

        # Create a mesh grid of (x,y) coordinates
        y_grid, x_grid = np.ogrid[0:height, 0:width]
        heatmap = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        return heatmap.astype(np.float32)

    def overlay_heatmaps_on_image(self, img, heatmaps, alpha=0.5):
        """
        Overlays the provided heatmaps onto the image.
    
        Args:
            img (np.array): The original image (in BGR or RGB format; adjust accordingly).
            heatmaps (np.array): Heatmaps of shape [N, H, W] (values assumed to be in [0,1]).
            alpha (float): The blending factor.
    
        Returns:
            overlay (np.array): The image with heatmaps overlayed.
        """
        # Combine the heatmaps; you can either sum them or take the maximum.
        combined_heatmap = np.sum(heatmaps, axis=0)
        combined_heatmap = np.clip(combined_heatmap, 0, 1)
    
        # Convert combined heatmap to 8-bit (0-255)
        combined_heatmap = (combined_heatmap * 255).astype(np.uint8)
    
        # Apply a color map (e.g., COLORMAP_JET) to get a colored heatmap.
        colored_heatmap = cv2.applyColorMap(combined_heatmap, cv2.COLORMAP_JET)
    
        # Make sure the image is in the same scale (0-255) and type.
        if img.dtype != np.uint8:
            img_vis = (img * 255).astype(np.uint8)
        else:
            img_vis = img.copy()
    
        # Blend the original image and the colored heatmap.
        overlay = cv2.addWeighted(img_vis, 1 - alpha, colored_heatmap, alpha, 0)
       
        # Create the folder if it doesn't exist.
        overlay_folder = "overlayimages"
        if not os.path.exists(overlay_folder):
            os.makedirs(overlay_folder)

        # Create a unique filename. For example, you might use a timestamp.
        import time
        filename = os.path.join(overlay_folder, f"debug_overlay_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(filename, overlay)
        print(f"Debug overlay saved as {filename}")


        return overlay
    
    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img
    
    def __getitem__(self, index):
        data_info = self.all_info[index]
        img = self.read_image(data_info['image_path'])
        # keypoints2d = np.array(data_info['uv'], dtype=np.float32)
        keypoints3d = np.array(data_info['xyz'], dtype=np.float32)
        K = np.array(data_info['K'], dtype=np.float32)
        
        proj_points = (K @ keypoints3d.T).T
        keypoints2d = proj_points[:, :2] / (proj_points[:, 2:] + 1e-7)
        
        vertices = np.array(data_info['vertices']).astype('float32')

        h, w = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            

        uv_norm = keypoints2d.copy()
        uv_norm[:, 0] /= w   
        uv_norm[:, 1] /= h

        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

        valid_points = [keypoints2d[i] for i in range(len(keypoints2d)) if coord_valid[i]==1]
        
        points = np.array(valid_points)
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord)/2
        scale = max_coord - min_coord

         # Store the original UV for training
        original_uv = keypoints2d.copy()  # This remains unchanged for loss computation
        original_uv[:, 0] /= IMAGE_SHAPE[0]
        original_uv[:, 1] /= IMAGE_SHAPE[1]


        results = {
            "img": img,
            "keypoints2d": keypoints2d,
            "keypoints3d": keypoints3d,
            "vertices": vertices,
            
            "center": center,
            "scale": scale,
            "K": K,
        }
        
         # Apply augmentations
        results = self.bbox_center_jitter(results)
        results = self.get_random_scale_rotation(results)
        results = self.mesh_perspective_trans(results)
    
        # Compute root-relative 3D keypoints
        root_point = results['keypoints3d'][self.root_index].copy()
        results['keypoints3d'] = results['keypoints3d'] - root_point[None, :]
        results['vertices'] = results['vertices'] - root_point[None, :]
    
        hand_img_len = IMAGE_SHAPE[0]
        root_depth = root_point[2]
        hand_world_len = 0.2
        fx = results['K'][0][0]
        fy = results['K'][1][1]
        camare_relative_k = np.sqrt(fx * fy * (hand_world_len**2) / (hand_img_len**2))
        gamma = root_depth / camare_relative_k
    
        # Further augmentations: flip, noise, brightness
        results = self.random_flip(results)
        results = self.random_channel_noise(results)
        results['img'] = self.random_bright(image=results['img'])['image']
    
        final_img = results['img']
        final_h, final_w = final_img.shape[:2]
    
        # --- For Marker Processing Only ---
        # Create a separate copy for marker normalization.
        uv_for_markers = results["keypoints2d"].copy()  # Start with the original UV
        # Normalize using final image dimensions (only for marker generation)
        uv_for_markers[:, 0] /= final_w
        uv_for_markers[:, 1] /= final_h
    
        # Optionally compute a valid region on uv_for_markers if needed:
        trans_coord_valid = (uv_for_markers > 0).astype("float32") * (uv_for_markers < 1).astype("float32")
        trans_coord_valid = trans_coord_valid[:, 0] * trans_coord_valid[:, 1]
        trans_coord_valid *= coord_valid  # if required
    
        # Generate marker heatmaps using uv_for_markers
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertips_uv = uv_for_markers[fingertip_indices, :]  # Only used for markers
        marker_heatmaps = []
        for uv in fingertips_uv:
            hm = self.generate_heatmap(final_h, final_w, uv, sigma=2)
            marker_heatmaps.append(hm)

        marker_heatmaps = np.stack(marker_heatmaps, axis=0)
        marker_heatmaps = torch.from_numpy(marker_heatmaps).float()
        # Optionally, debug the overlay (commented out during training)
        #debug_overlay = self.overlay_heatmaps_on_image(final_img, marker_heatmaps.numpy(), alpha=0.5)
        # --- End Marker Processing ---
    
        xyz = results["keypoints3d"]
        if NORMALIZE_3D_GT:
            joints_bone_len = np.sqrt(((xyz[0:1] - xyz[9:10])**2).sum(axis=-1, keepdims=True) + 1e-8)
            xyz = xyz / joints_bone_len
    
        xyz_valid = 1
        if trans_coord_valid[9] == 0 and trans_coord_valid[0] == 0:
            xyz_valid = 0
    
        img_final = results['img']
        img_final = np.transpose(img_final, (2, 0, 1))
    
        data = {
            "img": img_final,
            "uv": original_uv,  # Return the original UV for training!
            "xyz": xyz,
            "vertices": results['vertices'],
            "uv_valid": trans_coord_valid,
            "gamma": gamma,
            "xyz_valid": xyz_valid,
            "marker_heatmaps": marker_heatmaps
        }
    
        return data

def build_train_loader(batch_size):
	dataset = HandDataset(all_info)
	sampler = RandomSampler(dataset, replacement=True)
	dataloader = (DataLoader(dataset, batch_size=batch_size, sampler=sampler))
	return iter(dataloader)

# if __name__ == "__main__":
#     train_loader = build_train_loader(_CONFIG['TRAIN']['DATALOADER']['MINIBATCH_SIZE_PER_DIVICE'])
#     batch = next(train_loader)
#     with open('batch_data.pkl', 'rb') as f:
#         pickle.dump(batch, f)
#     from IPython import embed 
#     embed()
#     exit()