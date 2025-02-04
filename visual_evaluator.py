import json
import numpy as np
import matplotlib.pyplot as plt

# Define the file path
json_path = "/home/fabian/simpleHand/train_log/models_fastvit_ma36/evals/FreiHand/epoch_15.json"

# Load JSON data
with open(json_path, "r") as f:
    data = json.load(f)

def make_root(keypoints3d, root_index=0):
    root_point = keypoints3d[root_index].copy()
    return keypoints3d - root_point[None, :]

def visualize_hand_predictions(pred_coords, gt_coords, root_index=0):
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),    
        (0, 5), (5, 6), (6, 7), (7, 8),    
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # Calculate errors FIRST in absolute coordinates (matches loss calculation)
    absolute_error = np.linalg.norm(pred_coords - gt_coords, axis=1)
    l1_error = np.mean(np.abs(pred_coords - gt_coords))  # Directly matches loss
    
    # For visualization only: align both to GT root
    root_point = gt_coords[root_index].copy()
    gt_rel = gt_coords - root_point
    pred_rel = pred_coords - root_point
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth
    ax.scatter(gt_rel[:, 0], gt_rel[:, 1], gt_rel[:, 2], c='g', label='GT', s=50)
    for start, end in skeleton:
        ax.plot([gt_rel[start,0], gt_rel[end,0]],
                [gt_rel[start,1], gt_rel[end,1]],
                [gt_rel[start,2], gt_rel[end,2]], c='g', lw=2)
    
    # Plot predictions
    ax.scatter(pred_rel[:, 0], pred_rel[:, 1], pred_rel[:, 2], c='r', label='Pred', s=50)
    for start, end in skeleton:
        ax.plot([pred_rel[start,0], pred_rel[end,0]],
                [pred_rel[start,1], pred_rel[end,1]],
                [pred_rel[start,2], pred_rel[end,2]], c='r', linestyle='--', lw=2)
    
    # Display absolute errors (matches loss calculation)
    for i in range(len(gt_coords)):
        ax.text(gt_rel[i,0], gt_rel[i,1], gt_rel[i,2], 
                f'{absolute_error[i]:.1f}', color='blue', fontsize=8)
    
    ax.set_title(f"Hand Prediction vs GT\nL1 Error: {l1_error:.2f} mm")
    ax.legend()
    plt.show()

#Loop through the dataset and visualize
for i, sample in enumerate(data):
    pred_xyz = np.array(sample["pred_xyz"])
    gt_xyz = np.array(sample["xyz"])  # Keep raw coordinates
    
    gt_xyz = make_root(gt_xyz)
    print(f"Visualizing sample {i+1}/{len(data)}")
    visualize_hand_predictions(pred_xyz, gt_xyz) 
