import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.point_net_base.model import PointNetSegmentation
import open3d as o3d


# Load the trained model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load a model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")

# Function for inference
def infer_point_cloud(model, point_cloud, device):
    """
    Perform inference on a single point cloud.
    
    Args:
        model: Trained PointNetSegmentation model.
        point_cloud: Input point cloud of shape (num_points, 3).
        device: Device to run inference on.

    Returns:
        Predicted labels of shape (num_points,).
    """
    model.eval()
    with torch.no_grad():
        # Prepare the input
        points = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        points = points.permute(0, 2, 1).to(device)  # Shape: (1, 3, num_points)

        # Forward pass
        outputs = model(points)  # Shape: (1, num_points, num_classes)
        predictions = torch.argmax(outputs, dim=2)  # Shape: (1, num_points)

    return predictions.squeeze(0).cpu().numpy()  # Convert to NumPy array

# Configuration
checkpoint_path = "checkpoints/best_model.pth"  # Path to your saved checkpoint
num_classes = 2  # Number of classes (binary segmentation)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = PointNetSegmentation(num_classes=num_classes).to(device)

# Load the trained checkpoint
load_checkpoint(checkpoint_path, model)

# Example point cloud (replace with your input point cloud)
# Shape: (num_points, 3)
pcd_path = "/Users/sandeep/DOCS/Point Cloud Training/data/test/point_clouds/point_cloud_129.pcd"
point_cloud = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(point_cloud.points, dtype=np.float32)
indices = np.random.choice(points.shape[0], 6000, replace=False)
if points.shape[0] > 6000:
    indices = np.random.choice(points.shape[0], 6000, replace=False)
else:  # If fewer points, pad with duplicates
    indices = np.random.choice(points.shape[0], 6000, replace=True)

points = points[indices]


# Run inference
predicted_labels = infer_point_cloud(model, points, device)


np.savetxt("predicted_labels.txt", predicted_labels)
o3d.io.write_point_cloud("predicted_labels.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)), write_ascii=True)
# Output results
print(f"Predicted labels:\n{predicted_labels}")
print(f"Unique labels: {np.unique(predicted_labels)}")
