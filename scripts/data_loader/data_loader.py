import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class PointCloudDataset(Dataset):
    def __init__(self, point_cloud_dir, label_dir, num_points=6000, transform=None):
        """
        Args:
            point_cloud_dir (str): Directory containing point cloud files (.pcd).
            label_dir (str): Directory containing label files (one per point cloud).
            num_points (int): Fixed number of points to sample from each point cloud.
            transform (callable, optional): Optional transform to apply to the point clouds.
        """
        self.point_cloud_dir = point_cloud_dir
        self.label_dir = label_dir
        self.num_points = num_points
        self.transform = transform

        # Ensure that point cloud and label files are sorted to align
        self.point_cloud_files = sorted(os.listdir(point_cloud_dir))
        self.label_files = sorted(os.listdir(label_dir))

        assert len(self.point_cloud_files) == len(self.label_files), \
            "Mismatch between number of point cloud and label files."

    def __len__(self):
        return len(self.point_cloud_files)

    def __getitem__(self, idx):
        # Load point cloud
        point_cloud_path = os.path.join(self.point_cloud_dir, self.point_cloud_files[idx])
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # Load labels
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        labels = np.loadtxt(label_path, dtype=np.int64)

        # Ensure point cloud and labels match in shape
        assert points.shape[0] == labels.shape[0], \
            f"Shape mismatch between points and labels in {point_cloud_path} and {label_path}."

        # Sample fixed number of points
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:  # If fewer points, pad with duplicates
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)

        points = points[indices]
        labels = labels[indices]

        # Apply optional transformations
        if self.transform:
            points = self.transform(points)

        return torch.tensor(points), torch.tensor(labels)

