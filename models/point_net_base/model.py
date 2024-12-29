import torch
import torch.nn as nn

class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)  # Input: (batch_size, 3, num_points)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Conv1d(256, num_classes, 1)  # Output per point

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # Apply convolutional layers
        x = self.relu(self.conv1(x))  # Shape: (batch_size, 64, num_points)
        x = self.relu(self.conv2(x))  # Shape: (batch_size, 128, num_points)
        global_features = self.relu(self.conv3(x))  # Shape: (batch_size, 1024, num_points)

        # Global pooling
        global_features_pooled = self.maxpool(global_features)  # Shape: (batch_size, 1024, 1)
        global_features_pooled = global_features_pooled.view(global_features_pooled.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(global_features_pooled))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = x.unsqueeze(2).repeat(1, 1, global_features.shape[2])  # Broadcast global features

        # Combine local and global features for point-wise classification
        pointwise_features = torch.cat([global_features, x], dim=1)
        pointwise_output = self.fc3(pointwise_features)  # Shape: (batch_size, num_classes, num_points)

        return pointwise_output.permute(0, 2, 1)  # Shape: (batch_size, num_points, num_classes)
