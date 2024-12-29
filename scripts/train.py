from data_loader.data_loader import PointCloudDataset
from torch.utils.data import DataLoader
import yaml
import torch
import sys
import os

import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.point_net_base.model import PointNetSegmentation

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config("/Users/sandeep/DOCS/Point Cloud Training/config/config.yaml")


checkpoint_dir = config['training']['checkpoint_dir']
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


dataset = PointCloudDataset(
    point_cloud_dir=config['dataset_train']['point_cloud_dir'],
    label_dir=config['dataset_train']['label_dir'],
    num_points=config['dataset_train']['num_points']
)
dataloader = DataLoader(
    dataset,
    batch_size=config['dataset_train']['batch_size'],
    shuffle=config['dataset_train']['shuffle']
)

print("Data loaded successfully!")


# Training loop
import time
import torch
import torch.nn as nn
from datetime import timedelta

# Training setup
num_classes = 2  # Binary segmentation (0 or 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PointNetSegmentation(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

best_loss = float('inf')  # To track the best loss
total_train_time = 0  # Total time taken for training

for epoch in range(config['training']['num_epochs']):
    print(f"\nStart of Epoch {epoch+1}/{config['training']['num_epochs']}")
    model.train()
    total_loss = 0.0
    batch_loss = 0.0
    start_time_epoch = time.time()  # Track start time for the epoch
    num_batches = len(dataloader)

    for batch_idx, (points, labels) in enumerate(dataloader):
        start_time_batch = time.time()  # Track start time for the batch

        points = points.to(device).permute(0, 2, 1)  # Shape: (batch_size, 3, num_points)
        labels = labels.to(device)                  # Shape: (batch_size, num_points)

        # Forward pass
        outputs = model(points)                     # Shape: (batch_size, num_points, num_classes)

        # Compute loss
        loss = criterion(outputs.view(-1, num_classes), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update batch and total loss
        batch_loss = loss.item()
        total_loss += batch_loss

        # Calculate time for this batch
        batch_time = time.time() - start_time_batch

        if batch_idx % config['training']['log_interval'] == 0:  # Log every n batches
            eta = (num_batches - batch_idx) * batch_time
            eta_str = str(timedelta(seconds=int(eta)))  # Convert ETA to readable format
            print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], "
                  f"Batch [{batch_idx}/{num_batches}], "
                  f"Batch Loss: {batch_loss:.4f}, "
                  f"ETA: {eta_str}, "
                  f"Total Loss (Epoch): {total_loss/(batch_idx+1):.4f}")

    # Calculate epoch time and ETA for the next epoch
    epoch_time = time.time() - start_time_epoch
    total_train_time += epoch_time
    avg_epoch_loss = total_loss / num_batches
    eta_epoch = (config['training']['num_epochs'] - (epoch + 1)) * (epoch_time)
    eta_epoch_str = str(timedelta(seconds=int(eta_epoch)))

    print(f"\nEnd of Epoch {epoch+1}/{config['training']['num_epochs']}: "
          f"Avg Loss: {avg_epoch_loss:.4f}, "
          f"Epoch Time: {str(timedelta(seconds=int(epoch_time)))}")
    print(f"Estimated time for remaining epochs: {eta_epoch_str}")

    # Save checkpoint if this epoch's loss is the best so far
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        checkpoint_path = os.path.join(config['training']['checkpoint_dir'], "best_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Best model saved at epoch {epoch+1}, checkpoint saved at {checkpoint_path}")

    # Logging Total Training Time
    print(f"Total training time so far: {str(timedelta(seconds=int(total_train_time)))}")

print("Training complete!")
