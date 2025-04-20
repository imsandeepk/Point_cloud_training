# Dental Scan Segmentation using PointNet

This repository contains an implementation of PointNet for segmenting dental scans into gums and teeth. The model takes 3D point cloud data as input and produces pointwise segmentation labels.

## Overview

This project uses PointNet architecture to segment 3D point clouds of dental scans. The model identifies each point as either tooth or gum tissue, enabling automatic segmentation of dental structures.

### Features

- PointNet-based semantic segmentation
- Support for 3D point cloud processing
- Training and evaluation scripts
- Pre-trained model checkpoints
- Inference script for segmenting new dental scans

## Project Structure

```
Point_cloud_training/
├── model/
│   └── pointnet_segmentation.py     # PointNet model implementation
├── utils/
│   ├── data_loader                  # Data loading and preprocessing
│   └── train.py                     # Visualization utilities
├── inference                        # Inference script for new point clouds
├── checkpoints/                     # Saved model checkpoints
│   └── best_model.pth               # Pre-trained model
├── config/
│   └── config.yaml                  # Configuration files
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```
Dental Scan Segmentation using PointNet
This repository contains an implementation of PointNet for segmenting dental scans into gums and teeth. The model takes 3D point cloud data as input and produces pointwise segmentation labels.
Overview
This project uses PointNet architecture to segment 3D point clouds of dental scans. The model identifies each point as either tooth or gum tissue, enabling automatic segmentation of dental structures.
Features

PointNet-based semantic segmentation
Support for 3D point cloud processing
Training and evaluation scripts
Pre-trained model checkpoints
Inference script for segmenting new dental scans

Project Structure
Point_cloud_training/
├── model/
│   └── pointnet_segmentation.py     # PointNet model implementation
├── utils/
│   ├── data_loader                  # Data loading and preprocessing
│   └── train.py                     # Training utilities
├── inference/                       # Inference script for new point clouds
│   └── infer.py                     # Main inference script
├── checkpoints/                     # Saved model checkpoints
│   └── best_model.pth               # Pre-trained model
├── config/
│   └── config.yaml                  # Configuration files
├── data/                            # Dataset directory (after downloading)
│   ├── train/                       # Training data
│   └── test/                        # Test data
├── requirements.txt                 # Dependencies
└── README.md                        # This file
Dataset
Downloading the Data
The dataset contains 3D point clouds of dental scans with segmentation labels. To download the dataset:

Direct download:

Access the dataset from Google Drive
Download the ZIP file and extract it to the data/ directory in your project folder


## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/imsandeepk/Point_cloud_training
   cd Point_cloud_training
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Configuration

Modify the configuration file at `config/config.yaml` to adjust model parameters, training settings, and data paths.

Example configuration:
```yaml
model:
  num_classes: 2  # teeth and gums
  feature_transform: true
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: 'adam'
  
data:
  train_dir: 'data/train'
  test_dir: 'data/test'
  num_points: 2048
```

### Training

To train the model:

```bash
python train.py --config config/config.yaml
```

Options:
- `--config`: Path to configuration file
- `--resume`: Path to checkpoint to resume training (optional)
- `--gpu`: GPU ID to use (default: 0)


### Inference

To segment a new dental scan:

```bash
python inference.py --input your_scan.ply --output segmented_scan.ply --checkpoint checkpoints/pointnet_dental_seg.pth
```

Options:
- `--input`: Path to input point cloud file (.ply, .npy, .xyz formats supported)
- `--output`: Path to save segmented point cloud
- `--checkpoint`: Path to model checkpoint
- `--visualize`: Flag to visualize results

## Input Data Format

The model expects point cloud data in one of these formats:
- `.pcd` files with at least x, y, z coordinates

For training, label information should be included as the 4th column in the data files.

## Inference Example

Here's how to run inference on a new scan:

 - Open inference/infer.py
 - Change the pcd_path to your data
 - run the infer.py


## Model Details

The implemented PointNet architecture includes:
- Input transformation network (T-Net)
- Feature transformation network
- Global and local feature extraction
- Segmentation head for pointwise classification

## Results

The model achieves approximately 88.6% segmentation accuracy on our test dataset, with an IoU score of Y for teeth and Z for gums.

