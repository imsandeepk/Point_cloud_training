from data_loader import PointCloudDataset
from torch.utils.data import DataLoader
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config("/Users/sandeep/DOCS/Point Cloud Training/config/config.yaml")

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






