import yaml
import torch
import argparse

from torch.utils.data import ConcatDataset
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from dataset.dataset_helper import get_dataset, show_sample_images
from model.classifier_utils import run
from config.resolve_config import resolve_config

def main(config_file_path='config/config.yaml', augment=False, wandb_log=False):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    config = resolve_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset without augmentations
    dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=False)
    # show_sample_images(dataset, 15)
    combined_dataset = dataset
    if augment:
        augmented_dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=True)
        # show_sample_images(augmented_dataset, 15)
        # # Uncomment if you want to save aug
        # save_augmented_dataset(augmented_dataset, config['data_dir'])
        combined_dataset = ConcatDataset([dataset, augmented_dataset])

    run('baseline', combined_dataset, config, device, wandb_log=wandb_log)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Script")
    parser.add_argument('--augment', action='store_true', help='Flag to enable augmentation')
    parser.add_argument('--wandb', action='store_true', help='Flag to enable logging to wandb')

    args = parser.parse_args()
    print('args: ', args)
    main(augment=args.augment, wandb_log=args.wandb)