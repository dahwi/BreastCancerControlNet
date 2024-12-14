import torch
import yaml
import argparse

from dataset.dataset_helper import get_dataset,filter_dataset_by_label
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from model.control_net_utils import fine_tune
from torch.utils.data import ConcatDataset
from model.classifier_utils import run
from config.resolve_config import resolve_config


def main(config_file_path='config/config.yaml', finetune=False, wandb_log=False):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    config = resolve_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

     
    if finetune:
        label_map = {"benign": 0, "normal": 1, "malignant": 2}
        for c in ["malignant", "normal", "benign"]:
            dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 512, 512, [0.5], [0.5], augment=False)
            mask_dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 512, 512, [0.5], [0.5], augment=True, mask=True)
            dataset = filter_dataset_by_label(dataset, label_map[c])
            mask_dataset = filter_dataset_by_label(mask_dataset, label_map[c])
            fine_tune(config, dataset, mask_dataset, c, device, epochs=5, wandb_log=True)
    

    dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 256, 256, [0.5], [0.5], augment=False)
    augmented_dataset = get_dataset(UltrasoundBreastDataset, config[f'controlnet_ft_augmented_dir'], 256, 256, [0.5], [0.5], augment=False)

    combined_dataset = ConcatDataset([dataset, augmented_dataset])

    run('control_net', combined_dataset, config, device, wandb_log=wandb_log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ControlNet Generation Script")
    parser.add_argument('--finetune', action='store_true', help='Flag to enable fine-tuning')
    parser.add_argument('--wandb', action='store_true', help='Flag to enable logging to wandb')

    args = parser.parse_args()
    print('args: ', args)
    main(finetune=args.finetune, wandb_log=args.wandb)

