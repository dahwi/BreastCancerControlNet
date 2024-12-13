import torch
import yaml
import argparse

from dataset.dataset_helper import get_dataset, filter_dataset_by_label
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from model.stable_diffusion_utils import fine_tune
from torch.utils.data import ConcatDataset
from model.classifier_utils import run

def main(config_file_path='config/config.yaml', finetune=False):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if finetune:
        label_map = {"benign": 0, "normal": 1, "malignant": 2}
        for c in ["benign", "normal", "malignant"]:
            dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 256, 256, [0.5], [0.5], augment=False)
            dataset = filter_dataset_by_label(dataset, label_map[c])
            print(f"Finetuning for {c}")
            fine_tune(config, dataset, device, c, epochs=5, wandb_log=True)
            print(f"Finetuning complete and generated images for {c}")

    dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 256, 256, [0.5], [0.5], augment=False)
    augmented_dataset = get_dataset(UltrasoundBreastDataset, config[f'sd_ft_augmented_dir'], 256, 256, [0.5], [0.5], augment=False)

    combined_dataset = ConcatDataset([dataset, augmented_dataset])

    run('stable_diffusion', combined_dataset, config, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StableDiffusion Generation Script")
    parser.add_argument('config_file_path', type=str, help='Path to the configuration file')
    parser.add_argument('--finetune', action='store_true', help='Flag to enable fine-tuning')

    args = parser.parse_args()
    main(args.config_file_path, args.finetune)