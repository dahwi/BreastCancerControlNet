import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from model.vae import ClassConditionedVAE
from model.vae_utils import train, sample, get_dataset_vae
from dataset.dataset_helper import get_dataset, show_sample_images
from model.classifier_utils import run
from config.resolve_config import resolve_config

def main(config_file_path, args):

    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    config = resolve_config(config)
    
    vae_config = config['vae']

    dataset = get_dataset_vae(config['data_dir'])

    # Train the VAE
    if args.train:
        dataloader = DataLoader(dataset, batch_size=vae_config['batch_size'], shuffle=True)
        train(config, dataloader, vae_config['latent_dim'], vae_config['num_classes'], vae_config['input_channels'], vae_config['num_epochs'])
    
    # Generate samples
    elif args.generate:
        vae = ClassConditionedVAE(vae_config['input_channels'], vae_config['latent_dim'], vae_config['num_classes'])
        model_path = os.path.join(config["output_dir"], 'vae.pth')
        vae.load_state_dict(torch.load(model_path))

        for i in range(vae_config['num_classes']):
            sample(config, vae, i, int(len(dataset) / vae_config['num_classes']), vae_config['latent_dim'], vae_config['num_classes'])

    # Evaluate accuracy
    elif args.accuracy:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Original dataset
        dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 256, 256, [0.5], [0.5], augment=False)
        # Load dataset with augmentations
        augmented_dataset = get_dataset(UltrasoundBreastDataset, config['vae_augmented_dir'], 256, 256, [0.5], [0.5], augment=False)
        combined_dataset = ConcatDataset([dataset, augmented_dataset])

        run('vae', combined_dataset, config, device, wandb_log=args.wandb)

    else:
        print("Please specify a valid flag: --train, --generate, or --accuracy.")
    

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="VAE Augmentation: Script to train, generate, or evaluate accuracy.")

    parser.add_argument('--wandb', action='store_true', help='Flag to enable logging to wandb')
    parser.add_argument('--train', action='store_true', help="Run the training process.")
    parser.add_argument('--generate', action='store_true', help="Generate samples.")
    parser.add_argument('--accuracy', action='store_true', help="Evaluate accuracy.")

    args = parser.parse_args()
    main(config_file_path="config/config.yaml", args=args)