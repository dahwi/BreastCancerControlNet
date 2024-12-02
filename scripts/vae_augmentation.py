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

def main(config_file_path):

    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Argument parsing
    parser = argparse.ArgumentParser(description="VAE Augmentation: Script to train, generate, or evaluate accuracy.")

    parser.add_argument('--train', action='store_true', help="Run the training process.")
    parser.add_argument('--generate', action='store_true', help="Generate samples.")
    parser.add_argument('--accuracy', action='store_true', help="Evaluate accuracy.")

    args = parser.parse_args()

    # Train the VAE
    if args.train:
        dataset = get_dataset_vae(config['data_dir'])
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        train(dataloader, config['latent_dim'], config['num_classes'], config['input_channels'], config['num_epochs'])
    
    # Generate samples
    elif args.generate:
        vae = ClassConditionedVAE(config['input_channels'], config['latent_dim'], config['num_classes'])
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'vae.pth')
        vae.load_state_dict(torch.load(model_path))

        for i in range(config['num_classes']):
            sample(vae, i, int(len(dataset) / config['num_classes']), config['latent_dim'], config['num_classes'])

    # Evaluate accuracy
    elif args.accuracy:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Original dataset
        dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=False)
        # Load dataset with augmentations
        augmented_dataset = get_dataset(UltrasoundBreastDataset, config['augmented_dir'], 224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=True)
        combined_dataset = ConcatDataset([dataset, augmented_dataset])
        
        print("length of dataset: ", len(dataset))
        print("length of augmented dataset: ", len(augmented_dataset))
        print("length of combined dataset: ", len(combined_dataset))
        print(combined_dataset[0][0].shape)

        # run('baseline', combined_dataset, config, device)

    else:
        print("Please specify a valid flag: --train, --generate, or --accuracy.")
    

if __name__ == '__main__':
    config_file_path = 'config/vae.yaml'
    main(config_file_path)