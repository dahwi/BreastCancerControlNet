import torch
import yaml

from dataset.dataset_helper import get_dataset
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from model.control_net_utils import fine_tune

def main(config_file_path):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 512, 512, [0.5], [0.5], False, True)
    fine_tune(config, dataset, device, "benign", 5, True)


if __name__ == '__main__':
    config_file_path = 'config/config.yaml'
    main(config_file_path)


