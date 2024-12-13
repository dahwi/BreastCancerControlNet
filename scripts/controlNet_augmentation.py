import torch
import yaml

from dataset.dataset_helper import get_dataset,filter_dataset_by_label
from dataset.ultrasound_breast_dataset import UltrasoundBreastDataset
from model.control_net_utils import fine_tune


from dataset.dataset_helper import transformToGreyScale
from torchvision.transforms import ToPILImage
from PIL import Image
import os

def main(config_file_path):
    # Load configuration
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_map = {"benign": 0, "normal": 1, "malignant": 2}
    for c in ["malignant"]: #, "normal", "benign"]:
        dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 512, 512, [0.5], [0.5], augment=False)
        mask_dataset = get_dataset(UltrasoundBreastDataset, config['data_dir'], 512, 512, [0.5], [0.5], augment=True, mask=True)
        print(f'read full dataset: size of {len(dataset)}')
        dataset = filter_dataset_by_label(dataset, label_map[c])
        mask_dataset = filter_dataset_by_label(mask_dataset, label_map[c])
        print(f'dataset class: {c}, size: {len(dataset)}')
        print(f'mask dataset class: {c}, size: {len(dataset)}')
        fine_tune(config, dataset, mask_dataset, c, device, epochs=10, wandb_log=False)

        #######
        output_dir = config['controlnet_ft_augmented_dir']
        os.makedirs(output_dir, exist_ok=True)
        for j in range(10):
            # random_idx = torch.randint(len(dataset), (1,)).item()
            org_image, _ = dataset[j]
            mask_image, _ = mask_dataset[j]

            image_greyscale = transformToGreyScale(org_image, mean=[0.5], std=[0.5])
            org_image_pil = ToPILImage()(image_greyscale)

            mask_image_greyscale = transformToGreyScale(mask_image, mean=[0.5], std=[0.5])
            org_mask_image_pil = ToPILImage()(mask_image_greyscale)

            # Combine images side by side
            widths, heights = zip(*(img.size for img in [org_image_pil, org_mask_image_pil]))
            total_width = sum(widths)
            max_height = max(heights)

            combined_image = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in [org_image_pil, org_mask_image_pil]:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.size[0]

            # Save the combined image
            combined_image.save(f"{output_dir}/{c}_{j}.png")
        #######
if __name__ == '__main__':
    config_file_path = 'config/config.yaml'
    main(config_file_path)


