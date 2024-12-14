
# **Enhancing Breast Cancer Classification with Generative AI**

## **Overview**

Breast cancer classification is critically dependent on diverse and high-quality medical imaging datasets. However, the limited size and variability of existing datasets pose significant challenges to building robust machine learning models. This project addresses these challenges by leveraging generative AI techniques to:

- Generate mammogram-like synthetic images using **Variational Autoencoders (VAEs)**, **Stable Diffusion**, and **ControlNet**.
- Fine-tune **Stable Diffusion** and **ControlNet** models to generate medically relevant synthetic images tailored for cancer classification.
- Evaluate the feasibility, strengths, and limitations of applying these generative AI models in medical imaging tasks.

## **Dataset**

We utilize the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/) containing **780 labeled images** across three categories:
- **Normal**
- **Malignant**
- **Benign**

This dataset serves as the foundation for both model training and validation in our experiments.

## **Fine-Tuning Process**

### **Stable Diffusion Fine-Tuning**
- **Objective:** Enhance a pretrained **Stable Diffusion v1.5** model using LoRA (Low-Rank Adaptation) to generate grayscale mammogram-like images.
- **Process:**
  - Images are encoded into a latent space, where noise is progressively added during training using a scheduler.
  - **Text prompts** (e.g., "malignant mammogram grayscale cross-section image") are tokenized and converted into embeddings, which guide the **denoising process**. These embeddings steer the model to produce images consistent with the semantic meaning of the prompts.
  - The model is fine-tuned so that it learns to associate noise patterns in the latent space with specific mammogram categories (normal, benign, malignant) based on the text embeddings.
  - Fine-tuning adapts the Stable Diffusion model to generate images aligned with specific medical labels.
  - The fine-tuned LoRA weights are saved, and new synthetic images are generated for downstream tasks.

### **ControlNet Fine-Tuning**
- **Objective:** Utilize a pretrained **ControlNet** model and fine-tuned **Stable Diffusion** models from above with **masked segmentation images** as control signals into the image generation pipeline.
- **Process:**
  - Combines the strengths of **ControlNet** and **Stable Diffusion** to generate anatomically accurate mammogram-like images.
  - Mask images from the dataset are used as control signals to guide the generation process.
  - The training process is really similar to **Stable Diffusion** fine-tuning except that only parameters of **ControlNet** are updated.
  - The fine-tuned weights are saved, and new synthetic images are generated for downstream tasks.

### **Outputs** (Not available in the repo due to size)
- Fine-tuned models generate synthetic mammogram-like images for each category (normal, benign, malignant) and save them for classifier training.
- Models and weights are saved for reproducibility and further experimentation.

## **Installation**

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/dahwi/BreastCancerControlNet.git
   ```

2. Navigate to the project directory:
   ```bash
   cd BreastCancerControlNet
   ```

3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure environment variables (Make sure you on the root dir of the project):
   ```bash
   export BASE_DIR=$(pwd)
   export PYTHONPATH=$BASE_DIR/BreastCancerControlNet:$PYTHONPATH
   export WANDB_KEY="your-wandb-api-key-here"
   ```

## **How to run scripts**
Following commands are run in the project root directory. For all scripts, the `--wandb` flag enables logging and visualization of metrics on **Weights & Biases**. Below are the specific details for each script and its flags.

1. Baseline Model Training and Evaluation:
    - **--augment**: Includes augmented images in the training dataset.

   ```bash
    python scripts/baseline.py --augment --wandb
   ```
2. VAE Training and Evaluation:
    - **--train**: Trains the VAE model on the dataset.
	- **--generate**: Generates synthetic images using the trained VAE.
	- **--accuracy**: Evaluates classifier accuracy using VAE-augmented data.
    -	Training the VAE:
    ```bash
    python scripts/vae_augmentation.py --train
    ```
    -	Generating Samples:
    ```bash
    python scripts/vae_augmentation.py --generate
    ```
    -	Evaluating Classifier Accuracy:
    ```bash
    python scripts/vae_augmentation.py --accuracy
    ```

3. Stable Diffusion Fine-Tuning and Evalutaion:
    - **--finetune**: Enables fine-tuning 
   ```bash
    python scripts/stableDiffusion_augmentation.py --finetune --wandb
   ```

4. ControlNet Fine-Tuning and Evalutaion:
    - **--finetune**: Enables fine-tuning 
   ```bash
    python scripts/controlNet_augmentation.py --finetune --wandb
   ```