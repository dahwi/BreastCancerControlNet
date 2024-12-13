# BreastCancerControlNet: Enhancing Breast Cancer Classification with ControlNet

**BreastCancerControlNet** is a research project that explores the use of ControlNet, a diffusion-based generative model, to augment breast cancer imaging data. The goal is to improve breast cancer classification accuracy by leveraging advanced data augmentation and synthetic data generation techniques.

## **Overview**

Breast cancer classification relies heavily on high-quality and diverse datasets. However, medical imaging datasets are often limited in size and variability, which can hinder the performance of machine learning models. BreastCancerControlNet addresses this challenge by:
- Using ControlNet to generate synthetic mammogram-like images conditioned on edge maps.
- Augmenting real-world breast cancer datasets to improve model generalization.
- Exploring the feasibility and limitations of using ControlNet in medical imaging.

## **Features**
- **Data Augmentation**: Generate synthetic breast cancer images conditioned on edge maps or other inputs.
- **Conditional Generation**: Guide the model to generate "cancerous" or "non-cancerous" images.
- **Classification Improvement**: Train classifiers on a combination of real and synthetic data for enhanced performance.
- **Evaluation Metrics**: Assess the realism and diagnostic utility of generated images using domain-specific metrics.


## **Dataset**
The project uses a dataset of breast ultrasound images with labels: normal, malignant, and benign. The initial dataset consists of 780 images. [Link to dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/).


<!-- ### **Data Inputs**
- Grayscale mammograms or ultrasound images.
- Corresponding edge maps or segmentation masks.

### **Synthetic Data Outputs**
- Grayscale images conditioned on input edge maps.
- Optionally labeled as "cancerous" or "non-cancerous." -->

<!-- ## **Model Architecture**
- **ControlNet**: A diffusion-based model fine-tuned for generating breast cancer images.
- **Conditional Latent Vectors**: Used to guide image generation based on cancer/no-cancer labels.
- **Evaluation Framework**: Combines perceptual and classification-based metrics to validate synthetic data.
<!-->

## **Installation**
Clone this repository and install the required dependencies:

```bash
git clone
cd BreastCancerControlNet
pip install -r requirements.txt
export PYTHONPATH=/PATH_TO_PROJECT/BreastCancerControlNet:$PYTHONPATH
export BASE_DIR=$(pwd)
export WANDB_KEY="your-wandb-api-key-here"
```

## **Model Fine-tuning**

How to create vae model, stable diffusion lora weights

Once you have lora weights, then fine-tune controlnet

## **Image Generation**

How to generate images for vae, stable diffusion, and controlnet

## **Running Classifier**

How to run classifiers on baseline vae, stable diffusion, controlnet

## **Example Images**

