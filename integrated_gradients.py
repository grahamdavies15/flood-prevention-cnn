import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from captum.attr import IntegratedGradients

# Define a function to load your model
def load_model(model_path):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
    model.eval()
    return model

# Function to preprocess images
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        img = Image.open(image_path).convert('RGB')
        return preprocess(img).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to compute and visualize Integrated Gradients
def integrated_gradients(model, img_tensor, class_idx):
    ig = IntegratedGradients(model)
    attributions = ig.attribute(img_tensor, target=class_idx, baselines=img_tensor * 0)
    return attributions

# Function to visualize the attributions
def visualize_attributions(attributions, img_pil, ax, method_name):
    img_pil_resized = img_pil.resize((224, 224))

    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))

    # Summing across channels
    attributions = np.sum(attributions, axis=2)

    # Clipping outliers
    attributions = np.clip(attributions, 0, np.percentile(attributions, 99))

    # Normalizing the attributions
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    ax.imshow(img_pil_resized)

    # Overlaying the attributions with the extent set to cover the whole resized image
    height, width = img_pil_resized.size
    ax.imshow(attributions, cmap='hot', alpha=0.6, extent=(0, width, height, 0))

    ax.set_title(f'{method_name}')
    ax.axis('off')

# Function to process and visualize a single image
def process_image(model_path, image_path):
    model = load_model(model_path)
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None, None
    img_pil = Image.open(image_path).convert('RGB')

    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    # Integrated Gradients
    ig_attributions = integrated_gradients(model, img_tensor, class_idx)
    return ig_attributions, img_pil

# Main script to run integrated gradients on a single image using multiple models
if __name__ == "__main__":
    # List of model paths
    classifiers = ['winter', 'spring', 'autumn']
    model_paths = [f'weights/{classifier}_classifier.pth' for classifier in classifiers]
    image_path = 'Data/blockagedetection_dataset/images/Cornwall_PenzanceCS/blocked/2022_03_01_09_59.jpg'

    # Setup the 3x1 grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 3x1 grid

    for idx, model_path in enumerate(model_paths):
        attributions, img_pil = process_image(model_path, image_path)
        if attributions is not None:
            visualize_attributions(attributions, img_pil, axes[idx], classifiers[idx])

    plt.tight_layout()
    plt.savefig(f'plots/integrated_gradients_comparison.png')
    plt.show()