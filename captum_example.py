import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import numpy as np
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam
from captum.attr import visualization as viz

random.seed(55)

# ScreenDataset Class
class ScreenDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, xmin=-1, xmax=-1, ymin=-1, ymax=-1):
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        img = Image.open(self.filenames[item]).convert("RGB")
        if self.xmin > 0 and self.xmax > 0 and self.ymin > 0 and self.ymax > 0:
            img = img.crop((self.xmin, self.ymin, self.xmax, self.ymax))
        return self.filenames[item], self.preprocess(img)

# Function to load your model
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

# Function to compute and visualize Occlusion
def occlusion(model, img_tensor, class_idx):
    occlusion = Occlusion(model)
    attributions = occlusion.attribute(img_tensor, target=class_idx, strides=(3, 8, 8), sliding_window_shapes=(3, 15, 15))
    return attributions

# Function to visualize the attributions
def visualize_attributions(attributions, img_pil, method_name):
    # Resize the original image to match the preprocessed image size
    img_pil_resized = img_pil.resize((224, 224))

    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))

    # Summing across channels
    attributions = np.sum(attributions, axis=2)

    # Clipping outliers
    attributions = np.clip(attributions, 0, np.percentile(attributions, 99))

    # Normalizing the attributions
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    plt.figure(figsize=(8, 8))
    plt.imshow(img_pil_resized)

    # Overlaying the attributions with the extent set to cover the whole resized image
    height, width = img_pil_resized.size
    plt.imshow(attributions, cmap='hot', alpha=0.6, extent=(0, width, height, 0))

    plt.title(f'{method_name}')
    plt.axis('off')
    plt.show()

# Function to process and visualize a single image
def process_image(model_path, image_path):
    model = load_model(model_path)
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None
    img_pil = Image.open(image_path).convert('RGB')

    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    # Integrated Gradients
    ig_attributions = integrated_gradients(model, img_tensor, class_idx)
    visualize_attributions(ig_attributions, img_pil, 'Integrated Gradients')

    # Occlusion
    occlusion_attributions = occlusion(model, img_tensor, class_idx)
    visualize_attributions(occlusion_attributions, img_pil, 'Occlusion')

# Dataset and Visualization Process
classifier = 'combined_season'
model_path = f'weights/{classifier}_classifier.pth'
image_folder = 'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked'
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]
image_paths = image_paths[:9]

# Loading the dataset (Example usage)
dataset = ScreenDataset(image_paths)

for idx, (image_path, img_tensor) in enumerate(dataset):
    process_image(model_path, image_path)