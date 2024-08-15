import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from captum.attr import IntegratedGradients
import numpy as np

random.seed(55)

# Define a function to load your model
def load_model(model_path):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
    model.eval()
    return model

# Define a function to preprocess images
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

# Define a function to compute integrated gradients and visualize
def visualize_integrated_gradients(model, img_tensor, img_pil, class_idx):
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    # Compute the attributions using integrated gradients
    attributions = ig.attribute(img_tensor, target=class_idx, baselines=img_tensor * 0)

    # Convert the attributions to a numpy array for visualization
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))  # HWC format

    # Normalize the attributions for better visualization
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # Plot the original image and the attributions
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    ax[0].imshow(img_pil)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    # Integrated Gradients
    ax[1].imshow(img_pil)
    ax[1].imshow(attributions, cmap='hot', alpha=0.6)
    ax[1].axis('off')
    ax[1].set_title('Integrated Gradients')

    plt.show()

# Function to process and visualize a single image
def process_image(model_path, image_path):
    # Load model
    model = load_model(model_path)

    # Preprocess image
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None  # Skip if there was an error loading the image
    img_pil = Image.open(image_path).convert('RGB')

    # Get class prediction
    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    # Visualize integrated gradients
    visualize_integrated_gradients(model, img_tensor, img_pil, class_idx)

classifier = 'combined_season'
# Path to model
model_path = f'weights/{classifier}_classifier.pth'

# Path to images
image_folder = 'Data/blockagedetection_dataset/images/Cornwall_PenzanceCS/blocked'

# Get a list of image paths
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]

# Ensure we only take the first 9 images
image_paths = image_paths[:9]

# Process each image and display in a loop
for idx, image_path in enumerate(image_paths):
    process_image(model_path, image_path)