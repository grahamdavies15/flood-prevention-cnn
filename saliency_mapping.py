import torch
from torchvision import models, transforms
from torchcam.utils import overlay_mask
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

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

# Define a function to create a saliency map and visualize
def visualize_saliency(model, img_tensor, img_pil, class_idx):
    # Make sure the model is in evaluation mode and gradients are enabled
    model.eval()
    img_tensor.requires_grad = True

    # Forward pass
    output = model(img_tensor)
    output = output[0, class_idx]

    # Backward pass to get gradients
    output.backward()

    # Get the gradient of the input image
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu()

    # Convert to PIL image for visualization
    saliency_pil = transforms.functional.to_pil_image(saliency, mode='F')

    # Overlay the saliency map on the original image
    result = overlay_mask(img_pil, saliency_pil, alpha=0.5)
    return result

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

    # Visualize saliency map
    return visualize_saliency(model, img_tensor, img_pil, class_idx)

classifier = 'combined_season'
# Path to model
model_path = f'weights/{classifier}_classifier.pth'

# Path to images
image_folder = 'Data/blockagedetection_dataset/images/Cornwall_PenzanceCS/blocked'

# Get a list of image paths
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]

# Ensure we only take the first 9 images
image_paths = image_paths[:9]

# Create a 3x3 grid of images
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(f"Saliency Maps for {classifier} classifier", fontsize=16)

# Process each image and display in the grid
for idx, image_path in enumerate(image_paths):
    result = process_image(model_path, image_path)
    if result is not None:
        ax = axes[idx // 3, idx % 3]
        ax.imshow(result)
        ax.axis('off')

plt.tight_layout()

# Save the plot to a file
plt.savefig(f'plots/saliency_{classifier}_classifier.png')
plt.show()