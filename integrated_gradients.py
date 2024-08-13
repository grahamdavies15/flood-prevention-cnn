import torch
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from captum.attr import IntegratedGradients  # Removed visualize_image_attr

random.seed(55)


# Define a function to load your model
def load_model(model_path):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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


# Define a function to compute and visualize Integrated Gradients
def visualize_integrated_gradients(model, img_tensor, img_pil, class_idx):
    ig = IntegratedGradients(model)
    img_tensor.requires_grad_()

    # Compute integrated gradients
    attr = ig.attribute(img_tensor, target=class_idx, baselines=img_tensor * 0)
    attr = attr.squeeze().detach()
    attr, _ = torch.max(attr, dim=0)

    # Normalize and convert the integrated gradients to a PIL image
    attr = (attr - attr.min()) / (attr.max() - attr.min())  # Normalize to [0, 1]
    attr_pil = transforms.functional.to_pil_image(attr.cpu())  # Convert to PIL image
    return attr_pil


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

    # Visualize Integrated Gradients
    ig_map = visualize_integrated_gradients(model, img_tensor, img_pil, class_idx)

    # Overlay the integrated gradients on the original image
    return Image.blend(img_pil, ig_map, alpha=0.5)


# Set paths and image processing parameters
classifier = 'combined_season'
model_path = f'weights/{classifier}_classifier.pth'
image_folder = 'Data/blockagedetection_dataset/images/Cornwall_PenzanceCS/blocked'

# Get a list of image paths
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]
image_paths = image_paths[:9]  # Ensure we only take the first 9 images

# Create a 3x3 grid of images
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(f"Integrated Gradients for {classifier} classifier", fontsize=16)

# Process each image and display in the grid
for idx, image_path in enumerate(image_paths):
    result = process_image(model_path, image_path)
    if result is not None:
        ax = axes[idx // 3, idx % 3]
        ax.imshow(result)
        ax.axis('off')

plt.tight_layout()
plt.savefig(f'plots/integrated_gradients_{classifier}_classifier.png')
plt.show()
