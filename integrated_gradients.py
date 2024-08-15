import torch
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from captum.attr import IntegratedGradients

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

# Define a function to apply Integrated Gradients and visualize
def visualize_integrated_gradients(model, img_tensor, img_pil, class_idx):
    # Initialize Integrated Gradients
    integrated_gradients = IntegratedGradients(model)

    # Compute attributions
    attributions = integrated_gradients.attribute(img_tensor, target=class_idx, baselines=img_tensor * 0)

    # Convert attributions to numpy and remove batch dimension
    attributions = attributions.squeeze(0).permute(1, 2, 0).detach().numpy()

    # Normalize attributions for visualization
    attr_min, attr_max = attributions.min(), attributions.max()
    attributions = (attributions - attr_min) / (attr_max - attr_min)

    # Convert attributions to PIL image for overlay
    attributions_pil = Image.fromarray((attributions * 255).astype('uint8'))

    # Resize the original image to match the size of attributions
    img_pil_resized = img_pil.resize(attributions_pil.size, Image.Resampling.LANCZOS)

    # Overlay attributions on the resized original image
    result = Image.blend(img_pil_resized, attributions_pil, alpha=0.5)
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

    # Visualize Integrated Gradients
    return visualize_integrated_gradients(model, img_tensor, img_pil, class_idx)

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
fig.suptitle(f"Integrated Gradients for {classifier} classifier", fontsize=16)

# Process each image and display in the grid
for idx, image_path in enumerate(image_paths):
    result = process_image(model_path, image_path)
    if result is not None:
        ax = axes[idx // 3, idx % 3]
        ax.imshow(result)
        ax.axis('off')

plt.tight_layout()

# Save the plot to a file
plt.savefig(f'plots/integrated_gradients_{classifier}_classifier.png')
plt.show()