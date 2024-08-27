import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from captum.attr import IntegratedGradients

def load_model(model_path):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
    model.eval()
    return model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        img = Image.open(image_path).convert('RGB')
        return preprocess(img).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def integrated_gradients(model, img_tensor, class_idx):
    ig = IntegratedGradients(model)
    attributions = ig.attribute(img_tensor, target=class_idx, baselines=img_tensor * 0)
    return attributions

def visualize_attributions(attributions, img_pil, ax, method_name):
    img_pil_resized = img_pil.resize((224, 224))

    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))

    attributions = np.sum(attributions, axis=2)

    attributions = np.clip(attributions, 0, np.percentile(attributions, 99))

    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    ax.imshow(img_pil_resized)

    height, width = img_pil_resized.size
    ax.imshow(attributions, cmap='hot', alpha=0.6, extent=(0, width, height, 0))

    #ax.set_title(f'{method_name}')
    ax.axis('off')

def process_image(model_path, image_path):
    model = load_model(model_path)
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None, None
    img_pil = Image.open(image_path).convert('RGB').resize((224, 224))

    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    ig_attributions = integrated_gradients(model, img_tensor, class_idx)
    return ig_attributions, img_pil


if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Define the single classifier to be used
    classifier = 'all'
    model_path = f'weights/{classifier}_classifier.pth'

    # List of different image paths
    image_paths = [
        'Data/blockagedetection_dataset/images/Cornwall_Portreath/blocked/2022_11_26_12_59.jpg',
        'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_02_17_11_30.jpg',
        'Data/blockagedetection_dataset/images/sites_corshamaqueduct_cam1/blocked/2022_04_19_13_29.jpg'
    ]

    # Process each image and save the result individually
    for idx, image_path in enumerate(image_paths):
        attributions, img_pil = process_image(model_path, image_path)
        if attributions is not None:
            plt.figure(figsize=(5, 5))
            ax = plt.gca()  # Get the current axis
            visualize_attributions(attributions, img_pil, ax, f"Image {idx + 1} - {classifier.capitalize()}")
            plt.axis('off')

            plt.savefig(f'plots/integrated_{classifier}_{idx + 1}.png')
            plt.show()