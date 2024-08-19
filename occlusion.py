import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from captum.attr import Occlusion

def load_model(model_path, device):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
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

def compute_occlusion(model, img_tensor, class_idx):
    occlusion = Occlusion(model)
    attributions = occlusion.attribute(img_tensor, target=class_idx, strides=(3, 8, 8), sliding_window_shapes=(3, 15, 15))
    return attributions

def process_image(model_path, image_path, device):
    model = load_model(model_path, device)
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None, None
    img_tensor = img_tensor.to(device)
    img_pil = Image.open(image_path).convert('RGB')

    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    occlusion_attributions = compute_occlusion(model, img_tensor, class_idx)
    return occlusion_attributions, img_pil

def visualize_models(image_path, model_paths, model_names, device):
    attributions_list = []
    img_pil = None

    for model_path in model_paths:
        attributions, img_pil = process_image(model_path, image_path, device)
        if attributions is not None:
            attributions = attributions.squeeze().cpu().detach().numpy()
            attributions = np.transpose(attributions, (1, 2, 0))
            attributions = np.sum(attributions, axis=2)
            attributions = np.clip(attributions, 0, np.percentile(attributions, 99))
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())
            attributions_list.append(attributions)

    if img_pil is not None and len(attributions_list) == 3:
        img_pil_resized = img_pil.resize((224, 224))
        width, height = img_pil_resized.size

        plt.figure(figsize=(15, 5))

        for i, (attributions, model_name) in enumerate(zip(attributions_list, model_names)):
            plt.subplot(1, 3, i+1)
            plt.imshow(img_pil_resized)
            plt.imshow(attributions, cmap='hot', alpha=0.6, extent=(0, width, height, 0))
            plt.title(f'{model_name}')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('plots/occlusion_comparison.png')
        plt.show()

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_names = ['winter', 'spring', 'autumn']
    model_paths = [f'weights/{name}_classifier.pth' for name in model_names]
    image_path = 'Data/blockagedetection_dataset/images/Cornwall_PenzanceCS/blocked/2022_03_02_10_59.jpg'

    visualize_models(image_path, model_paths, model_names, device)