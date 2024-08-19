import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

def generate_saliency_map(model, img_tensor, class_idx):
    img_tensor.requires_grad = True
    output = model(img_tensor)
    model.zero_grad()
    output[0, class_idx].backward()
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency.squeeze().cpu().numpy()

def process_image(model_path, image_path, device):
    model = load_model(model_path, device)
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None, None
    img_tensor = img_tensor.to(device)
    img_pil = Image.open(image_path).convert('RGB').resize((224, 224))

    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    saliency_map = generate_saliency_map(model, img_tensor, class_idx)
    return saliency_map, img_pil

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    classifiers = ['winter', 'spring', 'autumn']
    model_paths = [f'weights/{classifier}_classifier.pth' for classifier in classifiers]
    image_path = 'Data/blockagedetection_dataset/images/Cornwall_PenzanceCS/blocked/2022_03_02_10_59.jpg'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, model_path in enumerate(model_paths):
        saliency_map, img_pil = process_image(model_path, image_path, device)
        if saliency_map is not None:
            saliency_map_resized = Image.fromarray(np.uint8(saliency_map * 255)).resize((224, 224), resample=Image.BILINEAR)
            ax = axes[idx]
            ax.imshow(img_pil, alpha=0.6)
            ax.imshow(saliency_map_resized, cmap='hot', alpha=0.4)
            ax.set_title(f'{classifiers[idx]} model')
            ax.axis('off')

    plt.tight_layout(pad=2.0)  # Add padding to ensure titles are not cut off
    plt.savefig(f'plots/saliency_comparison.png')
    plt.show()