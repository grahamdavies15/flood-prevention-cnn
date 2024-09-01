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

def process_image(model, image_path, device):
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
    classifier = 'spring'  # Use a single classifier
    model_path = f'weights/{classifier}_classifier.pth'
    image_paths = [
        'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_10_19_08_30.jpg',
        'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_02_07_08_30.jpg',
        'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_04_08_08_30.jpg'
    ]

    # Load the model once
    model = load_model(model_path, device)

    for idx, image_path in enumerate(image_paths):
        saliency_map, img_pil = process_image(model, image_path, device)
        if saliency_map is not None:
            saliency_map_resized = Image.fromarray(np.uint8(saliency_map * 255)).resize((224, 224), resample=Image.BILINEAR)

            plt.figure(figsize=(5, 5))
            plt.imshow(img_pil, alpha=0.6)
            plt.imshow(saliency_map_resized, cmap='hot', alpha=0.4)
            #plt.title(f'Saliecy Mapping using {classifier} classifier')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'plots/saliency_{classifier}_{idx + 1}.png', bbox_inches='tight')
            plt.show()