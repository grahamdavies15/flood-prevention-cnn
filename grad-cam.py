import torch
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def visualize_cam(model, cam_extractor, img_tensor, img_pil, class_idx):
    activation_maps = cam_extractor(class_idx, model(img_tensor))

    for activation_map in activation_maps:
        activation_map = activation_map.squeeze(0)
        activation_map_pil = transforms.functional.to_pil_image(activation_map, mode='F')
        result = overlay_mask(img_pil, activation_map_pil, alpha=0.6)
        return result

def process_image(model_path, image_path, device):
    model = load_model(model_path, device)

    cam_extractor = SmoothGradCAMpp(model)

    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None
    img_tensor = img_tensor.to(device)
    img_pil = Image.open(image_path).convert('RGB').resize((224, 224))

    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    return visualize_cam(model, cam_extractor, img_tensor, img_pil, class_idx)

if __name__ == "__main__":
    # Set a fixed seed for reproducibility
    set_seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Define the seasons and corresponding model paths
    classifier = 'autumn'  # Single classifier to be used
    model_path = f'weights/{classifier}_classifier.pth'

    # List of different image paths
    image_paths = [
        'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_10_19_08_30.jpg',
        'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_02_07_08_30.jpg',
        'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_04_08_08_30.jpg'
    ]

    # Process each image and save the result individually
    for idx, image_path in enumerate(image_paths):
        result = process_image(model_path, image_path, device)
        if result is not None:
            plt.figure(figsize=(5, 5))
            plt.imshow(result)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'plots/gradcam_{classifier}_{idx + 1}.png', bbox_inches='tight')
            plt.show()