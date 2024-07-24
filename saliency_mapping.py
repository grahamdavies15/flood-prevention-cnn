import torch
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchcam.utils import overlay_mask
import os


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


# Define a function to apply CAM and visualize
def visualize_cam(model, cam_extractor, img_tensor, img_pil, class_idx, title=""):
    activation_maps = cam_extractor(class_idx, model(img_tensor))

    for activation_map in activation_maps:
        activation_map = activation_map.squeeze(0)  # Remove batch dimension
        activation_map_pil = transforms.functional.to_pil_image(activation_map, mode='F')  # Convert to PIL image

        result = overlay_mask(img_pil, activation_map_pil, alpha=0.5)
        plt.imshow(result)
        plt.axis('off')
        plt.title(title)
        plt.show()


# Function to process and visualize a single image
def process_image(model, model_path, image_path, title=""):
    # Load model
    model = load_model(model_path)

    # Initialize CAM extractor
    cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')

    # Preprocess image
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return  # Skip if there was an error loading the image
    img_pil = Image.open(image_path).convert('RGB')

    # Get class prediction
    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    # Visualize CAM
    visualize_cam(model, cam_extractor, img_tensor, img_pil, class_idx, title=title)


# Define paths to models and example image
models_paths = {
    #"Autumn Model": 'weights/autumn_classifier.pth',
    #"Spring Model": 'weights/spring_classifier.pth',
    "Winter Model": 'weights/winter_classifier.pth'
}

example_image_path = 'Data/blockagedetection_dataset/images/sites_sheptonmallet_cam2/blocked/2022_10_19_15_50.jpg'

# Process and visualize for each model
for model_name, model_path in models_paths.items():
    print(f"Processing {model_name}...")
    process_image(model_name, model_path, example_image_path, title=model_name)
