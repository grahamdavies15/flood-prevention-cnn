import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn

# Assume these functions are defined in 'seasonal_data_split.py'
from seasonal_data_split import balanced_winter, balanced_spring, balanced_summer, balanced_autumn


class ScreenDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, xmin=-1, xmax=-1, ymin=-1, ymax=-1):
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        img = Image.open(self.filenames[item]).convert("RGB")
        if self.xmin > 0 and self.xmax > 0 and self.ymin > 0 and self.ymax > 0:
            img = img.crop((self.xmin, self.ymin, self.xmax, self.ymax))
        return self.filenames[item], self.preprocess(img)


def get_activations(model, image, layer_name):
    def hook_fn(module, input, output):
        activation[layer_name] = output.detach()

    activation = {}
    layer = dict([*model.named_modules()])[layer_name]
    handle = layer.register_forward_hook(hook_fn)

    model(image)

    handle.remove()
    return activation[layer_name].squeeze().cpu().numpy()


def compare_activations(models, image, layer_name):
    activations = {}
    for model_name, model in models.items():
        activations[model_name] = get_activations(model, image, layer_name)

    num_models = len(models)
    num_activations = list(activations.values())[0].shape[0]

    # Determine the grid size for plotting
    cols = 8
    rows = (num_activations // cols) + (num_activations % cols > 0)

    fig, axes = plt.subplots(rows, cols * num_models, figsize=(20, rows * 2))
    axes = axes.flatten()

    for i in range(num_activations):
        for j, (model_name, act) in enumerate(activations.items()):
            ax = axes[i * num_models + j]
            ax.imshow(act[i], cmap='viridis')
            if i == 0:  # Only add titles on the first row
                ax.set_title(model_name)
            ax.axis('off')

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig('plots/activation_functions.png')
    plt.show()


if __name__ == "__main__":
    # Select the season
    season = "spring"  # Change this to "winter", "spring", "summer", or "autumn"
    model_seasons = ["winter", "spring", "autumn"]

    # Set the dataset based on the selected season
    if season == "winter":
        dataset = balanced_winter
    elif season == "spring":
        dataset = balanced_spring
    elif season == "summer":
        dataset = balanced_summer
    elif season == "autumn":
        dataset = balanced_autumn
    else:
        raise ValueError(f"Invalid season: {season}. Choose from 'winter', 'spring', 'summer', 'autumn'.")

    image_filenames = dataset['file_path'].tolist()  # list of image filepaths
    xmin = -1  # coordinates of the trash screen window (-1 if no window), eg 10
    xmax = -1  # 235
    ymin = -1  # 10
    ymax = -1  # 235

    batch_size = 32

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    screen_dataset = ScreenDataset(image_filenames, xmin, xmax, ymin, ymax)
    dataloader = torch.utils.data.DataLoader(screen_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using device: {device}")

    models = {}
    for model_season in model_seasons:
        model_filepath = f'weights/{model_season}_classifier.pth'
        model = resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.to(device)
        model.load_state_dict(torch.load(model_filepath, map_location=device))
        model.eval()
        models[model_season] = model

    # Visualize activations for a sample from the dataset
    sample_dataloader = torch.utils.data.DataLoader(screen_dataset, batch_size=1, shuffle=True, num_workers=0)
    layer_name = 'layer4.0.conv1'

    # Get a single image to compare across models
    dataiter = iter(sample_dataloader)
    filenames, images = next(dataiter)
    images = images.to(device)

    compare_activations(models, images, layer_name)