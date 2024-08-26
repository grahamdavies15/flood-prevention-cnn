import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import pandas as pd

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
        img = Image.open(self.filenames[item])
        if self.xmin > 0 and self.xmax > 0 and self.ymin > 0 and self.ymax > 0:
            img = img.crop((self.xmin, self.ymin, self.xmax, self.ymax))
        return self.filenames[item], self.preprocess(img)

def predict_for_dataset(season, model, device, threshold=0.5):
    # Load the dataset based on the selected season
    if season == "winter":
        dataset = balanced_winter
    elif season == "spring":
        dataset = balanced_spring
    elif season == "summer":
        dataset = balanced_summer
    elif season == "autumn":
        dataset = balanced_autumn
    elif season == 'all':
        # Combine the data from winter, spring, and autumn
        dataset = pd.concat([balanced_spring, balanced_autumn, balanced_winter], ignore_index=True)
    else:
        raise ValueError(f"Invalid season: {season}. Choose from 'winter', 'spring', 'all', 'autumn'.")

    image_filenames = dataset['file_path'].tolist()  # list of image filepaths
    xmin, xmax, ymin, ymax = -1, -1, -1, -1  # Adjust these as needed

    screen_dataset = ScreenDataset(image_filenames, xmin, xmax, ymin, ymax)
    dataloader = torch.utils.data.DataLoader(screen_dataset, batch_size=32, shuffle=False)

    softmax = nn.Softmax(dim=1)

    # Add a 'pred' column to store predictions
    dataset['pred'] = None

    for filenames, images in dataloader:
        images = images.to(device)
        predictions = softmax(model(images)).detach()
        for i in range(len(filenames)):
            prediction = "blocked" if predictions[i, 1].item() > threshold else "clear"
            dataset.loc[dataset['file_path'] == filenames[i], 'pred'] = prediction

    # Save dataframe with predictions
    dataset.to_csv(f'csvs/{season}_pred_{model_season}_model.csv', index=False)
    print(f"Predictions saved for {season} dataset.")

if __name__ == "__main__":
    model_season = "spring"  # Choose the model season (e.g., "spring")
    model_filepath = f'weights/{model_season}_classifier.pth'

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare the model
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model.eval()

    # Predict for each dataset (winter, spring, summer, autumn)
    for season in ["all"]: # "winter", "spring", "autumn",
        predict_for_dataset(season, model, device)