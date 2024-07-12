import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import sys
import os
import time
import copy
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


if __name__ == "__main__":
    # Select the season
    season = "spring"  # Change this to "winter", "spring", "summer", or "autumn"
    model_season = "autumn"

    model_filepath = f'weights/{model_season}_classifier.pth'

    # Set the dataset and model filepath based on the selected season
    if season == "winter":
        dataset = balanced_winter
        #model_filepath = 'weights/winter_classifier.pth'
    elif season == "spring":
        dataset = balanced_spring
        #model_filepath = 'weights/spring_classifier.pth'
    elif season == "summer":
        dataset = balanced_summer
        #model_filepath = 'weights/summer_classifier.pth'
    elif season == "autumn":
        dataset = balanced_autumn
        #model_filepath = 'weights/autumn_classifier.pth'
    else:
        raise ValueError(f"Invalid season: {season}. Choose from 'winter', 'spring', 'summer', 'autumn'.")

    image_filenames = dataset['file_path'].tolist()  # list of image filepaths
    xmin = -1  # coordinates of the trash screen window (-1 if no window), eg 10
    xmax = -1  # 235
    ymin = -1  # 10
    ymax = -1  # 235
    threshold = 0.5  # blockage threshold value (between 0 and 1)

    batch_size = 32

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    screen_dataset = ScreenDataset(image_filenames, xmin, xmax, ymin, ymax)
    dataloader = torch.utils.data.DataLoader(screen_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using device: {device}")

    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model.to(device)
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model.eval()
    softmax = nn.Softmax(dim=1)

    # has a 'pred' column
    dataset['pred'] = None

    for filenames, images in dataloader:
        images = images.to(device)
        predictions = softmax(model(images)).detach()
        for i in range(len(filenames)):
            prediction = "blocked" if predictions[i, 1].item() > threshold else "clear"
            dataset.loc[dataset['file_path'] == filenames[i], 'pred'] = prediction

    # Save dataframe with predictions and original labels
    dataset.to_csv(f'{season}_pred_{model_season}_model.csv', index=False)
