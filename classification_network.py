import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import pandas as pd

from seasonal_data_split import autumn_test, winter_test, spring_test

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


def predict_for_dataset(season, model, device, test_filenames, threshold=0.5):
    xmin, xmax, ymin, ymax = -1, -1, -1, -1  # Adjust these as needed

    screen_dataset = ScreenDataset(test_filenames, xmin, xmax, ymin, ymax)
    dataloader = torch.utils.data.DataLoader(screen_dataset, batch_size=64, shuffle=False)

    softmax = nn.Softmax(dim=1)

    # Create a DataFrame to store predictions for the test set
    val_df = pd.DataFrame({'file_path': test_filenames})
    val_df['pred'] = None

    for filenames, images in dataloader:
        images = images.to(device)
        predictions = softmax(model(images)).detach()
        for i in range(len(filenames)):
            prediction = "blocked" if predictions[i, 1].item() > threshold else "clear"
            val_df.loc[val_df['file_path'] == filenames[i], 'pred'] = prediction

    # Save the test set DataFrame with predictions
    val_df.to_csv(f'csvs/{season}_pred_{model_season}_model.csv', index=False)
    print(f"Predictions saved for {season} test set.")


if __name__ == "__main__":
    # Import the test sets directly from script B
    season_test_dict = {
        'winter': winter_test,
        'spring': spring_test,
        'autumn': autumn_test,
        'all': autumn_test + winter_test + spring_test  # Combine all test sets for 'all'
    }

    for model_season in ["winter", "spring", "autumn", "all"]:  # Iterate over model seasons
        model_filepath = f'weights/{model_season}_classifier.pth'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load and prepare the model
        model = resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.to(device)
        model.load_state_dict(torch.load(model_filepath, map_location=device))
        model.eval()

        # Predict for each dataset (winter, spring, autumn, all)
        for season in ["winter", "spring", "autumn", "all"]:
            test_filenames = season_test_dict[season]
            predict_for_dataset(season, model, device, test_filenames)