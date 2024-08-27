import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

from seasonal_data_split import winter_test, winter_test_labels, spring_test, spring_test_labels, autumn_test, autumn_test_labels


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


def split_season_data(season_data, test_size=0.2, random_state=42):
    image_filenames = season_data['file_path'].tolist()
    labels = season_data['label'].apply(lambda x: 1 if x == 'blocked' else 0).tolist()
    _, val_filenames, _, val_labels = train_test_split(image_filenames, labels, test_size=test_size,
                                                       random_state=random_state)
    return val_filenames, val_labels


def predict_for_dataset(season, model, device, threshold=0.5):
    # Use a dictionary to map the season name to the corresponding data variable
    season_data_dict = {
        'winter': winter_test,
        'spring': spring_test,
        'autumn': autumn_test,
        'all': winter_test + spring_test + autumn_test
    }

    season_label_dict = {
        'winter': winter_test_labels,
        'spring': spring_test_labels,
        'autumn': autumn_test_labels,
        'all': winter_test_labels + spring_test_labels + autumn_test_labels
    }

    val_filenames = season_data_dict[season]
    val_labels =  season_label_dict[season]

    xmin, xmax, ymin, ymax = -1, -1, -1, -1  # Adjust these as needed

    screen_dataset = ScreenDataset(val_filenames, xmin, xmax, ymin, ymax)
    dataloader = torch.utils.data.DataLoader(screen_dataset, batch_size=64, shuffle=False)

    softmax = nn.Softmax(dim=1)

    # Create a DataFrame to store predictions for the validation set
    val_df = pd.DataFrame({'file_path': val_filenames, 'label': val_labels})
    val_df['pred'] = None

    for filenames, images in dataloader:
        images = images.to(device)
        predictions = softmax(model(images)).detach()
        for i in range(len(filenames)):
            prediction = "blocked" if predictions[i, 1].item() > threshold else "clear"
            val_df.loc[val_df['file_path'] == filenames[i], 'pred'] = prediction

    # Save the validation set DataFrame with predictions
    val_df.to_csv(f'csvs/{season}_pred_{model_season}_model.csv', index=False)
    print(f"Predictions saved for {season} test set.")


if __name__ == "__main__":
    for model_season in ["winter", "spring", "autumn", "all"]:  # Iterate over model seasons
        model_filepath = f'weights/{model_season}_classifier.pth'

        device = torch.device("cuda" if torch.cuda.is_available() else
                              ("mps" if torch.backends.mps.is_available() else "cpu"))
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
            predict_for_dataset(season, model, device)