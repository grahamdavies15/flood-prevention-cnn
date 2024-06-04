import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SiameseTestDataset(Dataset):
    def __init__(self, filenames, clean_ref_images, x_min=-1, x_max=-1, y_min=-1, y_max=-1):
        self.filenames = filenames
        self.clean_ref_images = clean_ref_images
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filenames) * len(self.clean_ref_images)

    def __getitem__(self, item):
        id_ref = item % len(self.clean_ref_images)
        id_tar = item // len(self.clean_ref_images)
        filename_tar = self.filenames[id_tar]
        filename_ref = self.clean_ref_images[id_ref]
        img_tar = Image.open(filename_tar)
        img_ref = Image.open(filename_ref)

        # Crop images if x_min, x_max, y_min, y_max are specified
        if self.x_min > -1 and self.y_min > -1 and self.x_max > -1 and self.y_max > -1:
            crop_ref = img_ref.crop((self.x_min, self.y_min, self.x_max, self.y_max))
            crop_tar = img_tar.crop((self.x_min, self.y_min, self.x_max, self.y_max))
        else:
            crop_ref = img_ref
            crop_tar = img_tar

        return filename_ref, filename_tar, self.preprocess(crop_ref), self.preprocess(crop_tar)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet50(pretrained=True, progress=True)

        out_features = list(self.backbone.modules())[-1].out_features

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        combined_features = feat1 * feat2
        output = self.cls_head(combined_features)
        return output



if __name__ == "__main__":
    model_filepath = "" # model filepath, eg: "weights/siamese.pth"
    image_filenames = [] # list of image filepaths to process (from a single camera), eg ["images/Cornwall_Crinnis/clear/2022_01_28_15_07.jpg", "images/Cornwall_Crinnis/blocked/2022_02_08_16_08.jpg"]
    clean_ref_imgs = [] # list of CLEAR reference images from the same camera as specified in image_filenames, eg ["images/Cornwall_Crinnis/clear/2022_03_26_06_10.jpg", "images/Cornwall_Crinnis/clear/2022_03_28_13_10.jpg"]
    x_min = -1  # coordinates of the trash screen window (-1 if no window), eg 10
    x_max = -1
    y_min = -1
    y_max = -1
    threshold = 0.5 # blockage threshold value (between 0 and 1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SiameseTestDataset(image_filenames, clean_ref_imgs, x_min, x_max, y_min, y_max)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model.eval()
    model.to(device)

    scores = {}

    with torch.no_grad():
        for filename_ref, filename_tar, crop_ref, crop_tar in dataloader:
            crop_ref = crop_ref.to(device)
            crop_tar = crop_tar.to(device)
            predictions = model(crop_ref, crop_tar).cpu().detach().numpy()
            for i in range(len(filename_tar)):
                if filename_tar[i] not in scores:
                    scores[filename_tar[i]] = []
                scores[filename_tar[i]].append(predictions[i][0])

    for key in scores.keys():
        min_score = np.min(scores[key])
        max_score = np.max(scores[key])
        avg_score = np.mean(scores[key])

        label = "blocked" if avg_score > threshold else "clear"

        print(f"{key}: {label}")

