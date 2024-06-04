import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import sys, os
from PIL import Image
from torchvision import transforms
import numpy as np

class TrashScreenDataset(Dataset):
    def __init__(self, list_of_files, x_min=-1, x_max=-1, y_min=-1, y_max=-1):
        self.images = list_of_files
        self.image_types = []
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fn_img1 = self.images[idx]
        img1 = Image.open(fn_img1)
        if self.x_min > -1 and self.y_min > -1 and self.x_max > -1 and self.y_max > -1:
            img1 = img1.crop((self.x_min, self.y_min, self.x_max, self.y_max))
        return fn_img1, self.preprocess(img1)

class ResNetExtractor(nn.Module):
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), layer=3):
        super().__init__()
        model = resnet50(pretrained=True).to(device)
        model.eval()
        self.s1_conv1 = model.conv1
        self.s2_bn1 = model.bn1
        self.s3_relu = model.relu
        self.s4_maxpool = model.maxpool
        self.s5_layer1 = model.layer1
        self.s6_layer2 = model.layer2
        self.s7_layer3 = model.layer3
        self.s8_layer4 = model.layer4
        self.avg_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layer = layer

    def forward(self, x):
        x = self.s1_conv1(x)
        x = self.s2_bn1(x)
        x = self.s3_relu(x)
        x = self.s4_maxpool(x)
        if self.layer>0:
            x = self.s5_layer1(x)
            if self.layer > 1:
                x = self.s6_layer2(x)
                if self.layer > 2:
                    x = self.s7_layer3(x)
                    if self.layer > 3:
                        x = self.s8_layer4(x)
        x = self.avg_layer(x)
        return x

if __name__ == "__main__":

    padim_mean_fpath = ""# model mean filepath, eg: 'weights/padim_resnet_mean_4.pth'
    padim_cov_fpath = "" # model cov filepath, eg: 'weights/padim_resnet_cov_4.pth'"
    list_of_files = "" # list of image filepaths, eg ["images/Cornwall_Crinnis/clear/2022_01_28_15_07.jpg", "images/Cornwall_Crinnis/blocked/2022_02_08_16_08.jpg"]
    x_min = -1 # coordinates of the trash screen window (-1 if no window), eg 10
    y_min = -1
    x_max = -1
    y_max = -1

    me = torch.load(padim_mean_fpath)
    ma = torch.load(padim_cov_fpath) + torch.eye(me.size()[0])
    covariance = torch.inverse(ma)
    fe = ResNetExtractor(layer=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fe = fe.to(device)
    fe.eval()
    dataset = TrashScreenDataset(list_of_files=list_of_files)
    dataloader = DataLoader(dataset, batch_size=10)

    for fn, img in dataloader:
        img = img.to(device)
        desc = fe(img).cpu().detach()
        (N, w, _, _) = desc.size()
        desc = desc.reshape((N, w))
        for i in range(N):
            v = desc[i, :]-me
            x = abs(float(torch.matmul(torch.matmul(v.reshape((1, w)), covariance), v.reshape((w, 1)))[0,0]))
            print(f"{fn[i]}: {x}")

