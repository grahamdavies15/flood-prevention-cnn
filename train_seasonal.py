import pandas as pd
import torch
import copy
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from seasonal_data_split import balanced_winter, balanced_spring, balanced_summer, balanced_autumn
import matplotlib.pyplot as plt


# Define the custom dataset class
class ScreenDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, xmin=-1, xmax=-1, ymin=-1, ymax=-1):
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
        self.labels = labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        img = Image.open(self.filenames[item])
        if self.xmin > 0 and self.xmax > 0 and self.ymin > 0 and self.ymax > 0:
            img = img.crop((self.xmin, self.ymin, self.xmax, self.ymax))
        label = self.labels[item]
        return self.preprocess(img), label


# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, min_delta=0.01, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.cpu().numpy())  # Convert to CPU and then to numpy
                if scheduler:
                    scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.cpu().numpy())  # Convert to CPU and then to numpy
                # Early stopping
                if epoch_acc - best_acc > min_delta:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == patience:
                    print("Early stopping")
                    model.load_state_dict(best_model_wts)
                    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, (train_losses, val_losses, train_accuracies, val_accuracies)


# Choose the season to train on
season_data = balanced_winter  # Change this to balanced_spring, balanced_summer, or balanced_autumn as needed

# Extract file paths and labels
image_filenames = season_data['file_path'].tolist()
labels = season_data['label'].apply(lambda x: 1 if x == 'blocked' else 0).tolist()

# Split the dataset
train_filenames, val_filenames, train_labels, val_labels = train_test_split(image_filenames, labels, test_size=0.2,
                                                                            random_state=42)

# Coordinates for cropping (change if needed)
xmin, xmax, ymin, ymax = -1, -1, -1, -1

# Create datasets
train_dataset = ScreenDataset(train_filenames, train_labels, xmin, xmax, ymin, ymax)
val_dataset = ScreenDataset(val_filenames, val_labels, xmin, xmax, ymin, ymax)

# Create dataloaders
batch_size = 32
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
}

# Load ResNet50 model
print("Loading ResNet50 model...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Modify the final layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=25, min_delta=0.01, patience=5)

# Plot training graph
train_losses, val_losses, train_accuracies, val_accuracies = history

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

# Confirmation prompt before saving the model
save_model = input("Do you want to save the model? (yes/no): ").strip().lower()
if save_model == 'yes':
    # Save the model
    model_filepath = 'weights/winter_classifier.pth'
    torch.save(model.state_dict(), model_filepath)
    print(f"Model saved to {model_filepath}")
else:
    print("Model not saved.")