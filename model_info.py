import torch

# Load the checkpoint
checkpoint = torch.load('weights/winter_classifier.pth')

# Inspect the contents
print("Checkpoint keys:", checkpoint.keys())

# Extracting information (if available)
model_state_dict = checkpoint.get('model_state_dict', None)
optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
epoch = checkpoint.get('epoch', None)
train_accuracy = checkpoint.get('train_accuracy', None)
val_accuracy = checkpoint.get('val_accuracy', None)

print(f"Model State Dict: {model_state_dict is not None}")
print(f"Optimizer State Dict: {optimizer_state_dict is not None}")
print(f"Epoch: {epoch}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

from torchvision.models import resnet50
import torch.nn as nn

def list_activation_functions(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Softmax, nn.Softplus)):
            print(f"{name}: {type(module).__name__}")

# Load the ResNet50 model
model = resnet50()

# List all activation functions in the model
list_activation_functions(model)

