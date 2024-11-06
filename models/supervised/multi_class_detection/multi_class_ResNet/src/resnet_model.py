# PyTorch imports
import torch
from torch import nn
from torchvision import models

# ResNet-based model for multi-class classification
class ResNetModel(nn.Module):
    def __init__(self, height, width, class_list):
        super(ResNetModel, self).__init__()
        
        # Specify the appropriate weights enum when loading the model
        weights = models.ResNet34_Weights.DEFAULT  # Use ResNet34_Weights if using torchvision >= 0.13
        
        # Load pre-trained ResNet34 model with specified weights
        self.model = models.resnet34(weights=weights)
        
        # Modify the first convolutional layer if input is grayscale (1-channel input)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer to match the number of classes
        num_classes = len(class_list)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Freeze all layers except the last block (`layer4`) and the fully connected layer (`fc`)
        self.freeze_layers()

    def freeze_layers(self):
        """
        Freeze all layers except for the last block (`layer4`) and the final fully connected layer (`fc`).
        """
        for name, param in self.model.named_parameters():
            # Unfreeze `layer4` and `fc` (final fully connected layer)
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)
