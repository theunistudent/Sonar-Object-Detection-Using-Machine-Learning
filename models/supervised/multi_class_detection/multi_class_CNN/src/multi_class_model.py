# PyTorch imports
import torch
from torch import nn

# CNN model for the detection of classes in a whole frame. Initial attempt will be capable of detecting 3 different classes
# Initial attempt used 5 CNNBlocks

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x
    
class Model(nn.Module):
    def __init__(self, height, width, class_list):
        super(Model, self).__init__()

        # Used for model output
        num_classes = len(class_list)

        self.conv_relu_stack = nn.Sequential(
            CNNBlock(1, 8),
            nn.MaxPool2d(2,2),  
            CNNBlock(8, 32),
            nn.MaxPool2d(2,2), 
            CNNBlock(32,32, kernel_size=4, stride=2),
            CNNBlock(32,64),
            nn.MaxPool2d(2,2),
            CNNBlock(64,96, kernel_size=4, stride=2),
        )
        
        # Calculate the output size of the last convolutional layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, height, width)
            dummy_output = self.conv_relu_stack(dummy_input)
            flattened_output_size = dummy_output.numel()
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flattened_output_size, num_classes)


    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
