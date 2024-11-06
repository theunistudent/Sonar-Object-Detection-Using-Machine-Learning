#general python imports]
# PyTorch imports
import torch
from torch import nn

# This model is desinged to detect if a frame has a man made object in it.
# It is a simple CNN with no time information. It has 1 output which can be true or false

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

class ManMade(nn.Module):
    def __init__(self, height, width):
        super(ManMade, self).__init__()
        self.conv_relu_stack = nn.Sequential(
            CNNBlock(1, 8),
            CNNBlock(8,8),
            nn.MaxPool2d(2,2),  
            CNNBlock(8, 32),
            CNNBlock(32,32),
            nn.MaxPool2d(2,2), 
            CNNBlock(32,32, kernel_size=4, stride=2),
            CNNBlock(32,64),
            CNNBlock(64,64),
            nn.MaxPool2d(2,2),
            CNNBlock(64,64),
            CNNBlock(64,96, kernel_size=4, stride=2),
        )
        
        # Calculate the output size of the last convolutional layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, height, width)
            dummy_output = self.conv_relu_stack(dummy_input)
            flattened_output_size = dummy_output.numel()
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flattened_output_size, 1)


    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    # Great for getting the network shape after the relustack. run the following in the main module
    # model = ManMade()
    # print(model.get_output_shape((1, 1, 507, 1024), device))  
    # Replace (507, 1024) with your input dimensions
    def get_output_shape(self, input_shape, device):
        with torch.no_grad():
            dummy_input = torch.randn(*input_shape).to(device)
            x = self.conv_relu_stack(dummy_input)
            return x.shape