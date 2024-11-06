#general python imports]
# PyTorch imports
import torch
from torch import nn

# This model is desinged to detect if a frame has a man made object in it.
# It is a simple CNN with no time information. It has 1 output which can be true or false
class ManMade(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            nn.MaxPool2d(3,stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.Dropout(0.2),
        )
        
        # Determine the output size of the last convolutional layer

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*14*30, 1)


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