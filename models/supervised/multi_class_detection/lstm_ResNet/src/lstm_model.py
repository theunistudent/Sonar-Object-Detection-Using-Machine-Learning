import torch
from torch import nn
from torchvision import models

# ResNet-based model for multi-class classification with an LSTM layer
class ResNetLSTMModel(nn.Module):
    def __init__(self, height, width, class_list, lstm_hidden_size=128, num_lstm_layers=1, bidirectional=False):
        super(ResNetLSTMModel, self).__init__()

        # Specify the appropriate weights enum when loading the model
        weights = models.ResNet18_Weights.DEFAULT  # Use ResNet18_Weights

        # Load pre-trained ResNet18 model with specified weights
        self.resnet = models.resnet18(weights=weights)

        # Modify the first convolutional layer if input is grayscale (1-channel input)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the fully connected layer of ResNet18 to use as a feature extractor
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Adaptive pooling layer to reduce spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size of (1, 1)

        # LSTM layer to process the ResNet features
        self.lstm = nn.LSTM(
            input_size=512,  # After pooling, each feature will have 512 dimensions
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layer after LSTM
        num_classes = len(class_list)
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

        # Freeze all layers except the last block (`layer4`) of ResNet
        self.freeze_layers()

    def freeze_layers(self):
        """
        Freeze all layers except for the last block (`layer4`) of ResNet.
        """
        for name, param in self.resnet.named_parameters():
            if "layer4" in name:  # Unfreeze layer4
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        # print(f"Input shape: {x.size()}")  # Check the shape

        # Check if x has 6 dimensions and squeeze the unnecessary dimension if it's size 1
        if x.dim() == 6 and x.size(1) == 1:
            x = x.squeeze(1)  # Remove the second dimension, resulting in shape [1, 10, 1, 1080, 1920]

        # Ensure we have a 5D tensor now
        if x.dim() != 5:
            raise ValueError("Input must be a 5D tensor after squeezing.")

        batch_size, time_steps, c, h, w = x.size()

        # Reshape to merge batch and time dimensions
        x = x.view(batch_size * time_steps, c, h, w)

        # Extract features with ResNet (ignoring the last pooling and FC layers)
        features = self.resnet(x)  # Shape: [batch_size * time_steps, 512, H', W']

        # Apply adaptive pooling to reduce H' and W' to 1 (average across spatial dimensions)
        pooled_features = self.adaptive_pool(features)  # Shape: [batch_size * time_steps, 512, 1, 1]

        # Flatten spatial dimensions (1, 1) to create a feature vector
        flattened_features = pooled_features.view(pooled_features.size(0), -1)  # Shape: [batch_size * time_steps, 512]

        # Reshape back to (batch_size, time_steps, features)
        lstm_input = flattened_features.view(batch_size, time_steps, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Take the last output of LSTM for classification
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, lstm_hidden_size)

        # Pass through the final fully connected layer
        output = self.fc(lstm_out)

        return output
