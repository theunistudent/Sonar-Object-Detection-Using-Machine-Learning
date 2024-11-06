import random
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from man_made_model_five_layer import ManMade
import random
import os

# used for running inference on models

# Paths
State_Dict_Path = Path(r"C:\Users\VineyLiam\Documents\Thesis\models\supervised\four_layer_binary_cnn\checkpoints", "state_dict_model.pt")
fishmarket_video_path = Path('C:/Users/VineyLiam/Documents/Thesis/my-data/19-06-2024_fishmarket/FUSION_20240619_120202_exported_1.avi')
image_folder_path = Path(r'C:\Users\VineyLiam\Documents\Thesis\3rd_party_data\shipwreck-dataset\Data\09-03-2022\2022-03-09_15-38-06rec21\sonar')

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
torch.cuda.empty_cache()

# Model
model = ManMade().to(device=device)

# Function to extract a random frame
def get_random_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select a random frame number within the bounds of the video
    random_frame_number = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise ValueError(f"Could not read frame number {random_frame_number}")
    
    print(f"Selected random frame number: {random_frame_number}")
    return frame

# Function to extract a random image from a folder
def get_random_image_from_folder(folder_path):
    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        raise ValueError(f"No image files found in folder {folder_path}")
    
    # Select a random image file from the list
    random_image_file = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image_file)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image {random_image_file}")
    
    print(f"Selected random image: {random_image_file}")
    return image

# Extract a random frame
frame = get_random_frame_from_video(fishmarket_video_path)
image = get_random_image_from_folder(image_folder_path)

# # Convert frame to grayscale if it's not already
# if len(frame.shape) == 3:
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # Convert frame to PIL Image
# frame_pil = Image.fromarray(frame)

# Convert frame to grayscale if it's not already
if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert frame to PIL Image
frame_pil = Image.fromarray(image)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((507, 1024)),
    transforms.ToTensor(),
    transforms.Grayscale(1),
    transforms.Normalize((0.5,), (0.5,)),
])

# Apply transformations
frame_transformed = transform(frame_pil)

# Add batch dimension and move to device
frame_tensor = frame_transformed.unsqueeze(0).to(device)  # Add batch dimension and move to device

# Forward pass through the model
model.eval()
with torch.no_grad():  # Disable gradient calculation
    pred = model(frame_tensor)  # Forward pass through the model

# Convert prediction to numpy for display
pred = torch.sigmoid(pred)
pred_np = pred.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

# Convert frame to numpy for display
frame_display = frame_transformed.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

# Check shape
print(f"Shape of frame_display: {frame_display.shape}")

# Ensure the shape is (H, W) for grayscale images
if frame_display.ndim == 2:
    plt.imshow(frame_display, cmap='gray')  # Display as grayscale image
elif frame_display.ndim == 3 and frame_display.shape[0] == 1:
    plt.imshow(frame_display[0], cmap='gray')  # Display first channel if grayscale
else:
    raise ValueError("Unexpected shape for image data")

plt.title(pred_np)
plt.axis('off')
plt.show()
