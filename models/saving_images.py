import os
import random
from PIL import Image
from multi_class_CNN_data import SonarImages, SonarVideo, CombinedSonarDataset

import os
import logging
import random
import shutil
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch

# Imports for dataset classes
import pandas as pd
from torch.utils.data import Dataset
import cv2
import datetime


# Used for grabbing instances form each class to place in report

# Assuming SonarImages, SonarVideo, CombinedSonarDataset classes are already defined

# -------------- Directories Setup -----------------
sonar_training_data = '/mnt/sda2/liam/SONAR_thesis/sonar_training_data'  # Replace with actual data folder path

relative_checkpoint_dir = os.path.join(os.path.dirname(__file__), '../checkpoints/')
relative_historic_checkpoint_dir = os.path.join(relative_checkpoint_dir, 'historic_checkpoints/')
relative_checkpoint_path = os.path.join(relative_checkpoint_dir, "current_checkpoint.tar")
training_metrics_location = os.path.join(os.path.dirname(__file__), '../metrics/training_metrics.csv')
training_metrics = os.path.join(os.path.dirname(__file__), '../metrics/')

# Paths to parent folders
images = os.path.join(sonar_training_data, 'images')
videos = os.path.join(sonar_training_data, 'videos')

class_list = [
    "tire",
    "shipwreck",
    "pylon",
    "concrete_block",
    "pipe",
    "man_made_unclassified",
    "car",
    "rope",
    "ROV",
    "dock",
    "propeller",
    "ship_hull",
    "rudder"
]


# Transformation to apply to images (optional)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# -------------------- Create DataLoaders -------------------------
image_dataset = []
video_dataset = []

# Check if sonar images directory exists
if os.path.exists(images):
    for current_folder in os.listdir(images):
        for folder in os.listdir(os.path.join(images, current_folder)):
            csv_file = ""
            img_dir = f"{images}/{current_folder}/{folder}/sonar"
            for file in os.listdir(f"{images}/{current_folder}/{folder}"):
                file_name, file_extension = os.path.splitext(file)
                if file_extension == ".csv" and file_name != "IMULog":
                    csv_file = f"{images}/{current_folder}/{folder}/{file}"

            if csv_file == "":
                logging.error(f'No CSV found in image folder: {os.path.join(current_folder, folder)}')
            else:
                image_dataset.append(SonarImages(class_list, csv_file, img_dir, transform=transform))
else:
    logging.warning('No images directory in sonar_training_data')

# Check if sonar videos directory exists
if os.path.exists(videos):
    for current_folder in os.listdir(videos):
        csv_file = None
        video_file = None
        current_folder_path = os.path.join(videos, current_folder)

        if os.path.isdir(current_folder_path):
            for file in os.listdir(current_folder_path):
                file_name, file_extension = os.path.splitext(file)
                if file_extension == ".csv" and file_name != "IMULog":
                    csv_file = os.path.join(current_folder_path, file)
                elif file_extension == ".avi":
                    video_file = os.path.join(current_folder_path, file)

                if csv_file is not None and video_file is not None:
                    break

        if csv_file is None or video_file is None:
            logging.error(f'Video file or CSV not loaded from: {os.path.join(videos, current_folder)}')
        else:
            video_dataset.append(SonarVideo(class_list, csv_file, video_file, transform=transform))
else:
    logging.warning('No video directory in sonar_training_data')

# Combine image and video datasets
combined_dataset = CombinedSonarDataset(image_datasets=image_dataset, video_datasets=video_dataset)

# Create DataLoader
data_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

# # -------------------- Save Random Images -------------------------
# def save_random_images_per_class(combined_dataset, class_list, num_images_per_class=10):
#     # Create DataLoader
#     data_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

#     # Parent folder where images will be stored
#     parent_folder = os.path.join(os.path.dirname(__file__), 'random_class_images')
#     if not os.path.exists(parent_folder):
#         os.makedirs(parent_folder)

#     # Track saved images per class
#     saved_images_per_class = {class_name: 0 for class_name in class_list}
#     saved_no_class_images = 0

#     for image, labels in data_loader:
#         # Save images for each class
#         for idx, class_name in enumerate(class_list):
#             if labels[0, idx].item() == 1 and saved_images_per_class[class_name] < num_images_per_class:
#                 class_folder = os.path.join(parent_folder, class_name)
#                 if not os.path.exists(class_folder):
#                     os.makedirs(class_folder)

#                 image_save_path = os.path.join(class_folder, f"{saved_images_per_class[class_name]:03d}.jpg")
#                 image_pil = transforms.ToPILImage()(image.squeeze(0))  # Convert Tensor to PIL Image
#                 image_pil.save(image_save_path)

#                 saved_images_per_class[class_name] += 1

#         # Save images that do not belong to any class
#         if labels.sum().item() == 0 and saved_no_class_images < num_images_per_class:
#             no_class_folder = os.path.join(parent_folder, 'no_class')
#             if not os.path.exists(no_class_folder):
#                 os.makedirs(no_class_folder)

#             no_class_image_save_path = os.path.join(no_class_folder, f"{saved_no_class_images:03d}.jpg")
#             image_pil.save(no_class_image_save_path)
#             saved_no_class_images += 1

#         # Break if we have saved enough images for all classes
#         if all(count >= num_images_per_class for count in saved_images_per_class.values()) and saved_no_class_images >= num_images_per_class:
#             break

#     print(f"Saved {num_images_per_class} images per class and {num_images_per_class} images with no class in {parent_folder}")

# # Call the function to save images
# save_random_images_per_class(combined_dataset, class_list, num_images_per_class=10)


# Assuming parent_folder and image_pil (PIL image) are defined and initialized appropriately
# Track saved images per class


num_images_per_class = 10
saved_images_per_class = {class_name: 0 for class_name in class_list}
saved_no_class_images = 0

parent_folder = os.path.join(os.path.dirname(__file__), 'No_Assigned_Class')
os.makedirs(parent_folder, exist_ok=True)

# Loop through the DataLoader
for images, labels_batch in data_loader:
    labels_batch = labels_batch.float() if not isinstance(labels_batch, torch.Tensor) else labels_batch

    # Iterate through each image and label in the batch
    for i, (image, labels) in enumerate(zip(images, labels_batch)):
        if labels.sum().item() == 0:
            # Save only up to the specified limit
            if saved_no_class_images < num_images_per_class:
                no_class_folder = os.path.join(parent_folder, 'no_class')
                os.makedirs(no_class_folder, exist_ok=True)

                # Save the image
                no_class_image_save_path = os.path.join(no_class_folder, f"{saved_no_class_images:03d}.jpg")
                image_pil = transforms.ToPILImage()(image)  # Convert the tensor to a PIL image
                image_pil.save(no_class_image_save_path)
                saved_no_class_images += 1
