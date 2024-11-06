import pandas as pd
from PIL import Image
import cv2
import datetime
import random

import torch
from torch.utils.data import Dataset

class SonarImages(Dataset):
    def __init__(self, class_list, csv_file, path, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.path = path
        self.transform = transform
        self.class_list = class_list

        # Initialize an empty dictionary to store class locations
        self.class_locations = {}

        # Dictionary containing the columns for each class in the CSV file
        for name in self.class_list:
            name_with_num = f"{name}_1"
            for idx, heading in enumerate(self.data_frame.columns):
                if name_with_num == heading:
                    # Add the class and its index to the dictionary
                    self.class_locations[name] = idx
                    break

    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        # Open the required image
        img_name = f"{self.path}/{idx:05d}.jpg"
        image = Image.open(img_name)

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        # binary flags for presence of classes. initialise with zeros
        num_classes = len(self.class_list)
        classes_in_image = torch.zeros(num_classes, dtype=torch.float32)

        # check the CSV file to see if the current image contains the classes
        for name, possition in self.class_locations.items():
           classes_in_image[self.class_list.index(name)] = float(pd.notna(self.data_frame.iloc[idx, possition]))

        # print(f"Getting Images")

        return image, classes_in_image
                
class SonarVideo(Dataset):
    def __init__(self, class_list, csv_path, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.data_frame = pd.read_csv(csv_path)
        self.class_list = class_list

        # Initialize an empty dictionary to store class locations
        self.class_locations = {}

        # Dictionary containing the columns for each class in the CSV file
        for name in self.class_list:
            name_with_num = f"{name}_1"
            for idx, heading in enumerate(self.data_frame.columns):
                if name_with_num == heading:
                    # Add the class and its index to the dictionary
                    self.class_locations[name] = idx
                    break

        # Load video using OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Binary flags for presence of classes. Initialize with zeros
        num_classes = len(self.class_list)
        classes_in_frame = torch.zeros(num_classes, dtype=torch.float32)

        # Check the CSV file to see if the current frame contains the classes
        for name, position in self.class_locations.items():
            classes_in_frame[self.class_list.index(name)] = pd.notna(self.data_frame.iloc[idx, position])

        # Get the corresponding timestamp
        timestamp = self.data_frame.iloc[idx, 0]
        timestamp_seconds = self._timestamp_to_seconds(timestamp)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)  # Assuming timestamp is in seconds
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame at timestamp {timestamp}")

        # Convert frame to PIL Image for transformations
        frame = Image.fromarray(frame)

        if self.transform:
            frame = self.transform(frame)

        # print(f"Getting Video")

        return frame, classes_in_frame

    # def _timestamp_to_seconds(self, timestamp):
    #     # Parse the MM:SS.SSS format and convert to total seconds
    #     time_obj = datetime.datetime.strptime(timestamp, "%M:%S.%f")
    #     total_seconds = time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    #     return total_seconds
    
    def _timestamp_to_seconds(self, timestamp):
        # Check if the timestamp contains 'sec'
        if 'sec' in timestamp:
            # Strip ' sec' and convert to float
            return float(timestamp.replace(' sec', ''))
        else:
            # Otherwise, assume the format is '%M:%S.%f'
            try:
                time_obj = datetime.datetime.strptime(timestamp, "%M:%S.%f")
                return time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
            except ValueError as e:
                print(f"Error parsing timestamp: {timestamp}. Error: {e}")
                return 0  # or handle the error appropriately







    
    def __del__(self):
        # Release video capture object
        if self.cap.isOpened():
            self.cap.release()


# combine the image and video data for more randomised data and simplified training loops
# provide lists of data sets for both types
class CombinedSonarDataset(Dataset):
    def __init__(self, image_datasets=None, video_datasets=None):
        """
        image_datasets: List of image datasets (instances of SonarImages or similar).
        video_datasets: List of video datasets (instances of SonarVideo or similar).
        """
        self.image_datasets = image_datasets if image_datasets else []
        self.video_datasets = video_datasets if video_datasets else []

        # Calculate total length by summing the lengths of all datasets
        self.image_lengths = [len(ds) for ds in self.image_datasets]
        self.video_lengths = [len(ds) for ds in self.video_datasets]
        
        self.total_image_length = sum(self.image_lengths)
        self.total_video_length = sum(self.video_lengths)

        # Total length is the sum of both image and video datasets
        self.length = self.total_image_length + self.total_video_length

    def __len__(self):
        return self.length

    def _get_image_sample(self, idx):
        # Find which image dataset the index belongs to
        for i, length in enumerate(self.image_lengths):
            if idx < length:
                return self.image_datasets[i][idx]
            idx -= length
        raise IndexError("Index out of range for image datasets")

    def _get_video_sample(self, idx):
        # Find which video dataset the index belongs to
        for i, length in enumerate(self.video_lengths):
            if idx < length:
                return self.video_datasets[i][idx]
            idx -= length
        raise IndexError("Index out of range for video datasets")

    def __getitem__(self, idx):
        # Randomly select whether to return an image or a video frame
        if random.random() < (self.total_image_length / (self.length + 1e-6)):
            # Get a sample from the image datasets
            sample = self._get_image_sample(idx % self.total_image_length)
        else:
            # Get a sample from the video datasets
            sample = self._get_video_sample(idx % self.total_video_length)

        return sample