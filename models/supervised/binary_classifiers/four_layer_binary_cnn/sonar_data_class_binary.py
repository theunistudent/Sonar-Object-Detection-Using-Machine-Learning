import pandas as pd
from PIL import Image
import cv2
import datetime

import torch
from torch.utils.data import Dataset

class SonarDataset(Dataset):
    def __init__(self, csv_file, root_dir, device, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        man_made = torch.zeros(1, dtype=torch.float32)
        # initialise to non-man-made
        man_made[0] = 0

        img_name = f"{self.root_dir}/{idx:05d}.jpg"
        image = Image.open(img_name)

        # Get the number of columns
        shape = self.data_frame.shape
        num_columns = shape[1]        

        # Iterate through columns to check for bounding boxes
        for j in range(1, num_columns):
            if pd.notna(self.data_frame.iloc[idx, j]):
                man_made[0] = 1
                break  # If any bounding box is found, set man_made to 1 and break the loop

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        # Print statements for debugging
        # print(f"Index: {idx}, Label: {self.man_made}")

        return image, man_made
    


class SonarVideo(Dataset):
    def __init__(self, csv_path, video_path, device, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.data_frame = pd.read_csv(csv_path)
        self.device = device

        # Convert timestamps to seconds
        # self.data_frame['timestamp'] = self.data_frame[0].apply(self._timestamp_to_seconds)

        # Load video using OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
        
    def _timestamp_to_seconds(self, timestamp):
        # Parse the MM:SS.SSS format and convert to total seconds
        time_obj = datetime.datetime.strptime(timestamp, "%M:%S.%f")
        total_seconds = time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
        return total_seconds

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        man_made = torch.tensor([0.0], dtype=torch.float32)
        # Get the number of columns
        shape = self.data_frame.shape
        num_columns = shape[1]        
        # Iterate through columns to check for bounding boxes
        for j in range(2, num_columns):
            if pd.notna(self.data_frame.iloc[idx, j]):
                man_made[0] = 1
                break  # If any bounding box is found, set man_made to 1 and break the loop

        # Get the corresponding timestamp and label
        timestamp = self.data_frame.iloc[idx, 0]
        timestamp_seconds = self._timestamp_to_seconds(timestamp)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)  # Assuming timestamp is in seconds
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame at timestamp {timestamp}")
        # Convert frame from BGR (OpenCV format) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # Convert frame to PyTorch tensor
         # Convert frame to PIL Image for transformations
        frame = Image.fromarray(frame)

        if self.transform:
            frame = self.transform(frame)


        return frame, man_made
        