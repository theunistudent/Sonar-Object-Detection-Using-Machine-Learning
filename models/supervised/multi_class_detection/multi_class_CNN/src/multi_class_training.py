import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import configparser
import multiprocessing
import logging
import ast

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F


from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, precision_recall_fscore_support, multilabel_confusion_matrix

from multi_class_CNN_data import SonarImages, SonarVideo, CombinedSonarDataset
from multi_class_model import Model
from plotting_metrics import plot_training_metrics

# Initialize configparser
config = configparser.ConfigParser()
config.read('/mnt/sda2/liam/SONAR_thesis/models/supervised/multi_class_detection/multi_class_CNN/src/config.ini')

# Load paths from the configuration file
sonar_training_data = config['Paths']['sonar_training_data']

# Load class list (as a list of strings)
class_list = config['Classes']['class_list'].split(',')

# Load training parameters from the config file
batch_size = config.getint('Training', 'batch_size')
shuffle = config.getboolean('Training', 'shuffle')
pin_memory = config.getboolean('Training', 'pin_memory')
epochs = config.getint('Training', 'epochs')
lr = config.getfloat('Training', 'lr')
momentum = config.getfloat('Training', 'momentum')
weight_decay = config.getfloat('Training', 'weight_decay')
train_ratio = config.getfloat('Training', 'train_ratio')
num_workers = config.getint('Training', 'num_workers')
class_weights_str = config.get('Training', 'fixed_class_weights')


# Directories relative to file location
relative_checkpoint_dir = os.path.join(os.path.dirname(__file__), '../checkpoints/')
relative_historic_checkpoint_dir = os.path.join(relative_checkpoint_dir, 'historic_checkpoints/')
relative_checkpoint_path = os.path.join(relative_checkpoint_dir, "current_checkpoint.tar")
training_metrics_location = os.path.join(os.path.dirname(__file__), '../metrics/training_metrics.csv')
training_metrics = os.path.join(os.path.dirname(__file__), '../metrics/')

# check multi processing is working
print(f"Available CPU cores: {multiprocessing.cpu_count()}")

# Create necessary directories
os.makedirs(relative_checkpoint_dir, exist_ok=True)
os.makedirs(relative_historic_checkpoint_dir, exist_ok=True)
os.makedirs(training_metrics, exist_ok=True)


# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
torch.cuda.empty_cache()

# Model, Loss, Optimizer, Scaler
model = Model(1080, 1920, class_list).to(device=device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scaler = GradScaler()
# Weigts for dealing with class Imbalance
fixed_class_weights = torch.tensor(ast.literal_eval(class_weights_str)).to(device)

# Transformations
transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor(),
    transforms.Grayscale(1),
    transforms.Normalize((0.5,), (0.5,)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
])

# # print cpu memory usage
# print(f"{torch.cuda.memory_summary(device=None, abbreviated=False)}")

# Initialize epoch and loss
epoch = 0
loss = None

# Load checkpoints from previous training if they exist
if os.path.exists(relative_checkpoint_path):
    checkpoint = torch.load(relative_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded successfully. Resuming from epoch {epoch+1} with loss {loss}.")
else:
    print("No checkpoint found at", relative_checkpoint_path)

# Arrays for saving performance metrics
metrics_labels = np.array(["Epoch", "test_loss", "accuracy", "hamming", "precision", "recall", "f1"])
metrics = np.empty(0)

# load existing metrics if they exist
if os.path.isfile(training_metrics_location):
    metrics = pd.read_csv(training_metrics_location).to_numpy()

# -------------------- Create DataLoaders -------------------------
# paths to parrent folders
images = os.path.join(sonar_training_data, 'images')
videos = os.path.join(sonar_training_data, 'videos')

# list for storing data loaders
image_dataset = []
video_dataset = []

# check that there are sonar images in training data folder
if os.path.exists(images):
    # nested for loops deal with shipwreck data set structure
    for current_folder in os.listdir(images):
        # scroll through the folders
        for folder in os.listdir(os.path.join(images,current_folder)):
            csv_file = ""
            img_dir = f"{images}/{current_folder}/{folder}/sonar"
            for file in os.listdir(f"{images}/{current_folder}/{folder}"):
                file_name, file_extension = os.path.splitext(file)
                if file_extension == f".csv" and file_name != f"IMULog":
                    csv_file = f"{images}/{current_folder}/{folder}/{file}"

            if csv_file == None:
                logging.error(f'no csv found in image folder: {os.path.join(current_folder, folder)}')
            # Add to list of data loaders
            image_dataset.append(SonarImages(class_list, csv_file, img_dir, transform=transform))
else:
    logging.warning('No images directory in sonar_training_data')

# check that there are sonar videos in training data folder
if os.path.exists(videos):
    # each sub folder should only contain one avi video and one csv

    for current_folder in os.listdir(videos):
        csv_file = None
        video_file = None
        
        current_folder_path = os.path.join(videos, current_folder)
        
        # Check if the current folder is indeed a directory
        if os.path.isdir(current_folder_path):
            for file in os.listdir(current_folder_path):
                file_name, file_extension = os.path.splitext(file)  # Use 'file' here, not os.listdir
                if file_extension == ".csv" and file_name != "IMULog":
                    csv_file = os.path.join(current_folder_path, file)
                elif file_extension == ".avi":  # Use elif to ensure only one of the two can be assigned
                    video_file = os.path.join(current_folder_path, file)
                
                # Only break if both files have been found
                if csv_file is not None and video_file is not None:
                    break

        # check for issues
        if csv_file == None or video_file == None:
            logging.error(f'video file or csv not loaded from: {os.path.join(videos, current_folder)}')

        video_dataset.append(SonarVideo(class_list, csv_file, video_file, transform=transform))

else:
    logging.warning('No video directory in sonar_training_data')

# class for combining both data sets into one
combined_dataset = CombinedSonarDataset(image_datasets=image_dataset, video_datasets=video_dataset)

# Set up the split ratio for training and testing
train_size = int(train_ratio * len(combined_dataset))
test_size = len(combined_dataset) - train_size

# Generate the indices for the split
indices = list(range(len(combined_dataset)))

# Split into training and testing indices
train_indices, test_indices = indices[:train_size], indices[train_size:]

# Create subsets for training and testing
train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
test_dataset = torch.utils.data.Subset(combined_dataset, test_indices)

# Create DataLoader for the training dataset
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=shuffle,  # Enable shuffling if required
    num_workers=num_workers, 
    pin_memory=pin_memory
)

# Create DataLoader for the test dataset (no shuffling in test loader)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,  # No shuffling for test loader
    num_workers=num_workers, 
    pin_memory=pin_memory
)

# ------------------- TRAINING -----------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    device_type = str(device).split(":")[0]  # Extracts 'cuda' or 'cpu' from 'cuda:0' or 'cpu'

    scaler = GradScaler()  # Initialize mixed precision scaler

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # # Checking if different labels exist
        # value_to_check = 1
        # exists_in_column = any(row[0] == value_to_check for row in y)
        # if exists_in_column:
        #     print(y)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(device_type=device_type):
            pred = model(X)

            # Calculate the class weighting loss
            # Apply sigmoid to get probabilities and calculate weighted BCE loss
            weights = fixed_class_weights.unsqueeze(0).expand_as(y)  # Match weights to batch size and shape of y
            loss = F.binary_cross_entropy_with_logits(pred, y, weight=weights)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        if batch % 10 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    return loss.item()


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct, total = 0, 0

    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Apply sigmoid and threshold at 0.5 to get binary predictions
            pred_labels = (torch.sigmoid(pred) >= 0.5).float()

            # Store predictions and true labels for metric calculations
            all_true_labels.append(y.cpu().numpy())
            all_pred_labels.append(pred_labels.cpu().numpy())

            # Compare predictions with true labels to calculate accuracy
            correct += (pred_labels == y).sum().item()
            total += y.numel()  # Number of elements in y (total binary labels)

    # Calculate metrics over the entire dataset
    test_loss /= num_batches
    accuracy = 100 * correct / total

    # Convert accumulated true and predicted labels to numpy arrays
    y_true = np.concatenate(all_true_labels, axis=0)
    y_pred = np.concatenate(all_pred_labels, axis=0)

    # Calculate Hamming Loss
    hamming = hamming_loss(y_true, y_pred)

    # Calculate Precision, Recall, and F1-Score (with micro averaging)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    # Calculate Multi-Label Confusion Matrix
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)

    # Display the results
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Multi-Label Confusion Matrix:\n{conf_matrix}\n")

    return test_loss, accuracy, hamming, precision, recall, f1, conf_matrix

def checkpoint_save(path, epoch, model, optimizer, Loss):
    # save the model for future training each epoch
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
                }, path)

for t in range(epochs):
    Loss = 0
    Loss = train(train_loader, model, loss_fn, optimizer)

    print(f"\n-------------------------------\n Epoch {epoch + t+1}")  # Acount for the epochs in prevoiuse and current training
    [test_loss, accuracy, hamming, precision, recall, f1, conf_matrix] = test(test_loader, model, loss_fn)

    # save the model for future training each epoch
    checkpoint_save(relative_checkpoint_path, epoch+t, model, optimizer, Loss)
    # save each epoch for future use
    relative_historic_checkpoint_path = os.path.join(relative_historic_checkpoint_dir, f"man_made_checkpoint_epoch{epoch + t+1}.tar")
    checkpoint_save(relative_historic_checkpoint_path, epoch+t, model, optimizer, Loss)
    
    # save state directory for inference
    state_directroy_path = os.path.join(relative_checkpoint_dir, f"state_dict_model.pt")
    torch.save(model.state_dict(), state_directroy_path)

     # Save metrics
    if metrics.size == 0:
        metrics = np.array([[epoch+t+1, test_loss, accuracy, hamming, precision, recall, f1]])
    else:
        metrics = np.vstack([metrics, [[epoch+t+1, test_loss, accuracy, hamming, precision, recall, f1]]])

    # Save metrics to csv file
    df = pd.DataFrame(data=metrics, columns=metrics_labels)
    df.to_csv(training_metrics_location, index=False)

# plot metrics
total_epochs = np.arange(1, len(metrics) + 1)  # Assuming each row in the CSV represents an epoch

# Plot the metrics
plot_training_metrics(metrics, df)

print('Done!')