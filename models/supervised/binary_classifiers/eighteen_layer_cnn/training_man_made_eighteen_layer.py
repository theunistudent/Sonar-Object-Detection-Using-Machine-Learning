import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.transforms import v2

from sonar_data_class_binary import SonarDataset, SonarVideo
from eighteen_layer_model import ManMade
from plotting_metrics import plot_training_metrics

# Training parameters
batch_size = 15
epochs = 120
lr = 0.001
momentum = 0.9
weight_decay = 0.01
train_ratio = 0.85

# num_workers > 0 breaks when attempting to enumerate the data loader
# I am unsure why it used to work.
num_workers = 0
# num_workers = multiprocessing.cpu_count()  # Automatically use all available CPU cores

import multiprocessing
print(f"Available CPU cores: {multiprocessing.cpu_count()}")



# Directories and paths
shipwreck_data = '/mnt/sda2/liam/SONAR_thesis/liam-training-data/Data'
fishmarket_video_path = '/mnt/sda2/liam/SONAR_thesis/liam-training-data/FUSION_20240619_120202_exported_1.avi'
fish_market_labels = '/mnt/sda2/liam/SONAR_thesis/liam-training-data/fishMarket.csv'
relative_checkpoint_dir = "models/supervised/eighteen_layer_cnn/checkpoints"
relative_historic_checkpoint_dir = "models/supervised/eighteen_layer_cnn/checkpoints/historic_checkpoints"
metrics_location = f"{relative_checkpoint_dir}/training_metrics.csv"
relative_checkpoint_path = os.path.join(relative_checkpoint_dir, "eighteen_layer_cnn_checkpoint.tar")


# Create necessary directories
os.makedirs(relative_checkpoint_dir, exist_ok=True)

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
torch.cuda.empty_cache()

# Model, Loss, Optimizer, Scaler
model = ManMade(1080, 1920).to(device=device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scaler = GradScaler()

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
metrics_labels = np.array(["Epoch", "Accuracy", "Loss", "Precision", "Recall"])
metrics = np.empty(0)

# load existing metrics if they exist
if os.path.isfile("models/supervised/five_layer_binary_cnn/training_metrics.csv"):
    metrics = pd.read_csv("models/supervised/five_layer_binary_cnn/training_metrics.csv").to_numpy()

# ------------------- TRAINING -----------------------
def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        device_type = str(device).split(":")[0]  # Extracts 'cuda' or 'cpu' from 'cuda:0' or 'cpu'
    
        for batch, (X, y) in enumerate(dataloader):

            # print("labels:", y) # print the labels for debuging
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

             # Forward pass with mixed precision
            with autocast(device_type=device_type):
                pred = model(X)
                loss = loss_fn(pred, y)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, total = 0, 0, 0
    true_positives, false_positives, false_negatives = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Apply sigmoid and threshold at 0.5 to get binary predictions
            pred_labels = (torch.sigmoid(pred) >= 0.5).float()

            # Compare predictions with true labels
            correct += (pred_labels == y).sum().item()
            total += y.size(0)

            # Calculate true polosssitives, false positives, and false negatives
            true_positives += ((pred_labels == 1) & (y == 1)).sum().item()
            false_positives += ((pred_labels == 1) & (y == 0)).sum().item()
            false_negatives += ((pred_labels == 0) & (y == 1)).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / total
    
    # Calculate precision and recall
    precision = 100 * true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = 100 * true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"Precision: {precision:>0.1f}%, Recall: {recall:>0.1f}%\n")

    return test_loss, accuracy, precision, recall

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
    # Initialize a list to store all test datasets
    all_test_datasets = []
    train_dataset = []
    test_dataset = []

    sonar_dataset  = SonarVideo(fish_market_labels, fishmarket_video_path, transform=transform)
    print(f"The dataset contains {len(sonar_dataset)} samples.")

    # split dataset into training and testing sets
    dataset_length = len(sonar_dataset)
    train_length = int(train_ratio * dataset_length)
    test_length = dataset_length - train_length
    train_dataset, test_dataset = random_split(sonar_dataset, [train_length, test_length])

    # Add the current test dataset to the list
    all_test_datasets.append(test_dataset)

    # Create a DataLoaders
    # The Try catch is to deal with the loaders not having any images due to small recording. 
    # Just ignoring because the small amount of data is insignificant
    # Create DataLoader for the train dataset
    video_name = os.path.basename(fishmarket_video_path)
    if train_length > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,  pin_memory=True)
        # Training in folder
        print(f"Recording {video_name}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
    else:
        print(f"{video_name} does not contain enough data for training")



    for date_folder in os.listdir(shipwreck_data):
        parent_folder = f"{shipwreck_data}/{date_folder}"
        # scroll through the folders
        for folder in os.listdir(parent_folder):
            csv_file = ""
            img_dir = f"{parent_folder}/{folder}/sonar"

            for file in os.listdir(f"{parent_folder}/{folder}"):
                file_name, file_extension = os.path.splitext(file)
                if file_extension == f".csv" and file_name != f"IMULog":
                    csv_file = f"{parent_folder}/{folder}/{file}"

            # Instantiate the dataset
            sonar_dataset  = SonarDataset(csv_file, img_dir, transform=transform)
            print(f"The dataset contains {len(sonar_dataset)} samples.")

            # split dataset into training and testing sets
            dataset_length = len(sonar_dataset)
            train_length = int(train_ratio * dataset_length)
            test_length = dataset_length - train_length
            train_dataset, test_dataset = random_split(sonar_dataset, [train_length, test_length])

            # # debugging model learning
            # train_dataset= sonar_dataset
            # test_dataset = train_dataset


            # Add the current test dataset to the list
            all_test_datasets.append(test_dataset)

            # Create a DataLoaders
            # The Try catch is to deal with the loaders not having any images due to small recording. 
            # Just ignoring because the small amount of data is insignificant
            # Create DataLoader for the train dataset
            if train_length > 0:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,  pin_memory=True)
                # Training in folder
                print(f"Recording {folder}\n-------------------------------")
                train(train_loader, model, loss_fn, optimizer)
            else:
                print(f"{folder} does not contain enough data for training")
            
            # # Debuging
            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # # Training in folder
            # print(f"Recording {folder}\n-------------------------------")
            # train(train_loader, model, loss_fn, optimizer)            
        

    # Concatenate all test datasets into one larger dataset
    combined_test_dataset = ConcatDataset(all_test_datasets)
    # Create DataLoader for the combined test dataset
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # Testing for epoch
    print(f"\n-------------------------------\n Epoch {epoch + t+1}")  # Acount for the epochs in prevoiuse and current training
    [Loss, Accuracy, Precision, Recall] = test(combined_test_loader, model, loss_fn)

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
        metrics = np.array([[epoch+t+1, Accuracy, Loss, Precision, Recall]])
    else:
        metrics = np.vstack([metrics, [[epoch+t+1, Accuracy, Loss, Precision, Recall]]])

    # Save metrics to csv file
    df = pd.DataFrame(data=metrics, columns=metrics_labels)
    df.to_csv(metrics_location, index=False)

# plot metrics
total_epochs = np.arange(1, len(metrics) + 1)  # Assuming each row in the CSV represents an epoch

# Plot the metrics
plot_training_metrics(metrics, df)

print(f"Done!")




