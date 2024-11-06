import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
import torch.optim as optim
from torch.utils.data import random_split, ConcatDataset

# local module imports
from sonar_data_class_binary import SonarDataset, SonarVideo
from man_made_model_five_layer import ManMade

classes = [
     "Man-Made",
     "Not-Man-Made",
]

# Chose which folder to train with here
shipwreck_data = 'C:/Users/VineyLiam/Documents/Thesis/3rd_party_data/shipwreck-dataset/data'
fishmarket_video_path = 'C:/Users/VineyLiam/Documents/Thesis/my-data/19-06-2024_fishmarket/FUSION_20240619_120202_exported_1.avi'
fish_market_labels = 'C:/Users/VineyLiam/Documents/Thesis/my-data/19-06-2024_fishmarket/fishMarket.csv'

# Training parameters
batch_size = 80
batch_shuffel = True
epochs = 20         # Extera epochs on top of previous training
train_ratio = 0.85
test_ratio = 1 - train_ratio
lr = 0.001
momentum = 0.9
weight_decay = 0.01

transform = transforms.Compose([
    transforms.Resize((507,1024)),
    transforms.ToTensor(),
    transforms.Grayscale(1),    # Some of the data comes as an rgb image, despite being sonar
    transforms.Normalize((0.5), (0.5)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
])

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Create Model
model = ManMade().to(device=device)
print(model)
# test the model shape after stack
print(model.get_output_shape((1, 1, 507,1024),device)) 

#Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) 

# Define the relative checkpoint directory
relative_checkpoint_dir = "models/supervised/five_layer_binary_cnn/checkpoints"
relative_checkpoint_path = os.path.join(relative_checkpoint_dir, "five_layer_binary_cnn_checkpoint.tar")
os.makedirs(relative_checkpoint_dir, exist_ok=True)
# for storing historic checkpoints
relative_historic_checkpoint_dir = "models/supervised/five_layer_binary_cnn/checkpoints/historic_checkpoints"
os.makedirs(relative_historic_checkpoint_dir, exist_ok=True)

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
        for batch, (X, y) in enumerate(dataloader):

            # print("labels:", y) # print the labels for debuging
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Apply sigmoid and threshold at 0.5 to get binary predictions
            pred_labels = (torch.sigmoid(pred) >= 0.5).float()

            # Compare predictions with true labels
            correct += (pred_labels == y).sum().item()
            total += y.size(0)

            # Calculate true positives, false positives, and false negatives
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

    sonar_dataset  = SonarVideo(fish_market_labels, fishmarket_video_path, device, transform=transform)
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Training in folder
        print(f"Recording {video_name}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
    else:
        print(f"{video_name} does not contain enough data for training")



    # for date_folder in os.listdir(shipwreck_data):
    #     parent_folder = f"{shipwreck_data}/{date_folder}"
    #     # scroll through the folders
    #     for folder in os.listdir(parent_folder):
    #         csv_file = ""
    #         img_dir = f"{parent_folder}/{folder}/sonar"

    #         for file in os.listdir(f"{parent_folder}/{folder}"):
    #             file_name, file_extension = os.path.splitext(file)
    #             if file_extension == f".csv" and file_name != f"IMULog":
    #                 csv_file = f"{parent_folder}/{folder}/{file}"

    #         # Instantiate the dataset
    #         sonar_dataset  = SonarDataset(csv_file, img_dir, device, transform=transform)
    #         print(f"The dataset contains {len(sonar_dataset)} samples.")

    #         # split dataset into training and testing sets
    #         dataset_length = len(sonar_dataset)
    #         train_length = int(train_ratio * dataset_length)
    #         test_length = dataset_length - train_length
    #         train_dataset, test_dataset = random_split(sonar_dataset, [train_length, test_length])

    #         # # debugging model learning
    #         # train_dataset= sonar_dataset
    #         # test_dataset = train_dataset


    #         # Add the current test dataset to the list
    #         all_test_datasets.append(test_dataset)

    #         # Create a DataLoaders
    #         # The Try catch is to deal with the loaders not having any images due to small recording. 
    #         # Just ignoring because the small amount of data is insignificant
    #         # Create DataLoader for the train dataset
    #         if train_length > 0:
    #             train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #             # Training in folder
    #             print(f"Recording {folder}\n-------------------------------")
    #             train(train_loader, model, loss_fn, optimizer)
    #         else:
    #             print(f"{folder} does not contain enough data for training")
            
    #         # # Debuging
    #         # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #         # # Training in folder
    #         # print(f"Recording {folder}\n-------------------------------")
    #         # train(train_loader, model, loss_fn, optimizer)            
        

    # Concatenate all test datasets into one larger dataset
    combined_test_dataset = ConcatDataset(all_test_datasets)
    # Create DataLoader for the combined test dataset
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)
    # Testing for epoch
    print(f"\n-------------------------------\n Epoch {epoch + t+1}")  # Acount for the epochs in prevoiuse and current training
    [Loss, Accuracy, Precision, Recall] = test(combined_test_loader, model, loss_fn)

    # save the model for future training each epoch
    checkpoint_save(relative_checkpoint_path, epoch+t, model, optimizer, Loss)
    # save each epoch for future use
    relative_historic_checkpoint_path = os.path.join(relative_historic_checkpoint_dir, f"man_made_checkpoint_epoch{epoch + t+1}.tar")
    checkpoint_save(relative_historic_checkpoint_path, epoch+t, model, optimizer, Loss)
    
    # Save metrics
    if metrics.size == 0:
        metrics = np.array([[epoch+t+1, Accuracy, Loss, Precision, Recall]])
    else:
        metrics = np.vstack([metrics, [[epoch+t+1, Accuracy, Loss, Precision, Recall]]])

    # Save metrics to csv file
    df = pd.DataFrame(data=metrics, columns=metrics_labels)
    df.to_csv("models/supervised/five_layer_binary_cnn/training_metrics.csv", index=False)

# plot metrics
total_epochs = np.arange(1, len(metrics) + 1)  # Assuming each row in the CSV represents an epoch

#Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(total_epochs, df['Accuracy'], 'b-', label='Accuracy', marker='o')
ax1.plot(total_epochs, df['Precision'], 'g-', label='Precision', marker='o')
ax1.plot(total_epochs, df['Recall'], 'm-', label='Recall', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy / Precision / Recall', color='k')  # Use a neutral color for the label
ax1.tick_params(axis='y', labelcolor='k')
ax2 = ax1.twinx()
ax2.plot(total_epochs, df['Loss'], 'r-', label='Loss', marker='o')
ax2.set_ylabel('Loss', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('Training Metrics Over Epochs')
fig.tight_layout()
plt.grid(True)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

print(f"Done!")




