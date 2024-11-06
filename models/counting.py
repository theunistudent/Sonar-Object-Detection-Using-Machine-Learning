import os
import logging
import pandas as pd
from collections import Counter

# counting instances of classes for thesis report

sonar_training_data = '/mnt/sda2/liam/SONAR_thesis/sonar_training_data'

# Paths to parent folders
images = os.path.join(sonar_training_data, 'images/')
videos = os.path.join(sonar_training_data, 'videos/')

# Function to count non-zero entries in columns of CSV files
def count_non_zero_entries(csv_files):
    column_counts = Counter()

    for csv_file in csv_files:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Loop through each column in the DataFrame
            for column in df.columns:
                if column in column_counts:  # Only count if the column already exists
                    non_zero_count = (df[column] != 0).sum()  # Count non-zero entries
                    column_counts[column] += non_zero_count
                else:
                    non_zero_count = (df[column] != 0).sum()  # Count non-zero entries
                    column_counts[column] = non_zero_count  # Initialize the count

        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")

    return column_counts

# Function to count total empty rows excluding the Time column
def count_total_empty_rows(csv_files):
    total_empty_rows = 0

    for csv_file in csv_files:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Exclude the Time column and count empty rows
            if 'Time' in df.columns:
                empty_rows = df.drop(columns=['Time']).isnull().all(axis=1).sum()
            else:
                empty_rows = df.isnull().all(axis=1).sum()

            total_empty_rows += empty_rows

        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")

    return total_empty_rows

# Initialize lists to store CSV file paths
csv_files = []

# Check that there are sonar images in the training data folder
if os.path.exists(images):
    # Nested for loops deal with the shipwreck dataset structure
    for current_folder in os.listdir(images):
        # Scroll through the folders
        for folder in os.listdir(os.path.join(images, current_folder)):
            img_dir = os.path.join(images, current_folder, folder, 'sonar')
            for file in os.listdir(os.path.join(images, current_folder, folder)):
                file_name, file_extension = os.path.splitext(file)
                if file_extension == ".csv" and file_name != "IMULog":
                    csv_file = os.path.join(images, current_folder, folder, file)
                    csv_files.append(csv_file)  # Add the CSV file to the list

            if not csv_files:
                logging.error(f'No CSV found in image folder: {os.path.join(current_folder, folder)}')

else:
    logging.warning('No images directory in sonar_training_data')

# Check that there are sonar videos in the training data folder
if os.path.exists(videos):
    # Each subfolder should only contain one avi video and one csv
    for current_folder in os.listdir(videos):
        csv_file = None
        video_file = None
        
        current_folder_path = os.path.join(videos, current_folder)
        
        # Check if the current folder is indeed a directory
        if os.path.isdir(current_folder_path):
            for file in os.listdir(current_folder_path):
                file_name, file_extension = os.path.splitext(file)
                if file_extension == ".csv" and file_name != "IMULog":
                    csv_file = os.path.join(current_folder_path, file)
                    csv_files.append(csv_file)  # Add the CSV file to the list
                elif file_extension == ".avi":
                    video_file = os.path.join(current_folder_path, file)
                
                # Only break if both files have been found
                if csv_file is not None and video_file is not None:
                    break

        # Check for issues
        if csv_file is None or video_file is None:
            logging.error(f'Video file or CSV not loaded from: {os.path.join(videos, current_folder)}')

else:
    logging.warning('No video directory in sonar_training_data')

# At this point, `csv_files` contains all the CSV file paths found in the image and video directories
print("Collected CSV files:", csv_files)

# Count non-zero entries in each column across all collected CSV files
non_zero_counts = count_non_zero_entries(csv_files)

# Count total empty rows excluding the Time column
total_empty_rows = count_total_empty_rows(csv_files)

# Output the results
print("Non-Zero Counts per Column:")
for column, count in non_zero_counts.items():
    print(f"{column}: {count}")

print(f"\nTotal Empty Rows (excluding 'Time' column): {total_empty_rows}")
