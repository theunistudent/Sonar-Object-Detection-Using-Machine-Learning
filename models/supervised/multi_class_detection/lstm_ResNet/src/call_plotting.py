import os
import pandas as pd
import numpy as np
import sys
import os

import sys
import os
sys.path.append(os.path.abspath("/mnt/sda2/liam/SONAR_thesis/models/supervised/multi_class_detection/lstm_ResNet/src"))

# from models.supervised.multi_class_detection.lstm_ResNet.src.lstm_metrics import plot_training_metrics
from lstm_metrics import plot_training_metrics

training_metrics = os.path.join(os.path.dirname(__file__), '../metrics/')
training_metrics_location = os.path.join(os.path.dirname(__file__), '../metrics/training_metrics.csv')


metrics_labels = np.array(["Epoch", "Loss", "Accuracy", "Hamming Loss", "Precision", "Recall", "F1-Score"])


if os.path.isfile(training_metrics_location):
    metrics = pd.read_csv(training_metrics_location).to_numpy()

# Save metrics to csv file
df = pd.DataFrame(data=metrics, columns=metrics_labels)
df.to_csv(training_metrics_location, index=False)

# Plot the metrics
plot_training_metrics(metrics, df)

print('Done!')