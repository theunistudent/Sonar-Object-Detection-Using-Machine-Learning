import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(metrics, df):
    """
    Plot training metrics including Accuracy, Precision, Recall, F1-Score, Hamming Loss, and Loss over epochs.

    Parameters:
    - metrics (array-like): An array or list representing the total epochs.
    - df (Pandas DataFrame): A dataframe containing the columns 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Hamming Loss', and 'Loss'.
    """
    # Convert metrics (epochs) to numpy array
    total_epochs = np.arange(1, len(metrics) + 1)

    # Create a figure with multiple subplots to separate different metrics
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # Plot Accuracy on the first subplot
    # Plot Accuracy on the first subplot
    ax[0, 0].plot(total_epochs, df['Accuracy'], 'b-', label='Accuracy', marker='o')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Accuracy (%)')
    ax[0, 0].set_title('Accuracy Over Epochs')
    ax[0, 0].set_ylabel('Accuracy (%)')
    ax[0, 0].set_title('Accuracy Over Epochs')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Plot Loss on the second subplot
    ax[0, 1].plot(total_epochs, df['Loss'], 'r-', label='Loss', marker='o')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].set_title('Loss Over Epochs')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # Plot Hamming Loss on the third subplot
    ax[1, 0].plot(total_epochs, df['Hamming Loss'], 'y-', label='Hamming Loss', marker='o')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Hamming Loss')
    ax[1, 0].set_title('Hamming Loss Over Epochs')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plot Precision, Recall, and F1-Score on the fourth subplot
    ax[1, 1].plot(total_epochs, df['Precision'], 'g-', label='Precision', marker='o')
    ax[1, 1].plot(total_epochs, df['Recall'], 'm-', label='Recall', marker='o')
    ax[1, 1].plot(total_epochs, df['F1-Score'], 'c-', label='F1-Score', marker='o')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('Metrics')
    ax[1, 1].set_title('Precision, Recall, and F1-Score Over Epochs')
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    # # Plot Precision, Recall, and F1-Score on the fourth subplot
    # ax[1, 1].plot(total_epochs, df['Precision'], 'g-', label='Precision', marker='o')
    # ax[1, 1].plot(total_epochs, df['Recall'], 'm-', label='Recall', marker='o')
    # ax[1, 1].plot(total_epochs, df['F1-Score'], 'c-', label='F1-Score', marker='o')
    # ax[1, 1].set_xlabel('Epoch')
    # ax[1, 1].set_ylabel('Metrics')
    # ax[1, 1].set_title('Precision, Recall, and F1-Score Over Epochs')
    # ax[1, 1].legend()
    # ax[1, 1].grid(True)

    # Add the title for the entire figure
    plt.suptitle('Training Metrics Over Epochs, 18 Layer Res-Net with LSTM', fontsize=16)

    plt.suptitle('Training Metrics Over Epochs, 18 Layer Res-Net with LSTM', fontsize=16)

    plt.suptitle('Training Metrics Over Epochs, 18 Layer Res-Net with LSTM', fontsize=16)

    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the external title
    plt.show()


def plot_confusion_matrix_values(epochs, confusion_matrix_values):
    """
    Plot the values of True Positives, False Positives, True Negatives, and False Negatives over epochs.

    Parameters:
    - epochs (array-like): An array or list representing the total epochs.
    - confusion_matrix_values (ndarray): A 2D array where each row corresponds to the values [TP, FP, TN, FN] for each epoch.
    """
    # Extract the confusion matrix values for each metric
    tp = confusion_matrix_values[:, 0]
    fp = confusion_matrix_values[:, 1]
    tn = confusion_matrix_values[:, 2]
    fn = confusion_matrix_values[:, 3]

    # Create a new figure for the confusion matrix values
    plt.figure(figsize=(10, 6))

    # Plot True Positives, False Positives, True Negatives, and False Negatives
    plt.plot(epochs, tp, 'g-', label='True Positives (TP)', marker='o')
    plt.plot(epochs, fp, 'r-', label='False Positives (FP)', marker='o')
    plt.plot(epochs, tn, 'b-', label='True Negatives (TN)', marker='o')
    plt.plot(epochs, fn, 'm-', label='False Negatives (FN)', marker='o')

    # Set labels, title, and legend
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Values Over Epochs')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Assuming 'conf_matrix_values' contains the confusion matrix values for all epochs
# Format of 'conf_matrix_values': Each row contains [TP, FP, TN, FN] for a specific epoch
# Example: np.array([[10, 2, 50, 5], [12, 1, 52, 4], ...])
conf_matrix_values = np.array([[10, 2, 50, 5], [12, 1, 52, 4], [14, 3, 53, 6]])  # Example data
total_epochs = np.arange(1, conf_matrix_values.shape[0] + 1)


def plot_confusion_matrix_values(epochs, confusion_matrix_values):
    """
    Plot the values of True Positives, False Positives, True Negatives, and False Negatives over epochs.

    Parameters:
    - epochs (array-like): An array or list representing the total epochs.
    - confusion_matrix_values (ndarray): A 2D array where each row corresponds to the values [TP, FP, TN, FN] for each epoch.
    """
    # Extract the confusion matrix values for each metric
    tp = confusion_matrix_values[:, 0]
    fp = confusion_matrix_values[:, 1]
    tn = confusion_matrix_values[:, 2]
    fn = confusion_matrix_values[:, 3]

    # Create a new figure for the confusion matrix values
    plt.figure(figsize=(10, 6))

    # Plot True Positives, False Positives, True Negatives, and False Negatives
    plt.plot(epochs, tp, 'g-', label='True Positives (TP)', marker='o')
    plt.plot(epochs, fp, 'r-', label='False Positives (FP)', marker='o')
    plt.plot(epochs, tn, 'b-', label='True Negatives (TN)', marker='o')
    plt.plot(epochs, fn, 'm-', label='False Negatives (FN)', marker='o')

    # Set labels, title, and legend
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Values Over Epochs')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Assuming 'conf_matrix_values' contains the confusion matrix values for all epochs
# Format of 'conf_matrix_values': Each row contains [TP, FP, TN, FN] for a specific epoch
# Example: np.array([[10, 2, 50, 5], [12, 1, 52, 4], ...])
conf_matrix_values = np.array([[10, 2, 50, 5], [12, 1, 52, 4], [14, 3, 53, 6]])  # Example data
total_epochs = np.arange(1, conf_matrix_values.shape[0] + 1)
