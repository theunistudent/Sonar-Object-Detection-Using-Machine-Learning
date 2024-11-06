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
    ax[0, 0].plot(total_epochs, df['Accuracy'], 'b-', label='Accuracy', marker='o')
    ax[0, 0].set_xlabel('Epoch')
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

    # Add the title for the entire figure
    plt.suptitle('Training Metrics Over Epochs, 5 CNN Blocks', fontsize=16)

    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the external title
    plt.show()
