import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(metrics, df):
    """
    Plot training metrics including Accuracy, Precision, Recall, and Loss over epochs.

    Parameters:
    - metrics (array-like): An array or list representing the total epochs.
    - df (Pandas DataFrame): A dataframe containing the columns 'Accuracy', 'Precision', 'Recall', and 'Loss'.
    """
    # Convert metrics (epochs) to numpy array
    total_epochs = np.arange(1, len(metrics) + 1)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Accuracy, Precision, Recall on the left axis
    line1, = ax1.plot(total_epochs, df['Accuracy'], 'b-', label='Accuracy', marker='o')
    line2, = ax1.plot(total_epochs, df['Precision'], 'g-', label='Precision', marker='o')
    line3, = ax1.plot(total_epochs, df['Recall'], 'm-', label='Recall', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy / Precision / Recall', color='k')  # Use a neutral color for the label
    ax1.tick_params(axis='y', labelcolor='k')

    # Plot Loss on the right axis
    ax2 = ax1.twinx()
    line4, = ax2.plot(total_epochs, df['Loss'], 'r-', label='Loss', marker='o')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add the title
    plt.title('Training Metrics Over Epochs')

    # Combine legends from both axes and place them outside the plot
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the external legend
    plt.grid(True)
    plt.show()
