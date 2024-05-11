import matplotlib.pyplot as plt

def create_learning_curve(num_epochs, epoch_loss, test_loss, accuracy):
    # Create a list of epoch numbers
    epochs = range(1, num_epochs + 1)

    # Create a figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot loss
    ax1.plot(epochs, epoch_loss, label='Training Loss')
    ax1.plot(epochs, test_loss, label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, accuracy, label='Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Save the figure
    fig.savefig('learning_curves.pdf')