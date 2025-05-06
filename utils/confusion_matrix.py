# Import necessary libraries
import torch # PyTorch for building and evaluating the model
import matplotlib.pyplot as plt # For plotting the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Tools for creating and visualizing confusion matrices

# Define a function to plot a confusion matrix for a given model and test dataset
def plot_confusion_matrix(model, xtest, ytest, device, normalize=False, save_path=None):
    """
    Generates and displays (or saves) a confusion matrix for a given classification model.
    
    Parameters:
    - model: Trained PyTorch model to evaluate.
    - xtest: Test features (tensor).
    - ytest: True labels for the test set (tensor).
    - device: Device to run inference on (e.g., 'cpu' or 'cuda').
    - normalize: Whether to normalize the confusion matrix.
    - save_path: Optional path to save the generated confusion matrix image.
    """

    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # Lists to store actual and predicted labels
    y_true = []
    y_pred = []

    # Disable gradient calculation to speed up inference and reduce memory usage
    with torch.no_grad():
        # Create a DataLoader to batch the test data (no shuffling for consistent results)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xtest, ytest), # Wrap test data in a dataset
            batch_size=128, # Number of samples per batch
            shuffle=False # Keep order of data intact
        )

        # Iterate through the DataLoader to perform inference
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device) # Move data to the appropriate device (CPU/GPU)
            outputs = model(xb.float()) # Forward pass through the model
            preds = outputs.argmax(1) # Get predicted class labels (index of highest score)
            y_true.extend(yb.cpu().numpy()) # Collect actual labels
            y_pred.extend(preds.cpu().numpy()) # Collect predicted labels

    # Create a confusion matrix using sklearn (normalized)
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10)) # Create a large figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) # Prepare the confusion matrix display object
    disp.plot(ax=ax, cmap='Blues', values_format='.2f' if normalize else 'd') # Plot with formatting

    # Set the plot title based on normalization flag
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plt.title(f"{title} for Traffic Sign Recognition", fontsize=16)

    # Save the plot to disk if a path is provided; otherwise, display it on screen
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()




