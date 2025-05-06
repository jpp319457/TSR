# Import necessary libraries
import os # For interacting with the file system (e.g., building file paths)
import PIL # Python Imaging Library for loading and handling image files
import numpy as np # For numerical operations and handling image data in array form

# Function to load image data from a directory structure
def load_data(data_dir, transform=None):
    """
    Loads image data from a directory where each subfolder represents a class.

    Parameters:
    - data_dir: The root directory containing subfolders (each named by class ID).
    - transform: A function or transform (e.g., from torchvision) to apply to each image.

    Returns:
    - A tuple of two NumPy arrays:
        x: Array of image data.
        y: Array of corresponding labels (class IDs).
    """

    x, y = [], [] # Lists to hold image data and labels

    # Loop through each class folder (assuming folder names are 0 through 42)
    for folder in range(43):
        folder_path = os.path.join(data_dir, str(folder)) # Full path to the current class folder

        # Loop through each image file in the folder
        for i, img in enumerate(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img) # Construct full path to the image

            # Open the image and apply the transformation (e.g., resize, convert to tensor)
            img_tensor = transform(PIL.Image.open(img_path))

            # Convert transformed image tensor to NumPy array and store it
            x.append(img_tensor.numpy())

            # Store the label (folder name corresponds to class ID)
            y.append(folder)

        # Print number of images loaded from this folder (for progress tracking)
        print(f'Folder {folder}: {i + 1} images loaded.')

    # Debug information
    print(type(x)) # Should be a list of NumPy arrays (one per image)
    print(len(x)) # Total number of images loaded
    print([arr.shape for arr in x if isinstance(arr, np.ndarray)]) # Shapes of individual image arrays

    # Convert image and label lists into NumPy arrays for easier handling later
    return np.array(x), np.array(y)


