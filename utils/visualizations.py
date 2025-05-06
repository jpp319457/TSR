import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def plot_preprocessing_pipeline(data_dir, transform):
    """Display one original image and several augmented versions using the transform pipeline."""
    
    # Collect all image file paths
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(('.png', '.jpg', '.ppm')):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print("No image found to visualize preprocessing.")
        return

    # Randomly select one image
    sample_path = random.choice(image_paths)
    original_img = Image.open(sample_path)

    # Plot original image and 4 augmented samples
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(1, 5):
        transformed = transform(original_img).permute(1, 2, 0)
        axes[i].imshow(transformed)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')

    plt.suptitle("Preprocessing Pipeline: Augmented Samples", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_oversampling_distribution(y_before, y_after):
    """Visualize class distribution before and after applying oversampling."""
    
    classes = np.arange(43)
    _, counts_before = np.unique(y_before, return_counts=True)
    _, counts_after = np.unique(y_after, return_counts=True)

    # Side-by-side bar plots for before/after comparison
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].bar(classes, counts_before)
    ax[0].set_title("Before Oversampling")
    ax[0].set_xlabel("Class")
    ax[0].set_ylabel("Count")

    ax[1].bar(classes, counts_after)
    ax[1].set_title("After Oversampling")
    ax[1].set_xlabel("Class")
    ax[1].set_ylabel("Count")

    plt.suptitle("Class Distribution Before and After Oversampling", fontsize=16)
    plt.tight_layout()
    plt.show()
