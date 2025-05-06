# Project Documentation

## Overview
This project implements a traffic sign recognition system using a convolutional neural network (CNN). The system includes functionalities for training the model, real-time traffic sign detection using a webcam, and a modularized codebase for better maintainability.

---

## Recent Changes

### 1. **Added Files**
The following files were added to improve modularity and functionality:

#### **`model/traffic_sign_model.py`**
- Implements the `TrafficSignModel` class, a CNN for traffic sign recognition.
- Features:
  - Three convolutional layers with batch normalization and ReLU activation.
  - Fully connected layers with dropout for regularization.
  - Output layer with 43 classes (corresponding to traffic sign categories).

#### **`utils/load_data.py`**
- Provides the `load_data` function to load and preprocess image data.
- Features:
  - Supports data augmentation using transformations.
  - Handles loading images from class-specific directories.

#### **`utils/train.py`**
- Implements the `train_model` function for training the CNN.
- Features:
  - Configurable parameters for optimizer, epochs, batch size, and learning rate.
  - Tracks training loss and accuracy for each epoch.

#### **`utils/predict_frame.py`**
- Provides the `predict_frame` function for real-time traffic sign prediction.
- Features:
  - Processes video frames and predicts traffic sign classes with confidence scores.
  - Uses a trained model and a labels CSV file for predictions.

#### **`utils/video_capture.py`**
- Contains the `cap` object for webcam access.
- Configures the webcam resolution and frame rate.



#### **`utils/confusion_matrix.py`**
- Provides the `plot_confusion_matrix` function for visualizing classification results.
- Features:
  - Generates and displays (or saves) a confusion matrix for model predictions.
  - Supports normalization of values and optional saving to file.
  - Uses PyTorch and scikit-learn for evaluation and matrix generation.
  - Helps analyze model performance across traffic sign classes.


#### **`utils/evaluate_test_set.py`**
- Provides the `evaluate_test_performance` function for analyzing per-class model performance.
- Features:
  - Calculates success and failure rates for each traffic sign class.
  - Supports merging class IDs with human-readable names using a labels CSV.
  - Optionally saves the per-class evaluation results to a CSV file.
  - Highlights classes with the highest failure rates for targeted improvement.


#### **`utils/frame_utils.py`**
- Contains the `FrameProcessor` class to manage video frame processing and timing.
- Features:
  - Skips frames to reduce computation load (e.g., processes every 3rd frame).
  - Provides `should_process` method to control frame processing frequency.
  - Includes `measure_inference_time` to track and print the time taken for predictions.
  - Useful for optimizing performance in real-time video applications.


#### **`utils/speak.py`**
- Provides the `speak_if_stable` function for stable, non-repetitive speech output.
- Features:
  - Uses system text-to-speech to announce class names after consistent predictions.
  - Avoids repeating the same announcement unless the prediction changes.
  - Tracks the last few predictions using a deque to ensure stability.
  - Handles errors gracefully if text-to-speech tools are unavailable.


#### **`utils/visualizations.py`**
- Contains visualization functions to aid in understanding preprocessing and data balancing.
- Features:
  - `plot_preprocessing_pipeline` function displays an original image alongside multiple augmented versions using the applied transform.
  - `plot_oversampling_distribution` function shows side-by-side bar charts of class distribution before and after oversampling.
  - Helps verify preprocessing effectiveness and the impact of class balancing visually.



#### **`pipeline.py`**
- Implements the `TrafficSignPipeline` class to streamline the entire process of data loading, training, and saving the model.
- Features:
  - Modularized methods for data preprocessing, training, and performance visualization.
  - Saves the trained model to a specified path.

#### **`structure/labels.csv`**
- Contains the mapping of class IDs to traffic sign names.

#### **`main.py`**
- Implements the real-time traffic sign recognition system using a webcam.
- Features:
  - Loads a trained model and labels.
  - Uses the `predict_frame` function for predictions.

#### **`model/__init__.py`** and **`utils/__init__.py`**
- Added to mark the `model` and `utils` directories as Python packages.

---

### 2. **Updated Files**
#### **`requirements.txt`**
- Updated to include the necessary dependencies for the project:
  - PyTorch, torchvision, and torchsummary for building, training, and summarizing deep learning models.
  - OpenCV for video frame capture, processing, and real-time inference integration.
  - Matplotlib and Plotly for visualizing training metrics, evaluation results, and interactive graphs.
  - Imbalanced-learn and scikit-learn for data resampling, preprocessing, and evaluation metrics.
  - NumPy and Pandas for efficient numerical operations and structured data handling.
  - Pillow for image loading, format conversion, and preprocessing.
- Install them by - pip install -r requirements.txt


### 4. **Directory Structure**
The project now follows a modular structure for better maintainability:



## Running the Project
To Test the project you first need to train a model by running `pipeline.py` which will create the model to use and store it in the 
model folder as `model/utsr_NEWEST.pt`. Then to test it live you will just need to run `main.py` and it will use your main camera to 
identify the sign you show.
