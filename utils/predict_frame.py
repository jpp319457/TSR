# Import necessary libraries
import PIL.Image # For working with image data in Python
import torch # PyTorch for model inference
import torchvision as tv # For image transformations
import cv2 # OpenCV for image processing
import torch.nn.functional as F # For using softmax and other functions

# Function to process a video frame and make a prediction using a trained model
def predict_frame(frame, model, device, labels_df):
    """
    Predicts the class of a given video frame using a trained model.

    Parameters:
    - frame: A single video frame (BGR image from OpenCV).
    - model: The trained PyTorch model used for prediction.
    - device: The device ('cpu' or 'cuda') to run inference on.
    - labels_df: A DataFrame containing class IDs and corresponding human-readable class names.

    Returns:
    - class_name: The name of the predicted class (if confidence is high).
    - class_id: The numeric ID of the predicted class.
    - prob: The confidence (probability) of the prediction.
    If the confidence is too low or the frame is invalid, returns (None, None, None).
    """

    if frame is None:
        return None, None, None # Skip prediction if the frame is invalid

    # Convert the OpenCV frame from BGR to RGB and then to a PIL Image
    img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image to 32x32 pixels (assumed model input size)
    img_resized = img.resize((32, 32), PIL.Image.LANCZOS)

    # Convert the image to a tensor and move it to the appropriate device
    img_tensor = tv.transforms.ToTensor()(img_resized).unsqueeze(0).to(device).float()

    # Turn off gradient calculations for faster inference
    with torch.no_grad():
        output = model(img_tensor) # Get raw model outputs (logits)

        # Convert logits to probabilities using softmax
        probabilities = F.softmax(output, dim=1)

        # Get the class with the highest probability
        top_prob, top_class = torch.max(probabilities, 1)

        class_id = top_class.item() # Get predicted class ID as a Python int
        prob = top_prob.item() # Get confidence score as a float

        # Only return prediction if confidence is very high (e.g., > 98%)
        if prob > 0.98:
            # Look up the class name from the labels DataFrame
            class_name = labels_df.loc[labels_df['ClassId'] == class_id, 'Name'].values[0]
            return class_name, class_id, prob

    # If confidence is too low, return None for all outputs
    return None, None, None



