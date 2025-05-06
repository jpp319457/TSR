import cv2
import torch
import pandas as pd
import time

# Import custom utilities
from utils.video_capture import cap
from utils.predict_frame import predict_frame
from utils.speak import speak_if_stable
from utils.frame_utils import FrameProcessor

# Load label names from CSV
labels_df = pd.read_csv('structure/labels.csv')

# Load the trained model and prepare it for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./model/utsr_NEWEST.pt', map_location=device)
model.to(device)
model.eval()

# Initialize frame processor with frame skipping logic
frame_processor = FrameProcessor(frame_skip=3)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Main real-time prediction loop
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret or frame is None:
        continue # Skip if frame capture failed

    # Skip frames to reduce processing load
    if not frame_processor.should_process():
        continue

    # Predict traffic sign class from the current frame
    class_name, class_id, prob = frame_processor.measure_inference_time(
        predict_frame, frame, model, device, labels_df
    )

    # If prediction is confident, speak and display it
    if class_name is not None and prob >= 0.98:
        speak_if_stable(class_name)
        cv2.putText(frame, f"{class_name} ({class_id}) - {prob:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Calculate and display FPS (frames per second)
    total_time = time.time() - start_time
    fps = 1 / total_time if total_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    print(f"FPS: {fps:.2f}")

    # Show the frame in a window
    cv2.imshow('Traffic Sign Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
