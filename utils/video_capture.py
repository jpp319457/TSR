import cv2 # OpenCV library for computer vision tasks and camera handling

# Initialize webcam capture (0 = default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution and frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Set width to 640 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Set height to 480 pixels
cap.set(cv2.CAP_PROP_FPS, 30) # Set frame rate to 30 frames per second
