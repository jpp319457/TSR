# Import required libraries
import subprocess # For running external commands (used here for text-to-speech)
from collections import deque # For efficiently managing a fixed-length list of recent predictions

# Create a deque (double-ended queue) to keep track of the last few predictions
last_predictions = deque(maxlen=3) # Only store the last 3 predictions

# Variable to store the last class name that was spoken
last_spoken = None

# Function to trigger speech output if the prediction is stable and hasn't been spoken recently
def speak_if_stable(class_name):
    """
    Uses text-to-speech to announce a class name if it has been predicted consistently.

    Parameters:
    - class_name: The predicted class name as a string.

    Behavior:
    - If the same class name is predicted 3 times in a row (and hasn't already been spoken),
      it will be spoken aloud using the system's text-to-speech capability.
    - Prevents speaking the same class repeatedly unless it changes.
    """
    global last_spoken # Reference the global variable that tracks the last spoken word

    if class_name is None:
        return # Do nothing if there's no valid class name to speak

    # Add the current prediction to the recent history
    last_predictions.append(class_name)

    # Check if the class name has been predicted 3 times consecutively and wasn't just spoken
    if last_predictions.count(class_name) == 3 and class_name != last_spoken:
        last_spoken = class_name # Update the last spoken word

        try:
            # Use the system's text-to-speech to say the class name (non-blocking)
            subprocess.Popen(['say', class_name])
        except Exception as e:
            # Print error if text-to-speech fails (e.g., command not found on the system)
            print(f"Text-to-Speech Error: {e}")



