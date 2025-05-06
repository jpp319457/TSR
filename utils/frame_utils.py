# Import the time module to measure execution duration
import time

# Define a class to manage frame processing logic
class FrameProcessor:
    def __init__(self, frame_skip=3):
        """
        Initialize the FrameProcessor.

        Parameters:
        - frame_skip: number of frames to skip before processing one.
        For example, if frame_skip=3, every 3rd frame is processed.
        """
        self.frame_skip = frame_skip # How many frames to skip
        self.frame_count = 0 # Counter to keep track of frames seen

    def should_process(self):
        """
        Determine whether the current frame should be processed.
        
        This is used to reduce computational load by skipping frames.
        Returns True only when the current frame count is divisible by frame_skip.
        """
        self.frame_count += 1 # Increment the frame counter
        return self.frame_count % self.frame_skip == 0 # Process only every Nth frame

    def measure_inference_time(self, predict_fn, frame, *args, **kwargs):
        """
        Measure the time taken by a prediction function to process a frame.
        
        Parameters:
        - predict_fn: the function that performs inference (e.g., a model's predict method)
        - frame: the input frame to be processed
        - *args, **kwargs: additional arguments to pass to predict_fn

        Returns:
        - The result of the prediction function
        - Also prints the time taken in milliseconds for visibility/performance tracking
        """
        start_time = time.time() # Record start time
        result = predict_fn(frame, *args, **kwargs) # Run the prediction
        end_time = time.time() # Record end time

        # Calculate elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Frame Processing Time: {elapsed_time_ms:.2f} ms") # Print timing info

        return result # Return the result of the prediction



