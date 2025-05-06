import torch.nn as nn # PyTorch's neural network module

# CNN Model
class TrafficSignModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 2)), # Conv layer 1
            nn.Conv2d(16, 32, (3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)), # Conv layer 2
            nn.Conv2d(32, 64, (3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)), # Conv layer 3
            nn.Flatten(), # Flatten output for fully connected layers
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), # Fully connected layer 1
            nn.Linear(256, 43) # Output layer (43 classes)
        )
    
    def forward(self, x):
        return self.model(x)  # Forward pass
    
