# Import required libraries and modules
import gc
import torch
import numpy as np
import pandas as pd
import torchsummary as ts
import plotly.express as px
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from model.traffic_sign_model import TrafficSignModel
from utils.load_data import load_data
from utils.train import train_model
from utils.evaluate_test_set import evaluate_test_performance
from utils.confusion_matrix import plot_confusion_matrix
from utils.visualizations import plot_preprocessing_pipeline, plot_oversampling_distribution
from torchvision import transforms

class TrafficSignPipeline:
    def __init__(self, data_dir, labels_path, model_save_path):
        self.data_dir = data_dir
        self.labels_path = labels_path
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrafficSignModel().to(self.device)
        self.transform = self._get_transform()
        self.labels_df = self._load_labels()

    def _get_transform(self):
        # Define image transformations for preprocessing and augmentation
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def _load_labels(self):
        # Load label names from CSV
        return pd.read_csv(self.labels_path)

    def load_and_preprocess_data(self):
        # Load images and labels from folder structure
        x, y = load_data(self.data_dir, self.transform)

        # Visualize preprocessing effects
        plot_preprocessing_pipeline(self.data_dir, self.transform)

        print("Before resampling:", np.unique(y, return_counts=True))
        y_before = y.copy()

        # Flatten and oversample to fix class imbalance
        x_flat = x.reshape(len(x), 3 * 32 * 32)
        x_resampled, y_resampled = RandomOverSampler().fit_resample(x_flat, y)
        x = x_resampled.reshape(-1, 3, 32, 32)
        y = y_resampled
        y_after = y.copy()
        print("After resampling:", np.unique(y, return_counts=True))

        # Show oversampling distribution
        plot_oversampling_distribution(y_before, y_after)

        # Split into training and test sets
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y)
        del x, y
        gc.collect()

        return (
            torch.from_numpy(xtrain),
            torch.from_numpy(xtest),
            torch.from_numpy(ytrain),
            torch.from_numpy(ytest),
        )

    def train_model(self, xtrain, ytrain):
        # Display model architecture summary
        ts.summary(self.model, (3, 32, 32))
        # Train and return training history
        return train_model(self.model, xtrain=xtrain, ytrain=ytrain, device=self.device)

    def plot_training_performance(self, history):
        # Plot training loss and accuracy
        px.line(history, y=["Train Loss"], title="Loss Per Epochs").show()
        px.line(history, y=["Train Acc"], title="Accuracy Per Epochs").show()

    def save_model(self):
        # Save trained model to disk
        torch.save(self.model, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def evaluate_model(self, xtest, ytest):
        # Evaluate the model and show performance
        test_ds = torch.utils.data.TensorDataset(xtest, ytest)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

        self.model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb.float())
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                all_preds.append(predicted.cpu())
                all_labels.append(yb.cpu())

        final_acc = 100 * correct / total
        print(f"\n Final Test Accuracy: {final_acc:.2f}%")

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()

        results_df = evaluate_test_performance(y_true, y_pred, labels_df=self.labels_df, save_path="evaluation_results.csv")

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.colheader_justify', 'center')

        print("\n Worst to Top performing traffic sign classes:")
        print(results_df.to_string(index=False))

        # Plot normalized confusion matrix
        plot_confusion_matrix(self.model, xtest, ytest, self.device, normalize=True)

    def run(self):
        # Run the full training and evaluation pipeline
        xtrain, xtest, ytrain, ytest = self.load_and_preprocess_data()
        history = self.train_model(xtrain, ytrain)
        self.plot_training_performance(history)
        self.save_model()
        self.evaluate_model(xtest, ytest)

# Entry point to run the pipeline
if __name__ == "__main__":
    pipeline = TrafficSignPipeline(
        data_dir="data",
        labels_path="structure/labels.csv",
        model_save_path="model/utsr_NEWEST.pt",
    )
    pipeline.run()
