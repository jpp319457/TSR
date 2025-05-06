# Import necessary libraries
import numpy as np # For numerical operations
import pandas as pd # For working with tabular data

# Function to evaluate the model's performance on test data at a per-class level
def evaluate_test_performance(y_true, y_pred, labels_df=None, save_path=None):
    """
    Analyzes prediction results by calculating per-class performance statistics.

    Parameters:
    - y_true: array-like of true class labels
    - y_pred: array-like of predicted class labels
    - labels_df: optional DataFrame that maps class IDs to human-readable names (with columns 'ClassId' and 'Name')
    - save_path: optional path to save the evaluation results as a CSV file

    Returns:
    - stats_df: DataFrame containing per-class performance statistics
    """

    # Create a DataFrame to store true and predicted labels for analysis
    df = pd.DataFrame({'True': y_true, 'Pred': y_pred})

    class_stats = [] # List to hold statistics for each class
    classes = np.unique(y_true) # Get the unique class labels

    # Loop over each class and calculate statistics
    for c in classes:
        total = (df['True'] == c).sum() # Total samples for class c
        correct = ((df['True'] == c) & (df['Pred'] == c)).sum() # Correct predictions for class c
        failed = total - correct # Incorrect predictions

        # Compute success and failure rates as percentages
        success_rate = correct / total * 100
        failure_rate = failed / total * 100

        # Append statistics for the current class
        class_stats.append({
            'Class ID': c,
            'Total Samples': total,
            'Correct Predictions': correct,
            'Incorrect Predictions': failed,
            'Success Rate (%)': round(success_rate, 2),
            'Failure Rate (%)': round(failure_rate, 2)
        })

    # Create a DataFrame from the list of class-wise statistics
    stats_df = pd.DataFrame(class_stats)

    # If label names are provided, merge them with the stats using Class ID
    if labels_df is not None:
        stats_df = stats_df.merge(labels_df, left_on='Class ID', right_on='ClassId')
        # Reorder columns for clarity
        stats_df = stats_df[[
            'Class ID', 'Name', 'Total Samples', 'Correct Predictions',
            'Incorrect Predictions', 'Success Rate (%)', 'Failure Rate (%)'
        ]]

    # Sort the DataFrame by failure rate (highest to lowest) to highlight weakest classes
    stats_df = stats_df.sort_values(by='Failure Rate (%)', ascending=False)

    # Save the results to a CSV file if a path is provided
    if save_path:
        stats_df.to_csv(save_path, index=False)
        print(f"Per-class evaluation saved to {save_path}")

    # Return the statistics DataFrame
    return stats_df




