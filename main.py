# Standard library imports
import argparse
from datetime import datetime
import json
import os
import subprocess

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error


def softmax(logits, axis=1):
    exps = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)

def integrate_predictions(predictions, is_ambiguous_pred, labels, is_ambiguous, num_classes):
    """
    Integrate the predictions from base models and compute evaluation metrics.

    Parameters:
    - predictions: numpy array of shape (num_samples, num_base_learners), predicted class labels from the base models
    - is_ambiguous_pred: numpy array of shape (num_samples, num_base_learners), indicator whether the sample is difficult (True means difficult) predicted by the base models
    - labels: numpy array of shape (num_samples,), the true labels
    - is_ambiguous: numpy array of shape (num_samples,), indicator whether the sample is difficult
    - num_classes: int, the total number of classes

    Returns:
    - metrics: dictionary containing all computed metrics
    """
    num_samples, num_base_learners = predictions.shape

    # Initialize metric dictionary
    metrics = {}

    is_ambiguous = is_ambiguous.astype(bool)

    # Convert is_ambiguous to integer type (True -> 1, False -> 0) to accommodate scipy.stats.mode
    is_ambiguous_int = is_ambiguous_pred.astype(int)

    # Step 1: Predict whether each sample is difficult by the mode of is_ambiguous
    # Use scipy's mode function to compute the mode for each row
    is_ambiguous_mode, _ = mode(is_ambiguous_int, axis=1)
    is_ambiguous_mode = is_ambiguous_mode.flatten().astype(bool)  # 转换为布尔类型数组，形状为 (num_samples,)

    # We do not translate the actual error message string here because it's part of the code logic, not a comment:
    if is_ambiguous_mode.shape[0] != is_ambiguous.shape[0]:
        raise ValueError(f"is_ambiguous_mode 的样本数量 {is_ambiguous_mode.shape[0]} 与 is_ambiguous 的样本数量 {is_ambiguous.shape[0]} 不一致。")

    # Compute the F1 score for ambiguous recognition
    metrics['f1_ambiguous'] = f1_score(is_ambiguous, is_ambiguous_mode, zero_division=True)

    # Initialize the integrated prediction results
    label_preds = np.zeros(num_samples)

    # Process each sample
    for i in range(num_samples):
        if is_ambiguous_mode[i]:
            # For difficult samples: take the average of Predictions as the output label (expected value)
            label_preds[i] = np.mean(predictions[i, :])
        else:
            # For easy samples:
            easy_preds = predictions[i, :]
            # Compute the mode
            mode_result = mode(easy_preds)
            mode_pred = mode_result.mode

            if isinstance(mode_pred, np.ndarray):
                label_preds[i] = mode_pred[0]
            else:
                label_preds[i] = mode_pred

    # Split indices for difficult and easy samples
    ambiguous_indices = is_ambiguous.astype(bool)
    easy_indices = ~ambiguous_indices

    # 1. Calculate RMSE for difficult samples
    if np.sum(ambiguous_indices) > 0:
        ambiguous_rmse = np.sqrt(mean_squared_error(labels[ambiguous_indices], label_preds[ambiguous_indices]))
        metrics['rmse_ambiguous'] = ambiguous_rmse
    else:
        metrics['rmse_ambiguous'] = 0  # If there are no difficult samples

    # 2. For easy samples, compute F1 score and confusion matrix
    if np.sum(easy_indices) > 0:
        # Assume it is a multi-class setting, use 'weighted' average
        easy_f1 = f1_score(labels[easy_indices], label_preds[easy_indices].astype(int), average='weighted', zero_division=0)
        easy_confusion_matrix = confusion_matrix(labels[easy_indices], label_preds[easy_indices].astype(int), normalize="true")
        metrics['f1_unambiguous'] = easy_f1
        metrics['easy_confusion_matrix'] = easy_confusion_matrix
    else:
        metrics['f1_unambiguous'] = 0
        metrics['easy_confusion_matrix'] = None  # If there are no easy samples

    # 3. Normalize RMSE (assume the maximum RMSE is num_classes - 1)
    if num_classes > 1:
        normalized_rmse_ambiguous = 1.0 - (metrics['rmse_ambiguous'] / (num_classes - 1))
    else:
        normalized_rmse_ambiguous = 1.0  # If there is only one class
    normalized_rmse_ambiguous = np.clip(normalized_rmse_ambiguous, 0.0, 1.0)  # Ensure it stays between 0 and 1
    metrics['normalized_rmse_ambiguous'] = normalized_rmse_ambiguous

    # 4. Calculate the RMSE of samples predicted as Ambiguous but actually Unambiguous (using the Ambiguous method: expected value)
    mask_pred_ambiguous_but_unambiguous = is_ambiguous_mode & (~is_ambiguous)
    if np.sum(mask_pred_ambiguous_but_unambiguous) > 0:
        rmse_pred_ambiguous_but_unambiguous = np.sqrt(mean_squared_error(labels[mask_pred_ambiguous_but_unambiguous], label_preds[mask_pred_ambiguous_but_unambiguous]))
    else:
        rmse_pred_ambiguous_but_unambiguous = 0.0
    metrics['rmse_pred_ambiguous_but_unambiguous'] = rmse_pred_ambiguous_but_unambiguous

    # 5. Calculate the RMSE of samples predicted as Unambiguous but actually Ambiguous (using the Unambiguous method: argmax)
    mask_pred_unambiguous_but_ambiguous = (~is_ambiguous_mode) & is_ambiguous
    if np.sum(mask_pred_unambiguous_but_ambiguous) > 0:
        rmse_pred_unambiguous_but_ambiguous = np.sqrt(mean_squared_error(labels[mask_pred_unambiguous_but_ambiguous], label_preds[mask_pred_unambiguous_but_ambiguous]))
    else:
        rmse_pred_unambiguous_but_ambiguous = 0.0
    metrics['rmse_pred_unambiguous_but_ambiguous'] = rmse_pred_unambiguous_but_ambiguous

    # Add the prediction results and whether it is difficult to the metrics
    metrics["preds"] = label_preds
    metrics["is_ambiguous"] = is_ambiguous_mode

    return metrics


def main():
    # Env
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # If you are using multi GPU
    env['OMP_NUM_THREADS'] = '1'

    # Load configuration from JSON file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config = json.load(f)

    # Extract parameters from the config
    num_classes = base_config['num_classes']
    num_base_learners = base_config['num_base_learners']
    model_save_path = base_config.get('model_save_path', './model')
    data_path = base_config['data_path']
    test_data_path = base_config['test_data_path']
    train_file_path = base_config['train_file_path']

    test_data = pd.read_excel(test_data_path)

    # Train a model for each base learner
    for sample_i in range(num_base_learners):
        # Modify the configuration file, including hold_epoch
        model_dir = f"{model_save_path}/Model{sample_i+1}/"
        base_config["model_save_path"] = model_dir
        base_config["train_data_path"] = f"{data_path}/train_sample_{sample_i+1}.xlsx"
        os.makedirs(base_config["model_save_path"], exist_ok=True)
        # Save the configuration file, including hold_epoch
        config_file_path = os.path.join(model_dir, f"model{sample_i+1}_configs.json")
        with open(config_file_path, 'w') as f_out:
            json.dump(base_config, f_out, indent=4)
        # Train the model
        command = [
            "python", train_file_path, "--config", config_file_path
        ]
        print(f"Traning Base Model {sample_i+1}, Start at：{datetime.now()}")

        subprocess.run(command, env=env)

    # Collect prediction results
    test_predictions = []
    test_is_ambiguous = []

    for sample_i in range(num_base_learners):
        preds_dir = model_dir
        test_prediction_path = os.path.join(preds_dir, 'test_preds.json')
        with open(test_prediction_path, 'r') as f:
            test_prediction = json.load(f)
        test_predictions.append(test_prediction["preds"])
        test_is_ambiguous.append(test_prediction["is_ambiguous"])

    # Convert to numpy arrays and transpose
    test_predictions = np.array(test_predictions).T
    test_is_ambiguous = np.array(test_is_ambiguous).T

    # Call the integration function
    test_metrics = integrate_predictions(
        test_predictions,
        test_is_ambiguous,
        test_data["Defect Level"].values,
        test_data["Ambiguous"].values,
        num_classes
    )

    # Extract and map the required metrics
    output_metrics = {
        'f1_ambiguous': test_metrics.get('f1_ambiguous', 0.0),
        'f1_unambiguous': test_metrics.get('f1_unambiguous', 0.0),
        'rmse_ambiguous': test_metrics.get('rmse_ambiguous', 0.0),
        'rmse_pred_ambiguous_but_unambiguous': test_metrics.get('rmse_pred_ambiguous_but_unambiguous', 0.0),
        'rmse_pred_unambiguous_but_ambiguous': test_metrics.get('rmse_pred_unambiguous_but_ambiguous', 0.0)
    }

    print("Ensembled Test Metrics:")
    print(output_metrics)

    metrics_path = os.path.join(model_save_path, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output_metrics, f, ensure_ascii=False, indent=4)

    # Save the prediction results and is_ambiguous
    test_data_copy = test_data.copy()
    test_data_copy["is_ambiguous"] = test_metrics.get('is_ambiguous', [])
    test_data_copy["Prediction"] = test_metrics.get('preds', [])
    test_predictions_path = os.path.join(model_save_path, 'test_predictions.xlsx')
    test_data_copy.to_excel(test_predictions_path, index=False)

if __name__ == '__main__':
    main()
