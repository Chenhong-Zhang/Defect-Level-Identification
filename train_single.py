# Standard library imports
import os
import json
import argparse

# Third-party library imports
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from scipy.special import softmax
from transformers import BertTokenizer, TrainingArguments, EarlyStoppingCallback, AutoConfig

# Custom module imports
from Model import BridgeDefectClassifier
from CustomDataset import TrainDataset, TestDataset
from Tools import AMTracker, AMTrainer



def softmax(logits, axis=1):
    exps = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)


def compute_metrics(eval_pred):
    """
    Custom compute_metrics method that calculates five specified metrics and adds a Combined Metric.
    
    Parameters:
    - eval_pred: a tuple containing logits and labels
    - trainer: a trainer object that has a threshold attribute
    """
    
    logits, labels = eval_pred
    num_classes = logits.shape[1]

    # Extract label information
    labels_array = labels[:, 0].astype(int)      # True class labels
    is_ambiguous = labels[:, 1].astype(bool)     # True Ambiguous or not

    # Step 1: Calculate margins (for judging whether Ambiguous)
    margins = np.zeros(labels.shape[0])
    
    # Process the first class
    mask_first = (labels_array == 0)
    margins[mask_first] = logits[mask_first, 0] - logits[mask_first, 1]

    # Process the last class
    mask_last = (labels_array == num_classes - 1)
    margins[mask_last] = logits[mask_last, -1] - logits[mask_last, -2]

    # Process the middle classes
    mask_middle = ~(mask_first | mask_last)
    if np.any(mask_middle):
        middle_labels = labels_array[mask_middle]
        correct_class = middle_labels
        # Calculate the minimum of the two
        margin_left = logits[mask_middle, correct_class] - logits[mask_middle, correct_class - 1]
        margin_right = logits[mask_middle, correct_class] - logits[mask_middle, correct_class + 1]
        margins[mask_middle] = np.minimum(margin_left, margin_right)        

    # Step 2: Classify as Ambiguous or Unambiguous based on the threshold
    threshold = trainer.threshold if hasattr(trainer, 'threshold') and trainer.threshold is not None else 0.0
    pred_ambiguous = margins < threshold
    pred_unambiguous = ~pred_ambiguous

    # Step 3: Compute F1 score for the samples predicted as Ambiguous
    f1_ambiguous = f1_score(is_ambiguous, pred_ambiguous, zero_division=0)

    # Step 4: Calculate RMSE for samples that are actually Ambiguous (using the Ambiguous method: expected value)
    mask_actual_ambiguous = is_ambiguous
    logits_actual_ambiguous = logits[mask_actual_ambiguous]
    labels_actual_ambiguous = labels_array[mask_actual_ambiguous]

    if len(labels_actual_ambiguous) > 0:
        probs_actual_ambiguous = softmax(logits_actual_ambiguous, axis=1)
        class_indices = np.arange(num_classes)
        expected_values_actual_ambiguous = np.dot(probs_actual_ambiguous, class_indices)
        rmse_actual_ambiguous = np.sqrt(mean_squared_error(expected_values_actual_ambiguous, labels_actual_ambiguous))
    else:
        rmse_actual_ambiguous = 0.0

    # Step 5: Calculate F1 score for samples that are actually Unambiguous (using the Unambiguous method: argmax)
    mask_actual_unambiguous = ~is_ambiguous
    logits_actual_unambiguous = logits[mask_actual_unambiguous]
    labels_actual_unambiguous = labels_array[mask_actual_unambiguous]

    if len(labels_actual_unambiguous) > 0:
        preds_actual_unambiguous = np.argmax(logits_actual_unambiguous, axis=1)        
        f1_actual_unambiguous = f1_score(labels_actual_unambiguous, preds_actual_unambiguous, average='macro', zero_division=0)
    else:
        f1_actual_unambiguous = 0.0

    # Step 6: Calculate the RMSE of samples predicted as Ambiguous but actually Unambiguous (using the Ambiguous method: expected value)
    mask_pred_ambiguous_but_unambiguous = pred_ambiguous & (~is_ambiguous)
    logits_pred_ambiguous_but_unambiguous = logits[mask_pred_ambiguous_but_unambiguous]
    labels_pred_ambiguous_but_unambiguous = labels_array[mask_pred_ambiguous_but_unambiguous]

    if len(labels_pred_ambiguous_but_unambiguous) > 0:
        probs_pred_ambiguous_but_unambiguous = softmax(logits_pred_ambiguous_but_unambiguous, axis=1)
        expected_values_pred_ambiguous_but_unambiguous = np.dot(probs_pred_ambiguous_but_unambiguous, class_indices)
        rmse_pred_ambiguous_but_unambiguous = np.sqrt(mean_squared_error(expected_values_pred_ambiguous_but_unambiguous, labels_pred_ambiguous_but_unambiguous))
    else:
        rmse_pred_ambiguous_but_unambiguous = 0.0

    # Step 7: Calculate the RMSE of samples predicted as Unambiguous but actually Ambiguous (using the Unambiguous method: argmax)
    mask_pred_unambiguous_but_ambiguous = pred_unambiguous & is_ambiguous
    logits_pred_unambiguous_but_ambiguous = logits[mask_pred_unambiguous_but_ambiguous]
    labels_pred_unambiguous_but_ambiguous = labels_array[mask_pred_unambiguous_but_ambiguous]

    if len(labels_pred_unambiguous_but_ambiguous) > 0:
        preds_pred_unambiguous_but_ambiguous = np.argmax(logits_pred_unambiguous_but_ambiguous, axis=1)
        rmse_pred_unambiguous_but_ambiguous = np.sqrt(mean_squared_error(preds_pred_unambiguous_but_ambiguous, labels_pred_unambiguous_but_ambiguous))
    else:
        rmse_pred_unambiguous_but_ambiguous = 0.0

    combined_metric = (
        f1_ambiguous + 
        f1_actual_unambiguous
    ) / 2

    return {
        'f1_ambiguous': f1_ambiguous,
        'rmse_actual_ambiguous': rmse_actual_ambiguous,
        'f1_actual_unambiguous': f1_actual_unambiguous,
        'rmse_pred_ambiguous_but_unambiguous': rmse_pred_ambiguous_but_unambiguous,
        'rmse_pred_unambiguous_but_ambiguous': rmse_pred_unambiguous_but_ambiguous,
        'combined': combined_metric,
    }


def compute_metrics_test(logits, labels, is_ambiguous, am_threshold):
    """
    Compute the metrics required during testing, including five main metrics and a combined metric.
    
    Parameters:
    - logits: Model output logits of shape (num_samples, num_classes)
    - labels: The true class labels of shape (num_samples,)
    - is_ambiguous: A boolean array of shape (num_samples,) indicating whether the sample is Ambiguous
    - am_threshold: The threshold used to separate Ambiguous and Unambiguous
    
    Returns:
    - A dictionary containing all computed metrics
    """

    is_ambiguous = is_ambiguous.astype(bool)
    
    num_classes = logits.shape[1]
    total_samples = len(labels)
    
    # Initialize prediction array
    preds = np.zeros(total_samples)
    labels_array = labels.astype(int)
    
    # Step 1: Calculate margins (for judging whether Ambiguous)
    margins = np.zeros(labels.shape[0])
    
    # Process the first class
    mask_first = (labels_array == 0)
    margins[mask_first] = logits[mask_first, 0] - logits[mask_first, 1]

    # Process the last class
    mask_last = (labels_array == num_classes - 1)
    margins[mask_last] = logits[mask_last, -1] - logits[mask_last, -2]

    # Process the middle classes
    mask_middle = ~(mask_first | mask_last)
    if np.any(mask_middle):
        middle_labels = labels_array[mask_middle]
        correct_class = middle_labels
        # Calculate the minimum of the two
        margin_left = logits[mask_middle, correct_class] - logits[mask_middle, correct_class - 1]
        margin_right = logits[mask_middle, correct_class] - logits[mask_middle, correct_class + 1]
        margins[mask_middle] = np.minimum(margin_left, margin_right)
    
    # Step 2: Classify as Ambiguous or Unambiguous based on am_threshold   
    pred_ambiguous = margins < am_threshold
    pred_unambiguous = ~pred_ambiguous
    
    # Step 3: Compute F1 score for samples predicted as Ambiguous
    f1_ambiguous = f1_score(is_ambiguous, pred_ambiguous, zero_division=0)
    
    # Step 4: Calculate RMSE for samples that are actually Ambiguous (using the Ambiguous method: expected value)
    mask_actual_ambiguous = is_ambiguous
    logits_actual_ambiguous = logits[mask_actual_ambiguous]
    labels_actual_ambiguous = labels_array[mask_actual_ambiguous]

    if len(labels_actual_ambiguous) > 0:
        probs_actual_ambiguous = softmax(logits_actual_ambiguous, axis=1)
        class_indices = np.arange(num_classes)
        expected_values_actual_ambiguous = np.dot(probs_actual_ambiguous, class_indices)
        rmse_actual_ambiguous = np.sqrt(mean_squared_error(expected_values_actual_ambiguous, labels_actual_ambiguous))
    else:
        rmse_actual_ambiguous = 0.0
    
    # Step 5: Calculate F1 score for samples that are actually Unambiguous (using the Unambiguous method: argmax)
    mask_actual_unambiguous = ~is_ambiguous
    logits_actual_unambiguous = logits[mask_actual_unambiguous]
    labels_actual_unambiguous = labels_array[mask_actual_unambiguous]

    if len(labels_actual_unambiguous) > 0:
        preds_actual_unambiguous = np.argmax(logits_actual_unambiguous, axis=1)        
        f1_actual_unambiguous = f1_score(labels_actual_unambiguous, preds_actual_unambiguous, average='macro', zero_division=0)
    else:
        f1_actual_unambiguous = 0.0
    
    # Step 6: Calculate the RMSE of samples predicted as Ambiguous but actually Unambiguous (using the Ambiguous method: expected value)
    mask_pred_ambiguous_but_unambiguous = pred_ambiguous & (~is_ambiguous)
    logits_pred_ambiguous_but_unambiguous = logits[mask_pred_ambiguous_but_unambiguous]
    labels_pred_ambiguous_but_unambiguous = labels_array[mask_pred_ambiguous_but_unambiguous]

    if len(labels_pred_ambiguous_but_unambiguous) > 0:
        probs_pred_ambiguous_but_unambiguous = softmax(logits_pred_ambiguous_but_unambiguous, axis=1)
        expected_values_pred_ambiguous_but_unambiguous = np.dot(probs_pred_ambiguous_but_unambiguous, class_indices)
        rmse_pred_ambiguous_but_unambiguous = np.sqrt(mean_squared_error(expected_values_pred_ambiguous_but_unambiguous, labels_pred_ambiguous_but_unambiguous))
    else:
        rmse_pred_ambiguous_but_unambiguous = 0.0
    
    # Step 7: Calculate the RMSE of samples predicted as Unambiguous but actually Ambiguous (using the Unambiguous method: argmax)
    mask_pred_unambiguous_but_ambiguous = pred_unambiguous & is_ambiguous
    logits_pred_unambiguous_but_ambiguous = logits[mask_pred_unambiguous_but_ambiguous]
    labels_pred_unambiguous_but_ambiguous = labels_array[mask_pred_unambiguous_but_ambiguous]

    if len(labels_pred_unambiguous_but_ambiguous) > 0:
        preds_pred_unambiguous_but_ambiguous = np.argmax(logits_pred_unambiguous_but_ambiguous, axis=1)
        rmse_pred_unambiguous_but_ambiguous = np.sqrt(mean_squared_error(preds_pred_unambiguous_but_ambiguous, labels_pred_unambiguous_but_ambiguous))
    else:
        rmse_pred_unambiguous_but_ambiguous = 0.0 
    
    # Step 8: Store prediction results
    # For samples that are actually Ambiguous, store the expected value predictions
    if len(labels_actual_ambiguous) > 0:
        preds[mask_actual_ambiguous] = expected_values_actual_ambiguous
    # For samples that are actually Unambiguous, store the argmax predictions (already computed in Step 5)
    if len(labels_actual_unambiguous) > 0:
        preds[mask_actual_unambiguous] = preds_actual_unambiguous
    
    # Return all metrics and related information
    return {        
        'margins': margins,
        'preds': preds, 
        'is_ambiguous': pred_ambiguous,
        'f1_ambiguous': f1_ambiguous,
        'rmse_actual_ambiguous': rmse_actual_ambiguous,
        'f1_actual_unambiguous': f1_actual_unambiguous,
        'rmse_pred_ambiguous_but_unambiguous': rmse_pred_ambiguous_but_unambiguous,
        'rmse_pred_unambiguous_but_ambiguous': rmse_pred_unambiguous_but_ambiguous
    }


# Main function
def main():
    # Load configuration from JSON file    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract parameters from the config
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    early_stopping_patience = config['early_stopping_patience']
    num_classes = config['num_classes']
    max_grad_norm = config.get('max_grad_norm', 1.0)
    model_save_path = config.get('model_save_path', './model')
    bert_path = config['bert_path']
    train_data_path = config['train_data_path']
    val_data_path = config['val_data_path']
    test_data_path = config['test_data_path']
    max_seq_len = config['max_seq_len']
    projected_dim = config['projected_dim']
    num_layers = config['num_layers']
    growth_rate = config['growth_rate']
    transition_layers = config['transition_layers']
    dropout_rate = config['dropout_rate']
    am_percentile = config['am_percentile']
    max_smoothing = config["max_smoothing"]
    smoothing_scale = config["smoothing_scale"]
    hold_epoch = config["hold_epoch"]

    config_model = AutoConfig.from_pretrained(bert_path)

    # Manually add custom attributes
    config_model.num_classes = num_classes
    config_model.bert_path = bert_path
    config_model.projected_dim = projected_dim
    config_model.num_layers = num_layers
    config_model.growth_rate = growth_rate
    config_model.transition_layers = transition_layers
    config_model.dropout_rate = dropout_rate

    # Ensure the model save directory exists
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # Prepare datasets
    train_data = pd.read_excel(train_data_path)
    val_data = pd.read_excel(val_data_path)
    test_data = pd.read_excel(test_data_path)    

    train_dataset = TrainDataset(train_data, tokenizer, max_seq_len, model_save_path, mode='train')
    val_dataset = TestDataset(val_data, tokenizer, max_seq_len, model_save_path, mode='val')
    test_dataset = TestDataset(test_data, tokenizer, max_seq_len, model_save_path, mode='test')
        
    # Initialize AMTracker
    train_size = len(train_dataset)
    am_tracker = AMTracker(dataset_size=train_size, num_classes=num_classes)

    # Initialize model
    model = BridgeDefectClassifier(config_model)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        logging_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='combined',
        greater_is_better=True,
        save_total_limit=1,
        fp16=True,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        weight_decay=1e-6,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1
    )

    # Early stopping callback
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
    ]

    # Initialize AMTrainer
    global trainer

    trainer = AMTrainer(
        am_tracker=am_tracker,
        num_classes=num_classes,
        percentile=am_percentile,
        max_smoothing=max_smoothing,
        smoothing_scale=smoothing_scale,
        hold_epoch=hold_epoch,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        compute_metrics=compute_metrics
    )

    # Start training
    if args.resume:
        # Find the latest checkpoint
        latest_checkpoint = None
        if os.path.isdir(model_save_path):
            checkpoints = [os.path.join(model_save_path, d) for d in os.listdir(model_save_path) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        if latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            print("No checkpoint found, starting training from scratch.")
            trainer.train()
    else:
        trainer.train() 
        
    # Save model and tokenizer
    trainer.save_model(model_save_path)  # Save model
    tokenizer.save_pretrained(model_save_path)  # Save tokenizer
    
    # Save trainer's state
    trainer_state = trainer.state
    with open(os.path.join(model_save_path, 'trainer_state.json'), 'w') as f:
        json.dump(trainer_state.__dict__, f)
    
    # Evaluate on the test set
    test_results = trainer.predict(test_dataset)
    test_logits = test_results.predictions
    
    # Best Epoch AM
    best_model_checkpoint = int(int(trainer_state.__dict__["best_model_checkpoint"].split("-")[-1])/7)
    best_checkpoint_am_path = os.path.join(model_save_path, f'AM_History//am{best_model_checkpoint}.json')
    with open(best_checkpoint_am_path, 'r') as f:
        best_checkpoint_am = json.load(f)
    best_checkpoint_am_values = np.array([best_checkpoint_am[k] for k in best_checkpoint_am])
    best_checkpoint_am_threshold = np.percentile(best_checkpoint_am_values, am_percentile)

    # Metrics
    test_metrics = compute_metrics_test(test_logits, test_data["Defect Level"].values, test_data["Ambiguous"].values, best_checkpoint_am_threshold)  
    output_metrics = {
        'f1_ambiguous': test_metrics.get('f1_ambiguous', 0.0),
        'f1_actual_unambiguous': test_metrics.get('f1_actual_unambiguous', 0.0),
        'rmse_actual_ambiguous': test_metrics.get('rmse_actual_ambiguous', 0.0),
        'rmse_pred_ambiguous_but_unambiguous': test_metrics.get('rmse_pred_ambiguous_but_unambiguous', 0.0),
        'rmse_pred_unambiguous_but_ambiguous': test_metrics.get('rmse_pred_unambiguous_but_ambiguous', 0.0),
        'threshold': best_checkpoint_am_threshold
    }
    print("Test Metrics: ")
    print(output_metrics)
     
    # Predictions    
    with open(os.path.join(model_save_path, 'test_preds.json'), 'w') as f:        
        preds_to_save = {k: test_metrics[k].tolist() if isinstance(test_metrics[k], np.ndarray) else test_metrics[k] for k in ['margins', 'preds', 'is_ambiguous']}
        json.dump(preds_to_save, f, ensure_ascii=False, indent=4) 

    # Save test evaluation results
    metrics_path = os.path.join(model_save_path, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output_metrics, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
