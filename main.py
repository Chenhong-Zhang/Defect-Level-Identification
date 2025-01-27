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
    集成基模型的预测结果，并计算评估指标。

    参数：
    - predictions: numpy数组，形状为 (num_samples, num_base_learners)，基模型的预测分类标签
    - is_ambiguous_pred: numpy数组，形状为 (num_samples, num_base_learners)，基模型预测样本是否困难的标识（True为困难）
    - labels: numpy数组，形状为 (num_samples,)，真实的标签
    - is_ambiguous: numpy数组，形状为 (num_samples,)，是否为困难样本
    - num_classes: int，类别总数

    返回：
    - metrics: 字典，包含所有计算的指标
    """
    num_samples, num_base_learners = predictions.shape

    # 初始化评估指标字典
    metrics = {}

    is_ambiguous = is_ambiguous.astype(bool)    

    # 将 is_ambiguous 转换为整数类型（True -> 1, False -> 0）以适应 scipy.stats.mode 的要求
    is_ambiguous_int = is_ambiguous_pred.astype(int)

    # 第一步：通过 is_ambiguous 的众数，预测每个样本是否困难
    # 使用 scipy 的 mode 函数计算每行的众数
    is_ambiguous_mode, _ = mode(is_ambiguous_int, axis=1)
    is_ambiguous_mode = is_ambiguous_mode.flatten().astype(bool)  # 转换为布尔类型数组，形状为 (num_samples,)

    # 检查 is_ambiguous_mode 的长度是否与 is_ambiguous 相同
    if is_ambiguous_mode.shape[0] != is_ambiguous.shape[0]:
        raise ValueError(f"is_ambiguous_mode 的样本数量 {is_ambiguous_mode.shape[0]} 与 is_ambiguous 的样本数量 {is_ambiguous.shape[0]} 不一致。")

    # 计算模糊识别的 F1 分数    
    metrics['f1_ambiguous'] = f1_score(is_ambiguous, is_ambiguous_mode, zero_division=True)

    # 初始化集成后的预测结果
    label_preds = np.zeros(num_samples)

    # 对每个样本进行处理
    for i in range(num_samples):
        if is_ambiguous_mode[i]:
            # 困难样本：取 Predictions 的平均值作为输出标签（期望值）
            label_preds[i] = np.mean(predictions[i, :])
        else:
            # 简单样本：            
            easy_preds = predictions[i, :]           
            # 计算众数
            mode_result = mode(easy_preds)
            mode_pred = mode_result.mode

            # 根据 SciPy 版本处理 mode_pred
            if isinstance(mode_pred, np.ndarray):
                label_preds[i] = mode_pred[0]
            else:
                label_preds[i] = mode_pred

    # 分割困难和简单样本的索引
    ambiguous_indices = is_ambiguous.astype(bool)
    easy_indices = ~ambiguous_indices   

    # 1. 对于困难样本，计算 RMSE
    if np.sum(ambiguous_indices) > 0:
        ambiguous_rmse = np.sqrt(mean_squared_error(labels[ambiguous_indices], label_preds[ambiguous_indices]))
        metrics['rmse_ambiguous'] = ambiguous_rmse
    else:
        metrics['rmse_ambiguous'] = 0  # 如果没有困难样本

    # 2. 对于简单样本，计算 F1 分数和混淆矩阵
    if np.sum(easy_indices) > 0:
        # 假设标签是多分类的，使用 'weighted' 平均方式        
        easy_f1 = f1_score(labels[easy_indices], label_preds[easy_indices].astype(int), average='weighted', zero_division=0)
        easy_conf_matrix = confusion_matrix(labels[easy_indices], label_preds[easy_indices].astype(int), normalize="true")
        metrics['f1_unambiguous'] = easy_f1
        metrics['easy_confusion_matrix'] = easy_conf_matrix
    else:
        metrics['f1_unambiguous'] = 0
        metrics['easy_confusion_matrix'] = None  # 如果没有简单样本

    # 3. 标准化 RMSE（假设最大 RMSE 为 num_classes - 1）
    if num_classes > 1:
        normalized_rmse_ambiguous = 1.0 - (metrics['rmse_ambiguous'] / (num_classes - 1))
    else:
        normalized_rmse_ambiguous = 1.0  # 如果只有一个类别
    normalized_rmse_ambiguous = np.clip(normalized_rmse_ambiguous, 0.0, 1.0)  # 确保在 0 到 1 之间
    metrics['normalized_rmse_ambiguous'] = normalized_rmse_ambiguous

    # 4. 计算预测为 Ambiguous 但实际为 Unambiguous 的样本的 RMSE（使用 Ambiguous 方法：期望值）
    mask_pred_ambiguous_but_unambiguous = is_ambiguous_mode & (~is_ambiguous)
    if np.sum(mask_pred_ambiguous_but_unambiguous) > 0:
        rmse_pred_ambiguous_but_unambiguous = np.sqrt(mean_squared_error(labels[mask_pred_ambiguous_but_unambiguous], label_preds[mask_pred_ambiguous_but_unambiguous]))
    else:
        rmse_pred_ambiguous_but_unambiguous = 0.0
    metrics['rmse_pred_ambiguous_but_unambiguous'] = rmse_pred_ambiguous_but_unambiguous

    # 5. 计算预测为 Unambiguous 但实际为 Ambiguous 的样本的 RMSE（使用 Unambiguous 方法：argmax）
    mask_pred_unambiguous_but_ambiguous = (~is_ambiguous_mode) & is_ambiguous
    if np.sum(mask_pred_unambiguous_but_ambiguous) > 0:
        rmse_pred_unambiguous_but_ambiguous = np.sqrt(mean_squared_error(labels[mask_pred_unambiguous_but_ambiguous], label_preds[mask_pred_unambiguous_but_ambiguous]))
    else:
        rmse_pred_unambiguous_but_ambiguous = 0.0
    metrics['rmse_pred_unambiguous_but_ambiguous'] = rmse_pred_unambiguous_but_ambiguous
    
    # 将预测结果和是否困难的标识添加到指标中
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

    # 为每个基学习器训练模型
    for sample_i in range(num_base_learners):
        # 配置文件修改，包含 hold_epoch
        model_dir = f"{model_save_path}/Model{sample_i+1}/"
        base_config["model_save_path"] = model_dir
        base_config["train_data_path"] = f"{data_path}/train_sample_{sample_i+1}.xlsx"
        os.makedirs(base_config["model_save_path"], exist_ok=True)
        # 保存配置文件，包含 hold_epoch        
        config_file_path = os.path.join(model_dir, f"model{sample_i+1}_configs.json")       
        with open(config_file_path, 'w') as f_out:
            json.dump(base_config, f_out, indent=4)
        # 模型训练
        command = [
            "python", train_file_path, "--config", config_file_path
        ]
        print(f"Traning Base Model {sample_i+1}, Start at：{datetime.now()}")

        subprocess.run(command, env=env)

    # 收集预测结果
    test_predictions = []
    test_is_ambiguous = []

    for sample_i in range(num_base_learners):
        preds_dir = model_dir                        
        test_prediction_path = os.path.join(preds_dir, 'test_preds.json')     
        with open(test_prediction_path, 'r') as f:
            test_prediction = json.load(f)
        test_predictions.append(test_prediction["preds"])
        test_is_ambiguous.append(test_prediction["is_ambiguous"])        

    # 转换为 numpy 数组并转置
    test_predictions = np.array(test_predictions).T
    test_is_ambiguous = np.array(test_is_ambiguous).T   

    # 调用集成函数
    test_metrics = integrate_predictions(
        test_predictions, 
        test_is_ambiguous, 
        test_data["病害标度"].values, 
        test_data["Ambiguous"].values,
        num_classes  
    )

    # 提取并映射所需的指标
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
    
    # 保存预测结果和 is_ambiguous
    test_data_copy = test_data.copy()
    test_data_copy["is_ambiguous"] = test_metrics.get('is_ambiguous', [])
    test_data_copy["预测标签"] = test_metrics.get('preds', [])
    test_predictions_path = os.path.join(model_save_path, 'test_predictions.xlsx')
    test_data_copy.to_excel(test_predictions_path, index=False)

if __name__ == '__main__':
    main()