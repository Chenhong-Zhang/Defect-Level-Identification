import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Trainer
import os
import json


class DynamicLossFunction(nn.Module):
    def __init__(self):
        """
        动态损失函数，基于AM对样本应用不同的损失策略。
        """
        super(DynamicLossFunction, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, smoothing):
        """
        计算损失。

        Args:
            logits (Tensor): 模型的预测结果，形状为 (batch_size, num_classes)
            targets (Tensor): 真实标签，形状为 (batch_size,)
            smoothing (Tensor): 张量根据每个样本的AM计算标签平滑程度，形状为 (batch_size,)
        Returns:
            Tensor: 均值损失
        """
        ce = self.ce_loss(logits, targets)  # Shape: (batch_size,)
        if smoothing is not None and torch.any(smoothing > 0):
            num_classes = logits.size(1)
            with torch.no_grad():
                one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            smooth = smoothing.view(-1, 1)  # Shape: (batch_size, 1)
            smooth_labels = one_hot * (1 - smooth) + smooth / num_classes  # Shape: (batch_size, num_classes)
            log_probs = F.log_softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
            smoothed_ce = -torch.sum(smooth_labels * log_probs, dim=1)  # Shape: (batch_size,)            
            loss = torch.where(smoothing > 0, smoothed_ce, ce)
        else:            
            loss = ce
        
        return loss.mean()


class AMTracker:
    def __init__(self, dataset_size, num_classes):
        self.dataset_size = dataset_size
        self.num_classes = num_classes  # 总类别数，包括最后一个类别
        self.am = np.zeros(dataset_size)  # 初始化AM值
        self.count = np.zeros(dataset_size)

    def update_am(self, indices, logits, labels):
        """
        向量化更新AM值。
        :param indices: 当前batch中样本的索引，形状为 (batch_size,)
        :param logits: 当前batch中样本的logits，形状为 (batch_size, num_classes)
        :param labels: 当前batch中样本的标签，形状为 (batch_size,)
        """        
        logits = logits.detach().cpu().numpy()  # (batch_size, num_classes)
        labels = labels.detach().cpu().numpy()  # (batch_size,)
        indices = indices.detach().cpu().numpy()

        margins = np.zeros(labels.shape[0])
        mask_first = (labels == 0)
        margins[mask_first] = logits[mask_first, 0] - logits[mask_first, 1]
        mask_last = (labels == self.num_classes - 1)
        margins[mask_last] = logits[mask_last, -1] - logits[mask_last, -2]
        mask_middle = ~(mask_first | mask_last)

        if np.any(mask_middle):
            middle_labels = labels[mask_middle]
            correct_class = middle_labels
            margin_left = logits[mask_middle, correct_class] - logits[mask_middle, correct_class - 1]
            margin_right = logits[mask_middle, correct_class] - logits[mask_middle, correct_class + 1]
            margins[mask_middle] = np.minimum(margin_left, margin_right)

        np.add.at(self.am, indices, margins)
        np.add.at(self.count, indices, 1)

    def compute_final_am(self):
        """
        计算最终的AM值，通常是在每个Epoch结束时调用。
        """
        non_zero_mask = self.count > 0
        self.am[non_zero_mask] /= self.count[non_zero_mask]

    def get_am(self):
        return self.am

    def zero_am(self):
        """
        重置AM值和计数器，在每个Epoch结束后调用。
        """
        self.am = np.zeros(self.dataset_size)
        self.count = np.zeros(self.dataset_size)


class AMTrainer(Trainer):
    def __init__(self, am_tracker, num_classes, percentile, max_smoothing, smoothing_scale, hold_epoch, *args, **kwargs):
        """
        Initializes the AMTrainer.
        """
        super().__init__(*args, **kwargs)
        self.am_tracker = am_tracker
        self.num_classes = num_classes
        self.percentile = percentile
        self.max_smoothing = max_smoothing
        self.smoothing_scale = smoothing_scale
        self.hold_epoch = hold_epoch
        self.threshold = None
        self.current_epoch = 0
        self.steps_in_epoch = 0
        self.total_steps_per_epoch = len(self.get_train_dataloader())
        self.dynamic_loss_fn = DynamicLossFunction()

    def is_world_process_zero(self):
        return self.args.local_rank in [-1, 0]

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides compute_loss to integrate AM logic.
        """
        labels = inputs.pop("labels")  # (batch_size,)
        if labels.dim() == 1:
            labels = labels
        else:
            labels = labels[:, 0]
        indices = inputs.get("idx")    # (batch_size,)

        outputs = model(**inputs)
        logits = outputs.get("logits")  # (batch_size, num_classes)

        # Update AM
        self.am_tracker.update_am(indices, logits, labels)

        # Get current batch AM values
        am_values = self.am_tracker.get_am()[indices.detach().cpu().numpy()]
        am_tensor = torch.tensor(am_values, dtype=torch.float32, device=logits.device)  # (batch_size,)

        # Compute smoothing vector
        with torch.no_grad():
            if self.current_epoch < self.hold_epoch:
                # 在前 hold_epoch 轮，不进行标签平滑
                smoothing = torch.zeros_like(am_tensor)
            else:
                if self.threshold is not None:
                    # Get all AM <= self.threshold
                    mask_smooth = am_tensor <= self.threshold  # (batch_size,)
                    smooth_am = am_tensor[mask_smooth]

                    if smooth_am.numel() > 0:
                        min_am = smooth_am.min()
                        if min_am < self.threshold:
                            # Compute smoothing values
                            smoothing_neg = self.max_smoothing * ((smooth_am - self.threshold) / (min_am - self.threshold))**self.smoothing_scale
                            # Ensure smoothing values are between [0, max_smoothing]
                            smoothing_neg = torch.clamp(smoothing_neg, min=0.0, max=self.max_smoothing)
                        else:
                            smoothing_neg = torch.zeros_like(smooth_am)
                    else:
                        smoothing_neg = torch.tensor([], device=logits.device)

                    # Initialize smoothing vector to 0
                    smoothing = torch.zeros_like(am_tensor)

                    # Update smoothing values for samples that need smoothing
                    smoothing[mask_smooth] = smoothing_neg
                else:
                    # 如果threshold还未设置，默认smoothing为0
                    smoothing = torch.zeros_like(am_tensor)

        # Compute loss
        loss = self.dynamic_loss_fn(logits, labels, smoothing)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """
        Overrides training_step to check if the end of an epoch has been reached.
        """
        # Call the parent class's training_step
        loss = super().training_step(model, inputs)

        # Update step count
        self.steps_in_epoch += 1

        # Check if the end of an epoch has been reached
        if self.steps_in_epoch >= self.total_steps_per_epoch:
            self.steps_in_epoch = 0  # Reset step count
            self.current_epoch += 1  # Increment epoch count

            # Compute current AM
            self.am_tracker.compute_final_am()
            am = self.am_tracker.get_am()

            # Ensure AM save directory exists
            am_directory = os.path.join(self.args.output_dir, 'AM_History')
            os.makedirs(am_directory, exist_ok=True)

            # Generate file path
            am_filename = os.path.join(am_directory, f'am{self.current_epoch}.json')

            if self.is_world_process_zero():
                try:
                    # Save as a dictionary
                    am_dict = {str(idx): float(am_val) for idx, am_val in enumerate(am)}
                    with open(am_filename, 'w') as f:
                        json.dump(am_dict, f)
                    print(f"Saved AM for epoch {self.current_epoch} to {am_filename}")
                except Exception as e:
                    print(f"Error saving AM for epoch {self.current_epoch}: {e}")

            # Update AM threshold based on percentile
            threshold_am = np.percentile(am, self.percentile)
            self.threshold = threshold_am
            print(f"Epoch {self.current_epoch}, AM Threshold updated: {threshold_am}")

            # Reset AM for the next epoch
            self.am_tracker.zero_am()

        return loss
