import math
import torch
import os
import pandas as pd
import numpy as np
from typing import Optional, List


class ImportanceScorer:
    """
    Importance score calculator for TaSL method.
    Calculates importance scores for LoRA parameters based on parameter sensitivity.

    Args:
        model: The model with LoRA adapters
        init_warmup (int): Number of steps for initial warmup
        beta1 (float): EMA hyperparameter for sensitivity smoothing
        beta2 (float): EMA hyperparameter for uncertainty quantification
        total_step (Optional[int]): Total training steps
    """

    def __init__(
            self,
            model,
            init_warmup=50,
            beta1=0.85,
            beta2=0.85,
            total_step=None,
    ):
        self.initial_warmup = init_warmup
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {}  # Parameter sensitivity
        self.exp_avg_ipt = {}  # Exponential moving average of sensitivity
        self.exp_avg_unc = {}  # Exponential moving average of uncertainty

        # Get parameter names for tracking
        self.get_lora_param_names()

        # Validation
        assert (self.beta1 < 1 and self.beta1 > 0)
        assert (self.beta2 < 1 and self.beta2 > 0)

    def set_total_step(self, total_step: int):
        """Set total training steps"""
        self.total_step = total_step
        self.initial_warmup = max(50, int(self.total_step / 20))
        print(f"Total steps: {self.total_step}, initial warmup: {self.initial_warmup}")
        assert self.total_step > self.initial_warmup

    def get_lora_param_names(self):
        """Identify all LoRA parameters in the model"""
        self.name_set = set()
        self.shape_dict = {}

        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                self.shape_dict[name] = param.shape
            if "lora_B" in name:
                self.shape_dict[name] = param.shape

    def update_importance(self, model, global_step):
        """Update importance scores during training"""
        # Skip during warmup period
        if global_step < self.initial_warmup:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    if param.grad is None:
                        continue

                    # Initialize tracking tensors if needed
                    if name not in self.ipt:
                        self.ipt[name] = torch.zeros_like(param)
                        self.exp_avg_ipt[name] = torch.zeros_like(param)
                        self.exp_avg_unc[name] = torch.zeros_like(param)

                    with torch.no_grad():
                        # Calculate sensitivity (parameter × gradient)
                        self.ipt[name] = (param * param.grad).abs().detach()

                        # Update exponential moving averages
                        self.exp_avg_ipt[name] = self.beta1 * self.exp_avg_ipt[name] + (1 - self.beta1) * self.ipt[name]
                        self.exp_avg_unc[name] = self.beta2 * self.exp_avg_unc[name] + (1 - self.beta2) * (
                                    self.ipt[name] - self.exp_avg_ipt[name]).abs()

    def calculate_importance_scores(self, metric="ipt"):
        """Calculate and return importance scores"""
        if not self.exp_avg_ipt:
            print("No importance scores calculated yet")
            return [], []

        ipt_name_list = []
        ipt_score_list = []

        for name in self.exp_avg_ipt:
            ipt_name_list.append(name)

            if metric == "ipt":
                # Combine sensitivity and uncertainty
                ipt_score = self.exp_avg_ipt[name] * self.exp_avg_unc[name]
                ipt_score_mean = torch.mean(ipt_score).item()
            else:
                raise ValueError(f"Unexpected metric: {metric}")

            ipt_score_list.append(ipt_score_mean)

        return ipt_name_list, ipt_score_list

    def save_importance_scores(self, task_id, output_dir, name="Importance_Score"):
        """Save importance scores to a CSV file"""
        ipt_name_list, ipt_score_list = self.calculate_importance_scores()

        if not ipt_name_list:
            print("No importance scores to save")
            return

        # Create DataFrame
        data = {'Module_Name': ipt_name_list, 'Importance_Score': ipt_score_list}
        df = pd.DataFrame(data)

        # Create directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, "ipt_file"), exist_ok=True)

        # Save CSV
        csv_path = os.path.join(output_dir, "ipt_file", f"{name}_{task_id}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved importance scores to {csv_path}")

        return csv_path