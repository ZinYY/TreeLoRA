import os
import torch
import pandas as pd
import numpy as np
import shutil
import json
from typing import Dict, Tuple, List


def get_threshold(importance_scores: List[float], target_percentage: int = 20) -> float:
    """
    Calculate the threshold for parameter importance based on the target percentage.

    Args:
        importance_scores: List of importance scores
        target_percentage: Percentage of parameters to consider important

    Returns:
        threshold: The threshold value for importance
    """
    return np.percentile(importance_scores, 100 - target_percentage)


def read_importance_scores(score_file: str) -> Tuple[Dict[str, float], List[float]]:
    """
    Read importance scores from a CSV file

    Args:
        score_file: Path to the CSV file with importance scores

    Returns:
        importance_dict: Dictionary mapping module names to importance scores
        importance_values: List of importance score values
    """
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"Importance score file not found: {score_file}")

    df = pd.read_csv(score_file)

    # Normalize scores to [0, 1]
    min_value = df['Importance_Score'].min()
    max_value = df['Importance_Score'].max()
    if max_value > min_value:  # Avoid division by zero
        df['Importance_Score'] = (df['Importance_Score'] - min_value) / (max_value - min_value)

    module_list = list(df['Module_Name'])
    score_list = list(df['Importance_Score'])

    # Create importance score dictionary
    importance_dict = {}
    for idx in range(len(module_list)):
        module_name = module_list[idx].replace(".default", "")
        importance_dict[module_name] = float(score_list[idx])

    return importance_dict, score_list


def consolidate_parameters(
        current_weights: Dict[str, torch.Tensor],
        previous_weights: Dict[str, torch.Tensor],
        current_importance: Dict[str, float],
        previous_importance: Dict[str, float],
        current_threshold: float,
        previous_threshold: float,
        weight_current: float = 0.3,
        weight_previous: float = 0.7
) -> Dict[str, torch.Tensor]:
    """
    Consolidate parameters from current and previous tasks based on importance scores

    Args:
        current_weights: Dictionary of current task parameters
        previous_weights: Dictionary of previous task parameters
        current_importance: Dictionary of current task importance scores
        previous_importance: Dictionary of previous task importance scores
        current_threshold: Threshold for current task importance
        previous_threshold: Threshold for previous task importance
        weight_current: Weight for current task parameters in weighted average
        weight_previous: Weight for previous task parameters in weighted average

    Returns:
        consolidated_weights: Dictionary of consolidated parameters
    """
    consolidated_weights = {}

    for key in current_weights:
        # Skip non-LoRA parameters
        if "lora_" not in key:
            consolidated_weights[key] = current_weights[key]
            continue

        tensor_current = current_weights[key]
        tensor_previous = previous_weights[key]

        # Ensure tensors are on the same device
        if tensor_current.device != tensor_previous.device:
            tensor_previous = tensor_previous.to(tensor_current.device)

        ipt_score_current = current_importance.get(key, 0.0)
        ipt_score_previous = previous_importance.get(key, 0.0)

        # Merge based on importance thresholds using the TaSL algorithm logic
        if ipt_score_previous > previous_threshold:
            if ipt_score_current > current_threshold:
                # Both important: weighted average
                consolidated_weights[key] = weight_current * tensor_current + weight_previous * tensor_previous
            else:
                # Only previous important: keep previous
                consolidated_weights[key] = tensor_previous
        else:
            if ipt_score_current > current_threshold:
                # Only current important: keep current
                consolidated_weights[key] = tensor_current
            else:
                # Neither important: simple average
                consolidated_weights[key] = 0.5 * tensor_current + 0.5 * tensor_previous

    return consolidated_weights


def save_consolidated_weights(
        consolidated_weights: Dict[str, torch.Tensor],
        config_source_path: str,
        output_path: str
) -> None:
    """
    Save consolidated weights and copy configuration file

    Args:
        consolidated_weights: Dictionary of consolidated parameters
        config_source_path: Path to the source adapter configuration file
        output_path: Path to save the consolidated model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save weights
    torch.save(consolidated_weights, os.path.join(output_path, "adapter_model.bin"))

    # Copy adapter configuration
    if os.path.exists(config_source_path):
        shutil.copy(config_source_path, os.path.join(output_path, "adapter_config.json"))

        # Update the adapter config
        adapter_config_path = os.path.join(output_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)

        # Ensure r_sum is set to 0 for compatibility
        adapter_config['r_sum'] = 0

        with open(adapter_config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
    else:
        print(f"Warning: Config file not found at {config_source_path}")


def perform_tasl_consolidation(
        task_id: int,
        output_dir: str,
        target_percentage: int = 20,
        weight_current: float = 0.3,
        weight_previous: float = 0.7
) -> None:
    """
    Perform TaSL consolidation for a specific task

    Args:
        task_id: ID of the current task
        output_dir: Directory where models and importance scores are saved
        target_percentage: Percentage of parameters to consider important
        weight_current: Weight for current task parameters
        weight_previous: Weight for previous task parameters
    """
    if task_id == 0:
        # For the first task, just copy the weights
        current_path = os.path.join(output_dir, str(task_id))
        save_path = os.path.join(output_dir, f"{task_id}_consolidated")

        os.makedirs(save_path, exist_ok=True)

        # Copy model files
        for filename in ["adapter_model.bin", "adapter_config.json"]:
            source = os.path.join(current_path, filename)
            if os.path.exists(source):
                shutil.copy(source, os.path.join(save_path, filename))

        print(f"First task (ID={task_id}): Copied weights to {save_path}")
        return

    # For subsequent tasks, perform consolidation
    current_path = os.path.join(output_dir, str(task_id))
    previous_path = os.path.join(output_dir, f"{task_id - 1}_consolidated")
    save_path = os.path.join(output_dir, f"{task_id}_consolidated")

    # Check if paths exist
    for path, name in [(current_path, "current"), (previous_path, "previous")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name.capitalize()} model path not found: {path}")

    # Load importance scores
    current_score_file = os.path.join(output_dir, "ipt_file", f"Importance_Score_{task_id}.csv")
    previous_score_file = os.path.join(output_dir, "ipt_file", f"Importance_Score_{task_id - 1}.csv")

    current_importance, current_scores = read_importance_scores(current_score_file)
    previous_importance, previous_scores = read_importance_scores(previous_score_file)

    # Calculate thresholds
    current_threshold = get_threshold(current_scores, target_percentage)
    previous_threshold = get_threshold(previous_scores, target_percentage)

    print(f"Task {task_id} threshold: {current_threshold}")
    print(f"Task {task_id - 1} threshold: {previous_threshold}")

    # Load model weights
    current_weights = torch.load(os.path.join(current_path, "adapter_model.bin"))
    previous_weights = torch.load(os.path.join(previous_path, "adapter_model.bin"))

    # Consolidate weights
    consolidated_weights = consolidate_parameters(
        current_weights,
        previous_weights,
        current_importance,
        previous_importance,
        current_threshold,
        previous_threshold,
        weight_current,
        weight_previous
    )

    # Save consolidated model
    save_consolidated_weights(
        consolidated_weights,
        os.path.join(current_path, "adapter_config.json"),
        save_path
    )

    print(f"Successfully consolidated weights for task {task_id} at {save_path}")