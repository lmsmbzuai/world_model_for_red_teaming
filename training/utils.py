"""
* File: ./training/utils.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Utilities functions to fine-tuned the model.
* Functions:
    - save_results()
"""

# Import External Libraries
import json
import os

from transformers.trainer_utils import TrainOutput

# Import Local Modules


def save_results(train_result: TrainOutput, eval_result, log_history) -> None:
    """
    Save log_history and training_metrics as json file.

    Args:
        train_result (TrainOutput): The train result object from Hugging Face.
        eval_result (dict): Specific data concerning the evaluation part.
        log_history (dict): The logs of the training.

    Returns:
        None.
    """

    final_metrics = {
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get(
            "train_samples_per_second"
        ),
        "global_step": train_result.global_step,
        "metrics": train_result.metrics,
        "eval_loss": eval_result.get("eval_loss"),
        "eval_runtime": eval_result.get("eval_runtime"),
        "eval_samples_per_second": eval_result.get("eval_samples_per_second"),
    }

    # Get directory of the current script (pipeline.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build path relative to this script
    output_dir = os.path.join(script_dir, "training_metrics")
    os.makedirs(output_dir, exist_ok=True)
    # Save the files
    with open(os.path.join(output_dir, "log_history.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
