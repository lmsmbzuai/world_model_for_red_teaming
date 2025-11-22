"""
* File: ./training/arguments.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Specific arguments for the training (using Trainer class from Hugging Face).
"""

# Import Libraries
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    output_dir: str = "./training/world_model_2"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    save_total_limit: int = 2
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    load_best_model_at_end: bool = True
