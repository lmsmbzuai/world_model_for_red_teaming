"""
* File: ./training/qualitative_evaluation/runner.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Run the environment to collect data in order to conduct a qualitative evaluation of the results.
"""

# Import Libraries
import json

from datasets import Dataset

# Import Local Modules
from training.preparation import (
    format_sample,
    split_dataset,
)
from training.setup import SetUpModel


def get_evalset() -> Dataset:
    """
    Get the specific dataset for the evaluation:
        - Import the data
        - Convert the dataset to list format
        - Create HF Dataset
        - Shuffle and split

    Args:
        None

    Returns:
        split_datasets (Dataset): Return the eval dataset in the Dataset format.
    """

    seed: int = 42
    data_list = []

    # Step 1: Import the data
    with open("./training/dataset/dataset_preprocessed.json", "r") as file:
        dataset = json.load(file)

    # Step 2: Convert the dataset to list format
    for key, value in dataset.items():
        data_list.append(format_sample(value))

    # Step 3: Create HF Dataset (but skip tokenization)
    hf_dataset = Dataset.from_list(data_list)
    print(f"Step 5 -> Create specific Dataset format: Completed\n{hf_dataset}\n\n")

    # Step 4: Shuffle and split
    hf_dataset = hf_dataset.shuffle(seed=seed)
    split_datasets = split_dataset(tokenized_dataset=hf_dataset)

    return split_datasets[1]


# Main Function: Run Qualitative Evaluation
if __name__ == "__main__":
    qualitative_eval: dict = {}

    # Step 1: Get the Evaluation set --same as during training
    eval_set: Dataset = get_evalset()

    # Step 2: Set the model
    setup: SetUpModel = SetUpModel(
        model_name="Qwen/Qwen2.5-1.5B",
        fine_tuned_path="./training/world_model/world_model_2",
    )

    # Step 3: Iterate through the eval set and generate an outpout from the model
    for i, sample in enumerate(eval_set):
        # Get action field in the input
        action: str = sample["input"].split("Action:", 1)[1].strip()  # pyright: ignore[reportArgumentType, reportCallIssue]
        # Generate a prediction given the input
        generated_text: str = setup.generate(sample["input"])  # pyright: ignore[reportArgumentType, reportCallIssue]
        qualitative_eval[i] = {
            "input_action": action,
            "ground_truth": sample["output"],  # pyright: ignore[reportArgumentType, reportCallIssue]
            "prediction": generated_text,
        }

    # Step 4: Save the results
    with open("training/qualitative_evaluation/qualitative_eval.json", "w") as f:
        json.dump(qualitative_eval, f, indent=4, ensure_ascii=False)
