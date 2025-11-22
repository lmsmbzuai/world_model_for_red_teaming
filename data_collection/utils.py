"""
* File: ./data_collection/utils.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Utilities functions for the data collection part.
* Functions:
    - merge_datasets()
"""

# Import Libraries
import json


def merge_datasets(
    dataset_path_1: str,
    dataset_path_2: str,
    output_path: str,
) -> None:
    """
    Merge different datasets in one.

    Args:
        dataset_path_1 (str): Path of a dataset.
        dataset_path_2 (str): Path of a dataset.
        output_path (str): Path of a dataset.

    Returns:
        None.
    """

    # Step 1: Load both files
    with open(dataset_path_1, "r") as f1:
        data1 = json.load(f1)

    with open(dataset_path_2, "r") as f2:
        data2 = json.load(f2)

    # Step 2: Merge by renumbering keys
    merged = {}
    counter = 0

    for key in sorted(data1.keys()):
        merged[f"{counter}"] = data1[key]
        counter += 1

    for key in sorted(data2.keys()):
        merged[f"{counter}"] = data2[key]
        counter += 1

    # Step 3: Save merged data
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)


# Main Function: Run the script to merge datasets
if __name__ == "__main__":
    # Merge datasets if needed
    # Conversation History
    merge_datasets(
        dataset_path_1="./data_collection/dataset/conversation_history_qwen.json",
        dataset_path_2="./data_collection/dataset/conversation_history_llama.json",
        output_path="./data_collection/dataset/conversation_history.json",
    )
    # Dataset
    merge_datasets(
        dataset_path_1="./data_collection/dataset/dataset_qwen.json",
        dataset_path_2="./data_collection/dataset/dataset_llama.json",
        output_path="./data_collection/dataset",
    )
    print("Step 0 -> Merge datasets: Completed.\n\n")
