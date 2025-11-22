"""
* File: ./training/pipeline.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Pipeline for the training.
"""

# Import External Libraries
import json
from functools import partial

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_utils import TrainOutput

# Import Local Modules
from training.arguments import TrainingArgs
from training.preparation import (
    format_sample,
    preprocessing,
    sample_size,
    split_dataset,
    tokenize_example,
    tokenizer_sanity_check,
)
from training.setup import SetUpModel
from training.utils import save_results


def run_pipeline(
    model_name: str, dataset_path: str, merge_datasets_bool: bool = False
) -> None:
    """
    Run the entire pipeline for the training:
        - Setup the model and tokenizer --apply LoRA to the model
        - Load the dataset
        - Preprocess the dataset
        - Convert the dataset to list format
        - Create specific Dataset format for PyTorch
        - Tokenize the data --sanity Check
        - Split Dataset
        - Apply arguments
        - Create the data collator for padding
        - Train the model
        - Save training and evaluation metrics
        -

    Args:
        model_name (str): Name of the model.
        dataset_path (str): The path to access to the dataset.
        merge_datasets_bool (bool): If we merge datasets.

    Returns:
        None.
    """
    seed: int = 42
    torch.cuda.empty_cache()

    # Step 1: Setup the model and tokenizer
    setup: SetUpModel = SetUpModel(model_name=model_name)
    # Apply LoRA to the model
    setup.setup_lora()

    print(
        "Step 1 -> Initializing SetUpModel + LoRA: Completed\n"
        f"Device = {setup.device}; Model = {setup.model_name}\n\n"
    )

    # Step 2: Load the dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        original_dataset: dict[str, dict[str, dict[str, str]]] = json.load(f)
    print("Step 2 -> Load the dataset: Completed.\n\n")

    # Step 3: Preprocess the dataset
    dataset: dict[int, dict[str, str]] = preprocessing(
        original_dataset=original_dataset
    )
    print(
        "Step 3 -> Preprocess the dataset: Completed\n"
        f"Size of the dataset = {len(dataset)}\n\n"
    )

    # Step 4: Convert the dataset to list format
    # List of dictionnaries --1 dictionnary for each sample where we have "input" and "output" keys for each sammple
    data_list = []
    for key, value in dataset.items():
        data_list.append(format_sample(value))
    print(
        "Step 4 -> Convert the dataset to list format: Completed\n"
        f"Size of the list = {len(data_list)}\n"
        "Sanity check:\n\n"
    )
    sample_size(data_list)

    # Step 5: Create specific Dataset format for PyTorch
    hf_dataset = Dataset.from_list(data_list)
    print(f"Step 5 -> Create specific Dataset format: Completed\n{hf_dataset}\n\n")

    # Step 6: Tokenize the data using tokenize_data function
    setup.tokenizer.truncation_side = "left"  # because we want to keep the full o_t+1
    tokenized_dataset: Dataset = hf_dataset.map(
        partial(tokenize_example, tokenizer=setup.tokenizer),
        batched=False,
        remove_columns=hf_dataset.column_names,
    )
    print("Step 6 -> Tokenize the data: Completed\nSanity Check:\n")
    # Sanity Check --print the output part to check if the mask -100 for the loss is right
    tokenizer_sanity_check(setup, tokenized_dataset)

    # Step 7: Split Dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
    split_datasets: tuple[Dataset, Dataset] = split_dataset(
        tokenized_dataset=tokenized_dataset
    )
    train_dataset = split_datasets[0]
    eval_dataset = split_datasets[1]

    print(
        "Step 7 -> Split Dataset: Completed\n"
        f"Sanity Check: Train Set: {len(train_dataset)}, Evaluation Set: {len(eval_dataset)}\n\n"
    )

    # Step 8: Apply arguments
    config: TrainingArgs = TrainingArgs()
    training_arguments: TrainingArguments = TrainingArguments(**vars(config))
    print("Step 8 -> Apply arguments: Completed\n\n")

    # Step 9: Create the data collator for padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=setup.tokenizer,
        model=setup.model,
        padding=True,
    )
    print("Step 9 -> Apply data collator: Completed\n\n")

    # Step 10: Train the model
    # Handle Training + Evaluation
    trainer = Trainer(
        model=setup.model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=setup.tokenizer,
        data_collator=data_collator,
    )
    train_result: TrainOutput = trainer.train()

    print("Step 10 -> Training the model: Completed\n\n")

    # Step 11: Save training and evaluation metrics
    # Get training history
    log_history = trainer.state.log_history

    # Get final evaluation
    eval_result = trainer.evaluate()

    # Save final metrics
    save_results(train_result, eval_result, log_history)
    # And model
    trainer.save_model("./training/world_model_2")

    print("Step 11 -> Save the model and training/evaluation metrics: Completed\n\n")
    print("Training Pipeline: Completed\n\n")


# Main Function: Run the training
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-1.5B"
    dataset_path = "./data_collection/dataset/dataset.json"
    run_pipeline(
        model_name=model_name, dataset_path=dataset_path, merge_datasets_bool=True
    )
