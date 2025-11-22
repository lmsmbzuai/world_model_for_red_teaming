"""
* File: ./training/preparation.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Functions to prepare the data for the training.
* Functions:
    - split_dataset()
    - tokenizer_sanity_check()
    - tokenize_example()
"""

# Import External Libraries
import json
from typing import cast

from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

# Import Local Modules
from training.setup import SetUpModel


def split_dataset(tokenized_dataset: Dataset) -> tuple[Dataset, Dataset]:
    """
    Split the dataset: Train and Eval.

    Args:
        tokenized_dataset (Dataset): Full dataset in the Dataset Hugging Face format.

    Returns:
        Arguments (tuple[Dataset, Dataset]): Train and Eval datasets in the Dataset Hugging Face format.
    """
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return (train_dataset, eval_dataset)


def tokenizer_sanity_check(setup: SetUpModel, tokenized_dataset: Dataset) -> None:
    """
    Check if the tokenization part respect the expectations.

    Args:
        setup (SetUpModel): The specific setup with the tokenizer and the model.
        tokenized_dataset (Dataset): Full dataset in the Dataset Hugging Face format.

    Returns:
        None.
    """
    sample = tokenized_dataset[0]
    output_ids = [
        id for id, lab in zip(sample["input_ids"], sample["labels"]) if lab != -100
    ]
    print(setup.tokenizer.decode(output_ids, skip_special_tokens=True))
    print("input_ids length:", len(sample["input_ids"]))
    print("labels length:", len(sample["labels"]))
    print("attention_mask length:", len(sample["attention_mask"]))
    print(
        "PAD token:", setup.tokenizer.pad_token, "| ID:", setup.tokenizer.pad_token_id
    )
    print(
        "EOS token:",
        setup.tokenizer.eos_token,
        "| ID:",
        setup.tokenizer.eos_token_id,
        "\n\n",
    )


def tokenize_example(
    example: dict[str, str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list[int]]:
    """
    Tokenize 1 example and return it in a specific format.

    Args:
        example (dict[str, str]): 1 Example in a dictionary format.
        tokenizer (PreTrainedTokenizerBase): The specific Tokenizer for our model.

    Returns:
        Tokenized Sample (dict[str, list[int]):
            BatchEncoding({
                'input_ids': [...],
                'attention_mask': [...],
                'labels': [...]
            })
    """
    # Step 1: Get the Input text
    input_text = example["input"] + "\n"

    # Step 2: Create the Full text (input + target)
    full_text = input_text + example["output"]

    # Step 3: Tokenize both
    input_tokens: BatchEncoding = tokenizer(
        input_text,
        add_special_tokens=True,
        truncation=True,
        max_length=2048,
    )
    full_tokens: BatchEncoding = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=2048,
    )

    input_tokens_dict = cast(dict[str, list[int]], input_tokens)
    full_tokens_dict = cast(dict[str, list[int]], full_tokens)

    # Step 4: Create labels: --Specific to only predict o_t+1
    # -100 for input tokens, actual token ids for output
    labels = [-100] * len(input_tokens_dict["input_ids"]) + full_tokens_dict[
        "input_ids"
    ][len(input_tokens_dict["input_ids"]) :]

    return {
        "input_ids": full_tokens_dict["input_ids"],
        "attention_mask": full_tokens_dict["attention_mask"],
        "labels": labels,
    }


def sample_size(data_list: list[dict[str, str]]) -> None:
    """
    Control the samples size, specially for the tokenization part.

    Args:
        data_list (list[dict[str, str]): List of samples in a dictionary format --the dataset

    Returns:
        None.
    """
    total_length = sum(len(s["input"]) + len(s["output"]) for s in data_list)
    max_length = max([len(s["input"]) + len(s["output"]) for s in data_list])
    average_length = total_length / len(data_list) if data_list else 0
    total_length_output = sum(len(s["output"]) for s in data_list)
    max_length_output = max([len(s["output"]) for s in data_list])
    average_length_output = total_length_output / len(data_list) if data_list else 0

    print(
        f"Average sample length (input and output): {average_length:.2f} characters.\n"
        f"Max sample length (input and output): {max_length:.2f} characters.\n"
        f"Average sample length (output): {average_length_output:.2f} characters.\n"
        f"Max sample length (output): {max_length_output:.2f} characters."
    )


def format_sample(sample: dict[str, str]) -> dict[str, str]:
    """
    Formats a sample into input/output format for world model training.

    Args:
        sample (dict[str, str]): Dictionary containing 'I', 'h_t', 'o_t', 'a_t', 'o_t+1'.

    Returns:
        sample_dict (dict[str, str]): Dictionary with 'input' and 'output' keys as strings.
    """
    sample_dict = {}
    sample_dict["input"] = (
        f"Instruction: {sample['I']}\n"
        f"History: {sample['h_t']}\n"
        f"Observation: {sample['o_t']}\n"
        f"Action: {sample['a_t']}"
    )
    sample_dict["output"] = f"Next Observation: {sample['o_t+1']}"
    return sample_dict


def preprocessing(
    original_dataset: dict[str, dict[str, dict[str, str]]],
) -> dict[int, dict[str, str]]:
    """
    Light preprocessing data:
        - Format the dataset --flattening, restructuring the data
        - Strip unnecessary metadata
        - Remove extra spaces
        - Remove where it is empty for a_t or o_t+1
        - Reset indices
        - Save preprocessed data

    Args:
        sample (dict[str, str]): Dictionary containing 'I', 'h_t', 'o_t', 'a_t', 'o_t+1'.

    Returns:
        sample_dict (dict[str, str]): Dictionary with 'input' and 'output' keys as strings.
    """
    # Step 1: Format the dataset --flattening, restructuring the data
    dataset: dict[int, dict[str, str]] = {}
    idx = 0
    for outer in original_dataset.values():
        if not outer:  # skip empty dicts like "353": {}
            continue
        for inner in outer.values():
            dataset[idx] = inner
            idx += 1

    # Step 2: Strip unnecessary metadata --messages from system error
    target_sentence = "[ERROR]: Invalid format. Resend your message using: [AGENT_NAME]: <message>\n    Valid agents: [RECOMMENDER_AGENT], [WEATHER_AGENT], [MESSAGING_AGENT], [TICKETING_AGENT]\n    Do not reply to this error."

    for key in dataset:
        for inner_key in dataset[key]:
            dataset[key][inner_key] = dataset[key][inner_key].replace(
                target_sentence, " "
            )

    # Step 3: Remove extra spaces
    for key in dataset:
        for inner_key in dataset[key]:
            dataset[key][inner_key] = dataset[key][inner_key].strip()

    # Step 4: Remove where it is empty for a_t or o_t+1
    filtered_data = {
        key: values
        for key, values in dataset.items()
        if values["a_t"] != "" and values["o_t+1"] != ""
    }

    # Step 5: Reset indices
    filtered_data = {i: values for i, values in enumerate(filtered_data.values())}

    # Step 6: Save preprocessed data
    with open("./training/dataset/dataset_preprocessed.json", "w") as f:
        json.dump(filtered_data, f, indent=2)

    return filtered_data
