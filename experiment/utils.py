"""
* File: ./experiment/utils.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Utilities functions to conduct the experiment(s).
* Functions:
    - nested_dict()
    - save_dict_to_json()
"""

# Import External Libraries
import json
import os
from collections import defaultdict


def nested_dict():
    """
    Ensure that the dictionary is in the right format and don't raise any errors.
    Args: None.
    Returns: Arguments (Args): Specific arguments.
    """
    return defaultdict(nested_dict)


def save_dict_to_json(data: dict, filename: str) -> None:
    """
    Save or update a dictionary to a JSON file.

    Args:
        data (dict): Specific data in a dictionary format.
        filename (str): Specific name of the file.

    Returns:
        None.
    """
    # Step 1: If file exists, load its content first to preserve old data
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        # Update existing data
        existing_data.update(data)
    else:
        existing_data = data

    # Step 2: Save back to file
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)
