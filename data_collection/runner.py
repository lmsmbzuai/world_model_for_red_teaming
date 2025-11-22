"""
* File: ./data_collection/runner.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Run the data collection process.
"""

# Import External Libraries
import gc
import json
import time

import torch

# Import Local Modules
from environment.messaging.message import Message
from environment.setup import EnvironmentSetup
from experiment.config import ExperimentConfig


def run_environment(
    env: EnvironmentSetup, config: ExperimentConfig
) -> tuple[dict[int, dict[str, str | None]], list[dict[str, str]]]:
    """
    Run the environment to collect data:
        - Send User message --initial message
        - Start conversation loop:
            - Validate Message
            - Send the message to the right receiver and generate a response
            - From this response, identify the recipient and create a Message
            - Save the sender and the messages
        - Save the Data in a specific format

    Args:
        env (EnvironmentSetup): The specific environment setup.
        config (ExperimentConfig): The specific configurations for the data collection process.

    Returns:
        data_collected (dict): Specific dictionary containing the data for training the world-model (o_t, a_t, o_t+1, o_t+2).
        env.message_bus.message_log (dict): Dictionary containing all the conversations.
    """

    # Step 0: Send initial message
    if config.user_request:
        request: Message = Message(
            sender="USER", receiver="PLANNER_AGENT", content=config.user_request
        )
    else:
        raise ValueError("Error: Missing or invalid 'user_request' in configuration.")

    message_count = 0
    index = 0
    max_messages = 50

    # Create variables to save the data
    list_of_senders: list[str] = []
    list_of_messages: list[str] = []
    data_collected: dict[int, dict[str, str | None]] = {}

    # Let agents communicate until completion
    while message_count < max_messages:
        index += 0
        # Step 1: Validate Message
        is_validated: bool = env.message_bus.validate_message(request)

        # If is_validated == False, the experiment is completed
        if not is_validated:
            break

        # Step 2: Send the message to the right receiver and generate a response
        response: str = env.message_bus.generate_response(request)

        # Step 3: From this response, identify the recipient and create a Message
        request = env.message_bus.create_message(request, response)

        # Step 4: Save the sender and the messages
        list_of_senders.append(request.sender)
        list_of_messages.append(request.content)

    # Step 5: Save the Data
    # Step 5.1: Get the index of the RECOMMENDER_AGENT --corresponds to a_t
    indexes = [
        i for i, sender in enumerate(list_of_senders) if sender == "RECOMMENDER_AGENT"
    ]
    # Step 5.2: Gather data to create the dataset
    for i, index_r_agent in enumerate(indexes):
        if index_r_agent + 2 < len(list_of_messages):  # safe to access index+2
            # Collect History
            if i > 0:
                prev_history = data_collected[i - 1].get("history", "") or ""
                o_t_prev = data_collected[i - 1]["o_t"] or ""
                a_t_prev = data_collected[i - 1]["a_t"] or ""
                o_t1_prev = data_collected[i - 1]["o_t+1"] or ""
                current_turn = o_t_prev + " " + a_t_prev + " " + o_t1_prev
                history: str = (prev_history + " " + current_turn).strip()
            else:
                history = ""

            # Gather informations
            data_collected[i] = {
                "I": config.attack_objective,
                "h_t": history,
                "o_t": list_of_messages[index_r_agent - 1],
                "a_t": list_of_messages[index_r_agent],
                "o_t+1": list_of_messages[index_r_agent + 1]
                + " "
                + list_of_messages[index_r_agent + 2],
            }

    return data_collected, env.message_bus.message_log


def run_data_collection() -> None:
    """
    Run the data collection process:
        - Load Data
        - Set configs
        - Loop Through the user_queries and attacker_objectives
        - Setup Experiment with specific environment and configurations
        - Run Environment
        - Save the data for the dataset and the conversation history in JSON
        - Clean memory

    Args:
        None.

    Returns:
        None.
    """

    # Start the timer
    start_time = time.perf_counter()

    # Step 1.1: Load Data --User requests
    with open("data_collection/instructions/user_requests.json", "r") as f:
        user_requests_data: list[dict[str, str]] = json.load(f)  # pyright: ignore[reportAny]
    user_requests: list[str] = [item["request"] for item in user_requests_data]

    # Step 1.2: Load Data --Attacker Objectives
    with open("data_collection/instructions/attacker_objectives.json", "r") as f:
        attacker_objectives_data: list[dict[str, str]] = json.load(f)  # pyright: ignore[reportAny]
    attacker_objectives: list[str] = [
        item["objective"] for item in attacker_objectives_data
    ]

    # Step 2: Set configs
    config: ExperimentConfig = ExperimentConfig()

    # Step 3: Loop Through the user_queries and attacker_objectives
    total_rounds = 10
    total_runs: int = len(user_requests) * len(attacker_objectives) * total_rounds
    current_run: int = 0
    data_collected = {}
    conversation_collected = {}

    for i, user_request in enumerate(user_requests):
        config.user_request = user_request
        for j, attack_goal in enumerate(attacker_objectives):
            config.attack_objective = attack_goal
            for round in range(total_rounds):
                current_run += 1
                name_of_run: str = f"{i}.{j}.{round}"

                print(f"----Run number: {current_run} / {total_runs}----")
                print("Settings:")
                print(f"User request: {user_request}")
                print(f"Attack Goal: {attack_goal}")
                print(f"Round: {round}")

                # Step 3.1: Set Specific variables

                # Step 3.2: Setup Experiment with specific environment and configurations
                environment_setup = EnvironmentSetup(config=config)

                # Step 3: Run Environment
                data_collected_round, conversation_collected_round = run_environment(
                    env=environment_setup, config=config
                )

                print(f"End of run number: {current_run} / {total_runs}")
                print(f"{len(data_collected_round)} data sample(s) recorded.")

                # Step 4: Save the data for the dataset in JSON
                data_collected[name_of_run] = data_collected_round
                with open("data_collection/dataset/test.json", "w") as f:
                    json.dump(data_collected, f, indent=4)

                # Step 5: Save the conversation history in JSON
                conversation_collected[name_of_run] = conversation_collected_round
                with open("data_collection/dataset/test.json", "w") as f:
                    json.dump(conversation_collected, f, indent=4)

                # Step 6: Clean memory
                # Clear GPU cache
                del environment_setup
                del data_collected_round
                del conversation_collected_round

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                _ = gc.collect()  # Python garbage collector

    end_time = time.perf_counter()
    duration = (end_time - start_time) / 60
    print(f"Data collection duration: {duration:.2f} minutes")


# Main Function: Run the data collection process
if __name__ == "__main__":
    run_data_collection()
