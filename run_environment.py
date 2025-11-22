"""
* File: ./run_environment.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Run the environment regarding the architecture number providing by the user.
"""

# Import Libraries
from argparse import ArgumentParser
from dataclasses import dataclass

# Import Local Modules
from environment.messaging.message import Message
from environment.setup import EnvironmentSetup
from experiment.config import ExperimentConfig
from experiment.utils import save_dict_to_json


@dataclass
class Args:
    """Dataclass to holds arguments from command-line"""

    architecture: int


def parse_args() -> Args:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        Arguments (Args): Specific arguments.
    """
    parser = ArgumentParser()
    _ = parser.add_argument(
        "--architecture",
        type=int,
        default="1",
        help="Type of architecture: 1-LLM-Only; 2-LLM + Reasoning Loop; 3-LLM + Reasoning Loop + World Model (default: 1)",
    )

    args = parser.parse_args()

    # Convert the argparse.Namespace to a typed dataclass
    return Args(**vars(args))


# Main Function: Run the environment
if __name__ == "__main__":
    """
    Run the environment regarding the architecture number providing by the user:
        - Parse command-line arguments
        - Create an instance of the DataClass config
        - Setup the environment
        - Setup some variables
        - Send User message --initial message
        - Start conversation loop:
            - Validate Message
            - Send the message to the right receiver and generate a response
            - From this response, identify the recipient and create a Message
        - Save Red-Team History for architecture 2 and 3

    Args:
        Arguments (Args): Specific architecture (1, 2 or 3).

    Returns:
        Arguments (Args): Specific arguments.
    """
    # Step 1: Parse command-line arguments
    args = parse_args()
    print("====Step 1 - Parse command-line arguments: Completed====\n")

    # Step 2: Create an instance of the DataClass config
    config: ExperimentConfig = ExperimentConfig()
    print("====Step 2 - Configurations Setup: Completed====\n")

    # Step 3: Setup the environment
    env: EnvironmentSetup = EnvironmentSetup(config=config)
    print("====Step 3 - Environment Setup: Completed====\n")

    # Step 4: Setup some variables
    config.architecture_id = args.architecture
    config.architecture_name = config.architectures[config.architecture_id - 1]
    config.attack_objective = config.attack_objectives[0]
    message_count = 0
    print(
        "Configurations:\n",
        f"Architecture Name: {config.architecture_name}\n",
        f"Attack Objective: {config.attack_objective}\n",
        "====Step 4 - Variables setup: Completed====\n",
    )

    # Step 5: Send User message --initial message
    if config.user_request:
        request: Message = Message(
            sender="USER", receiver="PLANNER_AGENT", content=config.user_request
        )
    else:
        raise ValueError("Error: Missing or invalid 'user_request' in configuration.")

    print("====Step 5 - User Request: Completed====\n")

    # Step 6: Start conversation loop
    while message_count <= config.max_messages:
        if request.sender != "SYSTEM":
            message_count += 1
            print(
                f"====Message {message_count} / {config.max_messages}====\n",
                f"[SENDER]: {request.sender}\n",
                f"[RECIPIENT]: {request.receiver}\n",
                f"[MESSAGE]: {request.content}\n",
            )

        # Step 6.1: Validate Message
        is_validated: bool = env.message_bus.validate_message(request)

        # If is_validated == False, the experiment is completed
        if not is_validated:
            break

        # Step 6.2: Send the message to the right receiver and generate a response
        response: str = env.message_bus.generate_response(request)

        # Step 6.3: From this response, identify the recipient and create a Message
        request = env.message_bus.create_message(request, response)

    print("====Step 6 - Conversation Loop: Completed====\n")

    # Step 7: Save Red-Team History for architecture 2 and 3
    if config.architecture_id == 2 | config.architecture_id == 3:
        save_dict_to_json(
            env.recommender_agent.red_team_history,
            "./red_team_data.json",
        )
        print(
            "====Step 7 - Save Red-Team History for architecture 2 and 3: Completed====\n"
        )
