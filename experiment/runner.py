"""
* File: ./experiment/runner.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Run the different experiments.
"""

# Import External Libraries
import os

import torch

# Import Local Modules
from environment.messaging.message import Message
from environment.setup import EnvironmentSetup
from experiment.config import ExperimentConfig
from experiment.utils import nested_dict, save_dict_to_json


def run_experiment() -> None:
    """
    Run and orchestrate the experiment:
        - Create an instance of the DataClass config
        - Create Results directory
        - Declare variables
        - Main loop (multiple steps)

    Args:
        None.

    Returns:
        Arguments (Args): Specific arguments.
    """

    # Step 1: Create an instance of the DataClass config
    config: ExperimentConfig = ExperimentConfig()

    # Step 2: Create Results directory
    os.makedirs(config.results_folder_path, exist_ok=True)

    # Step 3: Declare variables
    nb_round: int = (
        len(config.attack_objectives) * len(config.seeds) * len(config.architectures)
    )
    config.nb_rounds = 0
    config.message_count = 0
    # Experiment Data --create a data dictionary to save the data for the experiment
    experiment_data = nested_dict()

    # Step 4: Main loop
    for attack_objective in config.attack_objectives:
        config.attack_objective = attack_objective

        # Experiment Data --create key for this attack objective
        experiment_data[config.attack_objective] = {}

        for seed in config.seeds:
            config.seed = seed

            # Experiment Data --create key for this seed
            experiment_data[config.attack_objective][seed] = {}

            for index, architecture in enumerate(config.architectures, start=1):
                # Step 4.1: Set specific configurations
                config.architecture = architecture
                config.architecture_id = index
                config.architecture_name = f"architecture_{index}"
                config.nb_rounds += 1
                config.message_count = 0

                print(f"-------Round {config.nb_rounds}/{nb_round}: Start-------")

                # Step 4.2: Experiment Data --create key for this specific architecture
                experiment_data[config.attack_objective][config.seed][
                    config.architecture_name
                ] = {"conversation_history": {}, "red_team_history": {}}

                # Step 4.3: Setup specific environment --with specific red-team infrasstructure
                env: EnvironmentSetup = EnvironmentSetup(config=config)

                print("Environment Setup: Completed.")

                # Step 4.4: Set User request to start the conversation round
                request: Message = Message(
                    sender="USER",
                    receiver="PLANNER_AGENT",
                    content=config.user_request,
                )

                # Step 4.5: Start conversation turns
                while config.message_count < config.max_messages:
                    config.message_count += 1
                    # Step 4.5.1: Validate Message
                    is_validated: bool = env.message_bus.validate_message(request)

                    # If is_validated == False, the round is completed
                    if not is_validated:
                        break

                    # Step 4.5.2: Send the message to the right receiver and generate a response
                    response: str = env.message_bus.generate_response(request)

                    # Step 4.5.3: From this response, identify the recipient and create a Message
                    request = env.message_bus.create_message(request, response)

                    print(
                        f"Round {config.nb_rounds}/{nb_round}, Message: {config.message_count}"
                    )

                print(f"-------Round {config.nb_rounds}/{nb_round}: Completed-------")
                # Step 5: Save conversation_history + red_team_history dictionary in Json format

                experiment_data[config.attack_objective][config.seed][
                    config.architecture_name
                ]["conversation_history"] = env.message_bus.message_log

                experiment_data[config.attack_objective][config.seed][
                    config.architecture_name
                ]["red_team_history"] = env.recommender_agent.red_team_history

                save_dict_to_json(
                    experiment_data,
                    f"{config.results_folder_path}/experiment_data.json",
                )
                print(f"Save Data: Data saved for round {config.nb_rounds}.\n\n")

                # Step 6: Clean environment
                del env.recommender_agent.world_model.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


# Main Function: Run the experiment
if __name__ == "__main__":
    run_experiment()
