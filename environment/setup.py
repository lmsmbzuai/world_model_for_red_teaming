"""
* File: ./environment/setup.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Class to manage the full environment setup.
"""

# Import Local Modules
from environment.agents import (
    MessagingAgent,
    PlannerAgent,
    RecommenderAgent,
    TicketingAgent,
    WeatherAgent,
)
from environment.messaging.message_bus import MessageBus
from environment.model_manager import ModelManager
from experiment.config import ExperimentConfig

# Define Agent types
AgentTypes = (
    PlannerAgent | RecommenderAgent | WeatherAgent | MessagingAgent | TicketingAgent
)


class EnvironmentSetup:
    def __init__(self, config: ExperimentConfig) -> None:
        """
        Class to manage the full environment setup:
            - Set the model manager class
            - Create each agent for the environment
            - Set the message bus --for the communication between the agent

        Args:
            config (ExperimentConfig): The specific configurations for the environment setup.

        Returns:
            None.
        """
        # Step 1: Set the model manager
        # --for the target system
        self.model_manager_target: ModelManager = ModelManager(
            config=config, model_name=config.target_model_name
        )
        # --for the red-team agent
        self.model_manager_red_team: ModelManager = ModelManager(
            config=config, model_name=config.red_team_agent_model_name
        )

        # Step 2: Create each agent for the environment
        self.planner_agent: PlannerAgent = PlannerAgent(
            config=config, model_manager=self.model_manager_target
        )
        self.weather_agent: WeatherAgent = WeatherAgent(
            config=config,
            model_manager=self.model_manager_target,
        )
        self.messaging_agent: MessagingAgent = MessagingAgent(
            config=config,
            model_manager=self.model_manager_target,
        )
        self.ticketing_agent: TicketingAgent = TicketingAgent(
            config=config,
            model_manager=self.model_manager_target,
        )

        print("====Step 3.2 - Travel Planning agentic environment Setup: Completed====")

        self.recommender_agent: RecommenderAgent = RecommenderAgent(
            config=config,
            model_manager=self.model_manager_red_team,
        )

        print("====Step 3.4 - Red-Team Setup: Completed====")

        self.agents: list[AgentTypes] = [
            self.planner_agent,
            self.recommender_agent,
            self.weather_agent,
            self.messaging_agent,
            self.ticketing_agent,
        ]

        # Step 5: Set the message bus
        self.message_bus: MessageBus = MessageBus(agents=self.agents)

        print("====Step 3.5 - Communication system Setup: Completed====")
