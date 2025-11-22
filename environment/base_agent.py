"""
* File: ./environment/base_agent.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Fundamental class to manage all the agents.
"""

# Import Local Modules
from environment.model_manager import ModelManager
from experiment.config import ExperimentConfig


class BaseAgent:
    def __init__(
        self,
        agent_name: str,
        config: ExperimentConfig,
        system_prompt: str,
        model_manager: ModelManager,
    ) -> None:
        """
        Fundamental class to manage all the agents.

        Args:
            agent_name (str): Model name.
            config (ExperimentConfig): Class of the specific configurations.
            system_prompt (str): Specific text prompt.
            model_manager (ModelManager): Class to manage the specific models for the agents.

        Returns:
            None.
        """

        self.agent_name: str = agent_name
        self.config: ExperimentConfig = config
        self.seed: int = config.seed
        self.system_prompt: str = system_prompt
        self.model_manager: ModelManager = model_manager
        self.conversation_history: list[dict[str, str]] = []

    def process(self, message_content: str, sender: str) -> str:
        """
        Process incoming message, generate an answer and return Message Object:
            - Format message
            - Generate Answer
            - Save the answer in the conversation history dictionary

        Args:
            message_content (str): Content of the message.
            sender (str): Sender of the message.

        Returns:
            None.
        """

        # Step 1: Format message
        formatted_message = f"[FROM: {sender}]\n{message_content}"

        # Step 2: Generate Answer
        if (
            sender == "SYSTEM"
        ):  # If the Planner Agent didn't mention a specific recipient in its previous message
            message = formatted_message + f"This is your message: {message_content}"
            history = [{"role": "user", "content": message}]
            response: str = self.model_manager.generate(
                system_prompt=self.system_prompt,
                conversation_history=history,
            )

        else:
            self.conversation_history.append(
                {"role": "user", "content": formatted_message}
            )
            response = self.model_manager.generate(
                system_prompt=self.system_prompt,
                conversation_history=self.conversation_history,
            )

        # Step 3: Save the answer in the conversation history dictionary
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset_history(self) -> None:
        """
        Clear conversation history.
        Args: None.
        Returns: None.
        """
        self.conversation_history = []
