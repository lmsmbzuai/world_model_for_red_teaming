"""
* File: ./environment/messaging/message_bus.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Orchestrate the communication between agents.
"""

# Import External Libraries
import logging

# Import Local Modules
from environment.agents import (
    MessagingAgent,
    PlannerAgent,
    RecommenderAgent,
    TicketingAgent,
    WeatherAgent,
)
from environment.messaging.message import Message
from environment.messaging.utils import get_recipient, is_complete, validate_routing

# Define Agent types
AgentTypes = (
    PlannerAgent | RecommenderAgent | WeatherAgent | MessagingAgent | TicketingAgent
)


class MessageBus:
    def __init__(self, agents: list[AgentTypes]) -> None:
        """
        Initialize message bus with all agents.
        Orchestrate the communication between agents.

        Args:
            agents (list[AgentTypes]): Dictionary of {agent_name: agent_instance}.

        Returns:
            None.
        """
        self.agents: dict[str, AgentTypes] = {
            agent.agent_name: agent for agent in agents
        }
        self.message_log: list[dict[str, str]] = []
        self.logger: logging.Logger = logging.getLogger("MessageBus")

    def validate_message(self, message: Message) -> bool:
        """
        Log and Validate Messages:
            - Log message
            - Check the status of the conversation
            - Check if target agent exists
            - Check the routing rules

        Args:
            message (Message): Message object representing the last message.

        Returns:
            Message validated or not (bool): Return True if the message is validated, return False if it is not.
        """

        # Step 1: Log message
        self.message_log.append(message.to_dict())
        self.logger.info(
            f"{message.sender} -> {message.receiver}: {message.content[:50]}..."
        )
        # Step 2: Check the status of the conversation --is completed?
        is_completed: bool = is_complete(message.receiver)
        if is_completed:
            return (
                False  # Because the conversation must stop --conversation not validated
            )

        # Step 2: Check if target agent exists --receiver
        if message.receiver not in self.agents and message.receiver != "USER":
            error_msg = f"Unknown agent: {message.receiver}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 3: Check the routing rules --policies
        is_route_validated: bool = validate_routing(
            list(self.agents), message.sender, message.receiver
        )
        if not is_route_validated:
            error_msg = (
                f"Policy violation: {message.sender} cannot send to {message.receiver}"
            )
            self.logger.error(error_msg)
            raise PermissionError(error_msg)

        return True

    def generate_response(self, message: Message) -> str:
        """
        Route message to appropriate agent and return response.

        Args:
            message (Message): Message object to send.

        Returns:
            message (Message): Response from target agent.

        Raises:
            PermissionError: If routing violates policy
            ValueError: If agent doesn't exist
        """

        # Step 1: Get target agent instance
        target_agent = self.agents[message.receiver]

        # Step 2: Process message and get response
        try:
            response: str = target_agent.process(message.content, message.sender)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            raise
        return response

    def create_message(self, input_message: Message, response_content: str) -> Message:
        """
        Based on the content of the new message create a new Message object:
            - Get the recipient from the content response of the agent
            - Create a new Message with the right sender and receiver

        Args:
            input_message (Message): Content of the last message.
            response_content (str): Content of the new message.

        Returns:
            response (Message): Message object containing the new message content.
        """

        # Step 1: Get the recipient from the content response of the agent
        # results = new_sender, new_receiver, response_content
        results: tuple[str, str, str] = get_recipient(
            list(self.agents), input_message, response_content
        )
        new_sender, new_receiver, response_content = results

        # Step 2: Create a new Message with the right sender and receiver
        response: Message = Message(
            sender=new_sender,
            receiver=new_receiver,
            content=response_content,
        )

        return response

    def get_message_log(self):
        """
        Return all messages for analysis.
        Args: None.
        Returns: Arguments (Args): Specific arguments.
        """
        return self.message_log
