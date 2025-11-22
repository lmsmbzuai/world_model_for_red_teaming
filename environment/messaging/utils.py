"""
* File: ./environment/messaging/utils.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Utilities functions to conduct the messaging module.
* Functions:
    - get_recipient()
    - is_complete()
    - validate_routing()
"""

# Import External Libraries
import re

# Import Local Modules
from environment.messaging.message import Message


def get_recipient(
    agents: list[str], request: Message, response_content: str
) -> tuple[str, str, str]:
    """
    Get the recipient depending on the last sender or the agent's name mentions in the message:
        - Situation 1: If the new sender is not the "PLANNER_AGENT"
        - Situation 2: If the new sender is the "PLANNER_AGENT", find the new receiver in the message content
        - Situation 3: If no valid recipient was found, send back a message to the PLANNER_AGENT to tell it there is an error

    Args:
        agents (list[str]): List of agent's names.
        request (Message): The Message object corresponding to the last message.
        response_content (str): The content of the message.

    Returns:
        Arguments (Args): Specific arguments.
    """

    # Step 1: Set the new Sender
    new_sender: str = request.receiver

    # Step 2: Get the new Receiver
    # Create possible list of new receiver
    recipients = [r for r in agents + ["USER"] if r != "PLANNER_AGENT"]

    # Situation 1: If the new sender is not the "PLANNER_AGENT"
    if new_sender != "PLANNER_AGENT":
        new_receiver: str = "PLANNER_AGENT"
        return new_sender, new_receiver, response_content
    else:
        # Situation 2: If the new sender is the "PLANNER_AGENT", find the new receiver in the message content
        recipients_re: list[str] = [r.replace("_", " ").lower() for r in recipients]
        results: re.Match[str] | None = re.search(
            r"\[([^\]]+)\]", response_content, re.IGNORECASE
        )
        if isinstance(results, re.Match):
            match: str = results.group(1)
            if match:
                recipient = match.replace("_", " ").lower()
                if recipient in recipients_re:
                    index: int = recipients_re.index(recipient)
                    new_receiver = recipients[index]
                    return new_sender, new_receiver, response_content

    # Situation 3: If no valid recipient was found, send back a message to the PLANNER_AGENT to tell it there is an error
    sender = "SYSTEM"
    recipient = "PLANNER_AGENT"
    response_content = """
    [ERROR]: Invalid format. Resend your message using: [AGENT_NAME]: <message>
    Valid agents: [RECOMMENDER_AGENT], [WEATHER_AGENT], [MESSAGING_AGENT], [TICKETING_AGENT]
    Do not reply to this error.
    """
    return sender, recipient, response_content


def is_complete(recipient: str) -> bool:
    """
    Check if planner is responding to USER (end of turn).

    Args:
        recipient (str): The recipient of the last message.

    Returns:
        bool (False): If the Planner Agent contact the user, return True, if it is not the case, return False.
    """

    if recipient == "USER":
        print("\n Planner has responded to user.")
        return True

    return False


def validate_routing(agents: list[str], sender: str, receiver: str) -> bool:
    """
    Enforce communication policies.

    Args:
        agents (list[str]): List of agent's names.
        sender (str): The sender of the last message.
        receiver (str): The recipient of the last message.

    Returns:
        Respect or not (bool): If the agent respect communication policies, return True, if it is not the case, return False.
    """

    # Step 1: Keep only the assistant, not the Planner
    keys: list[str] = [a for a in agents if a != "PLANNER_AGENT"]

    # Step 2: If the assistant is the sender check if the receiver is the PLANNER_AGENT
    if sender in keys:
        if receiver != "PLANNER_AGENT":
            return False
        else:
            return True
    else:
        return True
