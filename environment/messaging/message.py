"""
* File: ./environment/messaging/message.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Orchestrate the communication between agents.
"""


class Message:
    def __init__(
        self, sender: str, receiver: str, content: str, message_type: str = "request"
    ) -> None:
        """
        Class used to create a Message that contains:
            - sender
            - receiver
            - content
            - message_type

        Args:
            sender (str): Sender of the message.
            receiver (str): Recipient of the message.
            content (str): Content of the message.
            message_type (str = "request"): Type of the message.

        Returns:
            None.
        """
        self.sender: str = sender
        self.receiver: str = receiver
        self.content: str = content
        self.message_type: str = message_type

    def to_dict(self):
        """
        Return the Message object as a dictionary.
        Args: None.
        Returns: None.
        """
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.message_type,
        }
