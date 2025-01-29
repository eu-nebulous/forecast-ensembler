
import json
import logging
from typing import Any, Dict, Optional

from exn.core.publisher import Publisher as EXNPublisher

class Publisher(EXNPublisher):
    """
    Custom Publisher subclass to send messages to ActiveMQ.
    """

    def __init__(self, name: str, address: str, topic: str = ""):
        """
        Initialize the CustomPublisher.
        """
        super().__init__(name, address, topic)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Publisher for topic '{self.topic}' initialized for address '{self.address}'.")

    def send_message(self, message: Dict[str, Any], application: Optional[str] = None):
        """
        Send a message to the specified destination.
        """
        try:
            self.send(message, application=application)
            self.logger.info(f"Sent message to '{self.address}' with application='{application}': {message}")
        except Exception as e:
            self.logger.error(f"Failed to send message to '{self.address}': {e}")