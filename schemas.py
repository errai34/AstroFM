from enum import Enum
from pydantic import BaseModel


class Sender(str, Enum):
    BOT = "bot"
    YOU = "you"


class MessageType(str, Enum):
    START = "start"
    STREAM = "stream"
    END = "end"
    ERROR = "error"
    INFO = "info"


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: Sender
    message: str
    type: MessageType
