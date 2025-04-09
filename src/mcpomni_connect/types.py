from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from enum import Enum


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class ContextInclusion(str, Enum):
    NONE = "none"
    THIS_SERVER = "thisServer"
    ALL_SERVERS = "allServers"


@dataclass
class ModelHint:
    name: Optional[str] = None


@dataclass
class ModelPreferences:
    hints: Optional[List[ModelHint]] = None
    cost_priority: Optional[float] = None
    speed_priority: Optional[float] = None
    intelligence_priority: Optional[float] = None


@dataclass
class MessageContent:
    type: ContentType
    text: Optional[str] = None
    data: Optional[str] = None  # base64 encoded for images
    mime_type: Optional[str] = None


@dataclass
class Message:
    role: str
    content: MessageContent


@dataclass
class CreateMessageRequestParams:
    messages: List[Message]
    model_preferences: Optional[ModelPreferences] = None
    system_prompt: Optional[str] = None
    include_context: Optional[ContextInclusion] = None
    temperature: Optional[float] = None
    max_tokens: int
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CreateMessageResult:
    model: str
    stop_reason: Optional[str] = None
    role: str = "assistant"
    content: Optional[MessageContent] = None


@dataclass
class ErrorData:
    code: str
    message: str
