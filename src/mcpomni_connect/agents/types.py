# types.py
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class AgentConfig(BaseModel):
    agent_name: str
    request_limit: int
    total_tokens_limit: int
    max_steps: int = Field(gt=0, le=20)
    tool_call_timeout: int = Field(gt=1, le=60)
    mcp_enabled: bool = False


class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    TOOL_CALLING = "tool_calling"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"
    STUCK = "stuck"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str = "function"
    function: ToolFunction


class ToolCallMetadata(BaseModel):
    has_tool_calls: bool = False
    tool_calls: List[ToolCall] = []
    tool_call_id: Optional[UUID] = None


class Message(BaseModel):
    role: MessageRole
    content: str
    tool_call_id: str = None
    tool_calls: str = None
    metadata: Optional[ToolCallMetadata] = None


class ParsedResponse(BaseModel):
    action: Optional[bool] = None
    data: Optional[str] = None
    error: Optional[str] = None
    answer: Optional[str] = None


class ToolCallResult(BaseModel):
    tool_executor: Any  # ToolExecutor instance
    tool_name: str
    tool_args: Dict


class ToolError(BaseModel):
    observation: str


class ToolData(BaseModel):
    action: bool
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    error: Optional[str] = None


class ToolCallRecord(BaseModel):
    tool_name: str
    tool_args: str
    observation: str


class ToolParameter(BaseModel):
    type: str
    description: str


class ToolRegistryEntry(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter] = []


class ToolExecutorConfig(BaseModel):
    handler: Any  # ToolExecutor instance
    tool_data: Dict[str, Any]
    available_tools: Dict[str, Any]


class LoopDetectorConfig(BaseModel):
    max_repeats: int = 3
    similarity_threshold: float = 0.9
