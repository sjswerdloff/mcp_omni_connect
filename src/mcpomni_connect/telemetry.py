import os
import json
from datetime import datetime
from typing import Optional, List, Any, Dict
from mcpomni_connect.utils import logger


class TelemetryLogger:
    """A class for logging telemetry data from agents and tools."""

    def __init__(
        self, log_dir: str = "./telemetry_logs", file_prefix: str = "telemetry"
    ):
        """Initialize the telemetry logger.

        Args:
            log_dir: Directory to store log files
            file_prefix: Prefix for log file names
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(
            log_dir, f"{file_prefix}_{timestamp}.jsonl"
        )
        self.logger = logger

    async def log_agent_step(
        self,
        source: str,
        agent_name: str,
        task: str,
        thought: Optional[str] = None,
        action: Optional[dict] = None,
        observation: Optional[Any] = None,
        status: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
        agent_state: Optional[str] = None,
        tool_calls: Optional[List[dict]] = None,
        extra: Optional[dict] = None,
    ):
        """Log a step in the agent's execution.

        Args:
            source: Source of the log (e.g., 'orchestrator', 'react_agent')
            agent_name: Name of the agent
            task: Current task being executed
            thought: Agent's thought process
            action: Action taken by the agent
            observation: Result of the action
            status: Status of the step ('success', 'fail', 'in_progress')
            duration_ms: Time taken for the step in milliseconds
            error: Any error that occurred
            agent_state: Current state of the agent
            tool_calls: List of tool calls made
            extra: Additional metadata
        """
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": source,
                "agent_name": agent_name,
                "task": task,
                "thought": thought,
                "action": action,
                "observation": observation,
                "status": status,
                "duration_ms": duration_ms,
                "error": error,
                "agent_state": agent_state,
                "tool_calls": tool_calls,
            }

            if extra:
                entry.update(extra)

            # Convert observation to string if it's a list or dict
            if isinstance(entry["observation"], (list, dict)):
                entry["observation"] = json.dumps(entry["observation"])

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            if self.logger:
                self.logger.debug(
                    f"Logged telemetry for {agent_name}: {status}"
                )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to log telemetry: {e}")

    async def log_tool_call(
        self,
        source: str,
        agent_name: str,
        tool_name: str,
        tool_args: dict,
        result: Any,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        """Log a tool call and its result.

        Args:
            source: Source of the log
            agent_name: Name of the agent making the call
            tool_name: Name of the tool called
            tool_args: Arguments passed to the tool
            result: Result of the tool call
            duration_ms: Time taken for the tool call
            error: Any error that occurred
            extra: Additional metadata
        """
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": source,
                "agent_name": agent_name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result,
                "duration_ms": duration_ms,
                "error": error,
            }

            if extra:
                entry.update(extra)

            # Convert result to string if it's a list or dict
            if isinstance(entry["result"], (list, dict)):
                entry["result"] = json.dumps(entry["result"])

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            if self.logger:
                self.logger.debug(
                    f"Logged tool call for {agent_name}: {tool_name}"
                )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to log tool call: {e}")


# Create a global instance
telemetry_logger = TelemetryLogger()
