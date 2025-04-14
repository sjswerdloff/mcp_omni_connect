# from mcpomni_connect.react_agent import ReActAgent
from mcpomni_connect.constants import AGENTS_REGISTRY
from typing import Any, Dict, Callable, Optional
import re
import json
from mcpomni_connect.utils import logger
from mcpomni_connect.system_prompts import generate_react_agent_prompt_template


import json
import logging
import os
import re
import uuid
from typing import Any, Callable, Dict, Optional
from mcpomni_connect.utils import (
    logger,
    RobustLoopDetector,
    handle_stuck_state,
)
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
from mcpomni_connect.schemas import AgentState


class ReActAgent:
    """Autonomous agent implementing the ReAct paradigm for task solving through iterative reasoning and tool usage.

    Key Features:
    - JSON-based interaction with external tools and services
    - Structured reasoning loop (Reason → Act → Observe → Repeat)
    - Integrated tool execution with schema validation
    - OpenAI model integration with retry logic
    - Production-ready logging and error handling
    - Dynamic tool schema injection for LLM context
    - Iteration-limited execution for cost control

    Implements a robust agent architecture supporting:
    - Multi-step problem solving with external resources
    - Self-correcting behavior through observation analysis
    - Secure tool execution with parameter validation
    - Maintainable tool ecosystem through plugin-style architecture
    """

    def __init__(
        self,
        max_steps: int = 20,
        tool_call_timeout: int = 30,
    ):
        self.max_steps = max_steps
        self.loop_detector = RobustLoopDetector()
        self.tool_call_timeout = tool_call_timeout
        self.messages = {}
        self.assistant_with_tool_calls = None
        self.pending_tool_responses = []
        self.state = AgentState.IDLE

    def parse_action(
        self, response: str, available_tools: dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse model response to extract actions"""
        try:
            action_start = response.find("Action:")
            if action_start != -1:

                action_text = response[action_start + len("Action:") :].strip()

                # Find the start of the JSON object (the first "{")
                if "{" in action_text:
                    # Start from the first opening brace
                    json_start = action_text.find("{")
                    json_text = action_text[json_start:]

                    # Now find the balanced closing brace
                    open_braces = 0
                    json_end_pos = 0

                    for i, char in enumerate(json_text):
                        if char == "{":
                            open_braces += 1
                        elif char == "}":
                            open_braces -= 1
                            if open_braces == 0:
                                json_end_pos = i + 1
                                break

                    if json_end_pos > 0:
                        # Extract just the balanced JSON part
                        json_str = json_text[:json_end_pos]

                        # Remove any comments
                        json_str = re.sub(
                            r"//.*?(\n|$)", "", json_str, flags=re.MULTILINE
                        )

                        logger.debug("Extracted JSON (fallback): %s", json_str)

                        # Parse the JSON
                        try:
                            action = json.loads(json_str)
                            tool_name = (
                                action["tool"].lower()
                                if "tool" in action
                                else None
                            )
                            tool_args = (
                                action["parameters"]
                                if "parameters" in action
                                else None
                            )
                            # if tool_name is None or tool_args is None, return an error
                            if tool_name is None or tool_args is None:
                                return {
                                    "error": "Invalid JSON format",
                                    "action": False,
                                }

                            # Validate JSON structure and tool exists
                            if "tool" in action and "parameters" in action:
                                for (
                                    server_name,
                                    tools,
                                ) in available_tools.items():
                                    tool_names = [
                                        tool.name.lower() for tool in tools
                                    ]
                                    if tool_name in tool_names:
                                        return {
                                            "action": True,
                                            "tool_name": tool_name,
                                            "tool_args": tool_args,
                                            "server_name": server_name,
                                        }
                            logger.warning("Tool not found: %s", tool_name)
                            return {
                                "action": False,
                                "error": f"Tool {tool_name} not found",
                            }
                        except json.JSONDecodeError:
                            return {
                                "error": "Invalid JSON format",
                                "action": False,
                            }

        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", str(e))
            return {"error": f"Invalid JSON format: {str(e)}", "action": False}
        except Exception as e:
            logger.error("Error parsing response: %s", str(e), exc_info=True)
            return {"error": str(e), "action": False}

    def parse_response(
        self, response: str, available_tools: dict[str, Any], debug: bool = False
    ) -> Dict[str, Any]:
        """Parse model response to extract actions or final answers"""
        try:
            # First, check for a final answer
            if "Final Answer:" in response or "Answer:" in response:
                if debug:
                    logger.info("Final Answer found in response: %s", response)
                # Split the response at "Final Answer:" or "Answer:"
                parts = re.split(
                    r"(?:Final Answer:|Answer:)", response, flags=re.IGNORECASE
                )
                if len(parts) > 1:
                    # Take everything after the "Final Answer:" or "Answer:"
                    answer = parts[-1].strip()
                    return {"answer": answer}

            if "Action" in response:
                if debug:
                    logger.info("Action found in response: %s", response)
                json_match_result = self.parse_action(
                    response, available_tools
                )
                if json_match_result and json_match_result.get("action"):
                    return json_match_result
                return {"error": "No valid action or answer found in response"}
            else:
                # its a normal response
                if debug:
                    logger.info("Normal response found in response: %s", response)
                return {"answer": response}
        except Exception as e:
            logger.error("Error parsing response: %s", str(e), exc_info=True)
            return {"error": str(e)}

        logger.warning(
            "No valid action or answer found in response: %s", response
        )
        return {"error": "No valid action or answer found in response"}

    async def _execute_tool(
        self,
        agent_name: str,
        sessions: dict,
        server_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        add_message_to_history: Callable[[str, str, Optional[dict]], Any],
    ) -> str:
        """Execute tool and return JSON-formatted observation"""
        try:
            result = await sessions[server_name]["session"].call_tool(
                tool_name, tool_args
            )
            # Handle dictionary response
            if isinstance(result, dict):
                if result.get("status") == "success":
                    tool_result = result.get("data", str(result))
                else:
                    tool_result = json.dumps(result)
            # Handle content-based response
            elif hasattr(result, "content"):
                tool_content = result.content
                tool_result = (
                    tool_content[0].text
                    if isinstance(tool_content, list)
                    else str(tool_content)
                )
            else:
                tool_result = str(result)
            # add the tool result to the message history
            tool_metadata = {
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "args": tool_args,
            }
            await add_message_to_history(
                agent_name, "tool", tool_result, tool_metadata
            )
            return tool_result
        except Exception as e:
            return json.dumps(
                {"status": "error", "data": None, "message": str(e)}
            )

    async def update_llm_working_memory(
        self, agent_name: str, message_history: Callable[[], Any]
    ):
        """Update the LLM's working memory with the current message history"""
        # Ensure the messages dictionary has an entry for this agent
        if agent_name not in self.messages:
            self.messages[agent_name] = []

        short_term_memory_message_history = await message_history(agent_name)
        if not short_term_memory_message_history:
            logger.warning(f"No message history found for agent: {agent_name}")
            return

        # Process message history in order that will be sent to LLM
        for _, message in enumerate(short_term_memory_message_history):
            if message["role"] == "user":
                # First flush any pending tool responses if needed
                if (
                    self.assistant_with_tool_calls
                    and self.pending_tool_responses
                ):
                    self.messages[agent_name].append(
                        self.assistant_with_tool_calls
                    )
                    self.messages[agent_name].extend(
                        self.pending_tool_responses
                    )
                    self.assistant_with_tool_calls = None
                    self.pending_tool_responses = []

                # append all the user messages in the message history to the messages that will be sent to LLM
                self.messages[agent_name].append(
                    {"role": "user", "content": message["content"]}
                )

            elif message["role"] == "assistant":
                # Check if the assistant has tool calls
                metadata = message.get("metadata", {})
                if metadata.get("has_tool_calls", False):
                    # If we already have a pending assistant with tool calls, flush it
                    if self.assistant_with_tool_calls:
                        self.messages[agent_name].append(
                            self.assistant_with_tool_calls
                        )
                        self.messages[agent_name].extend(
                            self.pending_tool_responses
                        )
                        self.pending_tool_responses = []

                    # Store this assistant message for later (until we collect all tool responses)
                    self.assistant_with_tool_calls = {
                        "role": "assistant",
                        "content": message["content"],
                        "tool_calls": metadata.get("tool_calls", []),
                    }
                else:
                    # Regular assistant message without tool calls
                    # First flush any pending tool calls
                    if self.assistant_with_tool_calls:
                        self.messages[agent_name].append(
                            self.assistant_with_tool_calls
                        )
                        self.messages[agent_name].extend(
                            self.pending_tool_responses
                        )
                        self.assistant_with_tool_calls = None
                        self.pending_tool_responses = []

                    # Add all the assistant messages in the message history to the messages that will be sent to LLM
                    self.messages[agent_name].append(
                        {"role": "assistant", "content": message["content"]}
                    )

            elif message["role"] == "tool" and "tool_call_id" in message.get(
                "metadata", {}
            ):
                # Collect tool responses
                # Only add if we have a preceding assistant message with tool calls
                if self.assistant_with_tool_calls:
                    self.pending_tool_responses.append(
                        {
                            "role": "tool",
                            "content": message["content"],
                            "tool_call_id": message["metadata"][
                                "tool_call_id"
                            ],
                        }
                    )

            elif message["role"] == "system":
                # add only the system message to the messages that will be sent to LLM.
                # it will be the first message sent to LLM and only one at all times
                self.messages[agent_name].append(
                    {"role": "system", "content": message["content"]}
                )

    async def act(
        self,
        agent_name: str,
        parsed_response: dict,
        response: str,
        add_message_to_history: Callable[[str, str, Optional[dict]], Any],
        sessions: dict,
        system_prompt: str,
    ):
        """Act on the parsed response from the LLM and the observation from the tool call"""
        tool_call_id = str(uuid.uuid4())
        tool_calls_metadata = {
            "has_tool_calls": True,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": parsed_response["tool_name"],
                        "arguments": json.dumps(parsed_response["tool_args"]),
                    },
                }
            ],
        }

        # Add the assistant message with tool calls to history
        await add_message_to_history(
            agent_name, "assistant", response, tool_calls_metadata
        )
        try:
            async with asyncio.timeout(self.tool_call_timeout):
                # Execute the tool
                observation = await self._execute_tool(
                    agent_name,
                    sessions,
                    parsed_response["server_name"],
                    parsed_response["tool_name"],
                    parsed_response["tool_args"],
                    tool_call_id,
                    add_message_to_history,
                )

                # Add the observation to messages and history
                self.messages[agent_name].append(
                    {"role": "user", "content": f"Observation:\n{observation}"}
                )
                await add_message_to_history(
                    agent_name, "user", f"Observation:\n{observation}"
                )
                # set the state to observing
                logger.info(
                    f"Agent state changed from {self.state} to {AgentState.OBSERVING}"
                )
                self.state = AgentState.OBSERVING

                # Check for tool call loop
                self.loop_detector.record_tool_call(
                    str(parsed_response["tool_name"]),
                    str(parsed_response["tool_args"]),
                    str(observation),
                )

        except asyncio.TimeoutError:
            timeout_response = {
                "role": "tool",
                "content": "Tool call timed out. Please try again or use a different approach.",
                "tool_call_id": tool_call_id,
            }
            logger.warning(timeout_response)
            # Add timeout response to the message history
            await add_message_to_history(
                "tool",
                "Tool call timed out. Please try again or use a different approach.",
                {"tool_call_id": tool_call_id},
            )
            # append the timeout response as user message to the messages that will be sent to LLM
            self.messages[agent_name].append(
                {
                    "role": "user",
                    "content": "Observation:\nTool call timed out. Please try again or use a different approach.",
                }
            )

        except Exception as e:
            error_response = {
                "role": "tool",
                "content": f"Error executing tool: {str(e)}",
                "tool_call_id": tool_call_id,
            }
            logger.error(error_response)
            # Add error response to the message history
            await add_message_to_history(
                "tool",
                f"Error executing tool: {str(e)}",
                {"tool_call_id": tool_call_id},
            )
            # append the error response as user message to the messages that will be sent to LLM
            # this ensure the llm knows about the error
            self.messages[agent_name].append(
                {
                    "role": "user",
                    "content": f"Observation:\nError executing tool: {str(e)}",
                }
            )
        if self.loop_detector.is_looping():
            loop_type = self.loop_detector.get_loop_type()
            logger.warning(f"Tool call loop detected: {loop_type}")
            new_system_prompt = handle_stuck_state(system_prompt)
            self.messages = await self.reset_system_prompt(
                self.messages, new_system_prompt, agent_name
            )
            loop_message = (
                f"Observation:\n"
                f"⚠️ Tool call loop detected: {loop_type}\n\n"
                f"Current approach is not working. Please:\n"
                f"1. Analyze why the previous attempts failed\n"
                f"2. Try a completely different tool or approach\n"
                f"3. If stuck, explain the issue to the user\n"
                f"4. Consider breaking down the task into smaller steps\n"
                f"5. Check if the tool parameters need adjustment"
            )
            self.messages[agent_name].append(
                {
                    "role": "user",
                    "content": loop_message,
                }
            )
            logger.info(
                f"Agent state changed from {self.state} to {AgentState.STUCK}"
            )
            self.state = AgentState.STUCK
            self.loop_detector.reset()

    async def reset_system_prompt(
        self, messages: list, system_prompt: str, agent_name: str
    ):
        # Reset system prompt and keep all messages
        old_messages = messages[agent_name][1:]
        messages[agent_name] = [{"role": "system", "content": system_prompt}]
        messages[agent_name].extend(old_messages)
        return messages

    @asynccontextmanager
    async def agent_state_context(self, new_state: AgentState):
        """Context manager to change the agent state"""
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid agent state: {new_state}")
        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Error in agent state context: {e}")
            raise
        finally:
            self.state = previous_state
    async def get_tools_registry(self, available_tools: dict, agent_name: str) -> str:
        tools_section = []
        agent_available_tools = available_tools.get(agent_name, [])

        for tool in agent_available_tools:
            tool_name = str(tool.name)
            tool_description = str(tool.description)
            tool_md = f"#### `{tool_name}`\n{tool_description}"

            # Add parameters if available
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                params = tool.inputSchema.get("properties", {})
                if params:
                    tool_md += "\n\n**Parameters:**\n"
                    tool_md += "| Name | Type | Description |\n"
                    tool_md += "|------|------|-------------|\n"
                    for param_name, param_info in params.items():
                        param_desc = param_info.get("description", "**No description**")
                        param_type = param_info.get("type", "any")
                        tool_md += f"| `{param_name}` | `{param_type}` | {param_desc} |\n"

            tools_section.append(tool_md)

        return "\n\n".join(tools_section)
    async def run(
        self,
        agent_name: str,
        sessions: dict,
        system_prompt: str,
        query: str,
        llm_connection: Callable,
        available_tools: dict[str, Any],
        add_message_to_history: Callable[[str, str, Optional[dict]], Any],
        message_history: Callable[[], Any],
        debug: bool = False,
    ) -> Optional[str]:
        """Execute ReAct loop with JSON communication"""
        # Initialize messages with system prompt
        if agent_name not in self.messages:
            self.messages[agent_name] = []

        # Reset messages for this agent and initialize with system prompt
        self.messages[agent_name] = [
            {"role": "system", "content": system_prompt}
        ]

        
        # Add initial user message to message history
        await add_message_to_history(agent_name, "user", query)

        # Initialize messages with current message history (only once at start)
        await self.update_llm_working_memory(agent_name, message_history)
        available_tools_registry = await self.get_tools_registry(available_tools, agent_name)
        self.messages[agent_name].append(
            {"role": "assistant", "content": f"### Tools Registry Observation\n\n{available_tools_registry}"}
        )
        # check if the agent is in a valid state to run
        if self.state not in [
            AgentState.IDLE,
            AgentState.STUCK,
            AgentState.ERROR,
        ]:
            raise RuntimeError(
                f"Agent is not in a valid state to run: {self.state}"
            )

        # set the agent state to running
        async with self.agent_state_context(AgentState.RUNNING):
            current_steps = 0
            while (
                self.state != AgentState.FINISHED
                and current_steps < self.max_steps
            ):
                current_steps += 1
                try:
                    logger.info(
                        f"Sending messages to LLM: {(self.messages[agent_name])}"
                    )
                    response = await llm_connection.llm_call(
                        messages=self.messages[agent_name]
                    )
                    if response:
                        response = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error("API error: %s", str(e))
                    return None

                parsed_response = self.parse_response(
                    response, available_tools, debug
                )
                # check for final answer
                logger.info(f"current steps: {current_steps}")
                if "answer" in parsed_response:
                    # add the final answer to the message history and the messages that will be sent to LLM
                    self.messages[agent_name].append(
                        {
                            "role": "assistant",
                            "content": parsed_response["answer"],
                        }
                    )
                    await add_message_to_history(
                        agent_name, "assistant", parsed_response["answer"]
                    )
                    # check if the system prompt has changed
                    if (
                        system_prompt
                        != self.messages[agent_name][0]["content"]
                    ):
                        # Reset system prompt and keep all messages
                        self.messages = await self.reset_system_prompt(
                            self.messages, system_prompt, agent_name
                        )
                    logger.info(
                        f"Agent state changed from {self.state} to {AgentState.FINISHED}"
                    )
                    self.state = AgentState.FINISHED
                    # reset the steps
                    current_steps = 0
                    return parsed_response["answer"]

                elif "action" in parsed_response and parsed_response["action"]:
                    # set the state to tool calling
                    logger.info(
                        f"Agent state changed from {self.state} to {AgentState.TOOL_CALLING}"
                    )
                    self.state = AgentState.TOOL_CALLING

                    await self.act(
                        agent_name,
                        parsed_response,
                        response,
                        add_message_to_history,
                        sessions,
                        system_prompt,
                    )
                    continue
                # append the invalid response to the messages and the message history
                error_message = "Invalid response format. Please use the correct required format"
                self.messages[agent_name].append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )
                await add_message_to_history(agent_name, "user", error_message)
                self.loop_detector.record_message(error_message, response)
                if self.loop_detector.is_looping():
                    logger.warning("Loop detected")
                    new_system_prompt = handle_stuck_state(
                        system_prompt, message_stuck_prompt=True
                    )
                    self.messages = await self.reset_system_prompt(
                        self.messages, new_system_prompt, agent_name
                    )
                    loop_message = (
                        f"Observation:\n"
                        f"⚠️ Message loop detected: {self.loop_detector.get_loop_type()}\n"
                        f"The message stuck is: {error_message}\n"
                        f"Current approach is not working. Please:\n"
                        f"1. Analyze why the previous attempts failed\n"
                        f"2. Try a completely different tool or approach\n"
                    )
                    self.messages[agent_name].append(
                        {
                            "role": "user",
                            "content": loop_message,
                        }
                    )
                    self.loop_detector.reset()
                    logger.info(
                        f"Agent state changed from {self.state} to {AgentState.STUCK}"
                    )
                    self.state = AgentState.STUCK


class OrchestratorAgent:
    react_agent = ReActAgent()

    def __init__(self, agent_registry: dict[str, Any]):
        self.agent_registry = agent_registry
        self.orchestrator_messages = []
        self.max_steps = 50

    def parse_action(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract actions"""
        try:
            action_start = response.find("Action:")
            if action_start != -1:

                action_text = response[action_start + len("Action:") :].strip()

                # Find the start of the JSON object (the first "{")
                if "{" in action_text:
                    # Start from the first opening brace
                    json_start = action_text.find("{")
                    json_text = action_text[json_start:]

                    # Now find the balanced closing brace
                    open_braces = 0
                    json_end_pos = 0

                    for i, char in enumerate(json_text):
                        if char == "{":
                            open_braces += 1
                        elif char == "}":
                            open_braces -= 1
                            if open_braces == 0:
                                json_end_pos = i + 1
                                break

                    if json_end_pos > 0:
                        # Extract just the balanced JSON part
                        json_str = json_text[:json_end_pos]

                        # Remove any comments
                        json_str = re.sub(
                            r"//.*?(\n|$)", "", json_str, flags=re.MULTILINE
                        )

                        logger.debug("Extracted JSON (fallback): %s", json_str)

                        # Parse the JSON
                        try:
                            action = json.loads(json_str)
                            logger.info(f"action: {action}")
                            agent_name_to_act = (
                                action["agent_name"]
                                if "agent_name" in action
                                else None
                            )
                            # strip away if Agent or agent is part of the agent name
                            agent_name_to_act = agent_name_to_act.replace("Agent", "").replace("agent", "").strip()
                            task_to_act = (
                                action["task"] if "task" in action else None
                            )
                            # if tool_name is None or tool_args is None, return an error
                            if (
                                agent_name_to_act is None
                                or task_to_act is None
                            ):
                                return {
                                    "error": "Invalid JSON format",
                                    "action": False,
                                }

                            # Validate JSON structure and tool exists
                            if "agent_name" in action and "task" in action:
                                for (
                                    agent_name,
                                    agent_description,
                                ) in self.agent_registry.items():
                                    agent_names = [
                                        agent_name.lower()
                                        for agent_name in self.agent_registry.keys()
                                    ]
                                    if (
                                        agent_name_to_act.lower()
                                        in agent_names
                                    ):
                                        return {
                                            "action": True,
                                            "agent_name": agent_name_to_act,
                                            "task": task_to_act,
                                        }
                            logger.warning(
                                "Agent not found: %s", agent_name_to_act
                            )
                            return {
                                "action": False,
                                "error": f"Agent {agent_name_to_act} not found",
                            }
                        except json.JSONDecodeError:
                            return {
                                "error": "Invalid JSON format",
                                "action": False,
                            }

        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", str(e))
            return {"error": f"Invalid JSON format: {str(e)}", "action": False}
        except Exception as e:
            logger.error("Error parsing response: %s", str(e), exc_info=True)
            return {"error": str(e), "action": False}

    def parse_response(self, response: str, debug: bool = False) -> Dict[str, Any]:
        """Parse model response to extract actions or final answers"""
        try:
            # First, check for a final answer
            if "Final Answer:" in response or "Answer:" in response:
                if debug:
                    logger.info("Final Answer found in response: %s", response)
                # Split the response at "Final Answer:" or "Answer:"
                parts = re.split(
                    r"(?:Final Answer:|Answer:)", response, flags=re.IGNORECASE
                )
                if len(parts) > 1:
                    # Take everything after the "Final Answer:" or "Answer:"
                    answer = parts[-1].strip()
                    return {"answer": answer}

            if "Action" in response:
                if debug:
                    logger.info("Action found in response: %s", response)
                json_match_result = self.parse_action(response)
                if json_match_result and json_match_result.get("action"):
                    return json_match_result
                return {"error": "No valid action or answer found in response"}
            else:
                # its a normal response
                if debug:
                    logger.info("Normal response found in response: %s", response)
                return {"answer": response}
        except Exception as e:
            logger.error("Error parsing response: %s", str(e), exc_info=True)
            return {"error": str(e)}

        logger.warning(
            "No valid action or answer found in response: %s", response
        )
        return {"error": "No valid action or answer found in response"}

    async def create_agent_system_prompt(
        self,
        agent_name: str,
        episodic_memory: str,
        available_tools: dict[str, Any],
    ) -> str:
        server_name = agent_name
        logger.info(f"server_name: {server_name}")
        agent_role = self.agent_registry[server_name]
        agent_system_prompt = generate_react_agent_prompt_template(
            agent_role_prompt=agent_role,
        )
        return agent_system_prompt

    async def update_llm_working_memory(
        self, message_history: Callable[[], Any]
    ):
        """Update the LLM's working memory with the current message history"""
        short_term_memory_message_history = await message_history(
            "orchestrator"
        )
        # Process message history in order that will be sent to LLM
        # for agent_name, messages in short_term_memory_message_history.items():
        #     for _, message in enumerate(messages):
        #         if message["role"] == "user":
        #             # append all the user messages in the message history to the messages that will be sent to LLM
        #             self.orchestrator_messages.append(
        #                 {"role": "user", "content": message["content"]}
        #             )

        #         elif message["role"] == "assistant":
        #             # Add all the assistant messages in the message history to the messages that will be sent to LLM
        #             self.orchestrator_messages.append(
        #                 {"role": "assistant", "content": message["content"]}
        #             )

        #         elif message["role"] == "system":
        #             # add only the system message to the messages that will be sent to LLM.
        #             # it will be the first message sent to LLM and only one at all times
        #             self.orchestrator_messages.append(
        #                 {"role": "system", "content": message["content"]}
        #             )

        for _, message in enumerate(short_term_memory_message_history):
            if message["role"] == "user":
                # append all the user messages in the message history to the messages that will be sent to LLM
                self.orchestrator_messages.append(
                    {"role": "user", "content": message["content"]}
                )

            elif message["role"] == "assistant":
                # Add all the assistant messages in the message history to the messages that will be sent to LLM
                self.orchestrator_messages.append(
                    {"role": "assistant", "content": message["content"]}
                )

            elif message["role"] == "system":
                # add only the system message to the messages that will be sent to LLM.
                # it will be the first message sent to LLM and only one at all times
                self.orchestrator_messages.append(
                    {"role": "system", "content": message["content"]}
                )

    async def act(
        self,
        sessions: dict,
        agent_name: str,
        task: str,
        add_message_to_history: Callable[[str, str, Optional[dict]], Any],
        llm_connection: Callable,
        available_tools: dict[str, Any],
        message_history: Callable[[], Any],
        episodic_memory: str,
    ) -> str:
        """Execute agent and return JSON-formatted observation"""
        try:
            agent_system_prompt = await self.create_agent_system_prompt(
                agent_name=agent_name,
                episodic_memory=episodic_memory,
                available_tools=available_tools,
            )
            observation = await self.react_agent.run(
                agent_name=agent_name,
                sessions=sessions,
                system_prompt=agent_system_prompt,
                query=task,
                llm_connection=llm_connection,
                available_tools=available_tools,
                add_message_to_history=add_message_to_history,
                message_history=message_history,
            )
            # if the observation is empty return general error message
            if not observation:
                observation = "No observation available right now. try again later. or try a different agent."
            # add the observation to the orchestrator messages and the message history
            self.orchestrator_messages.append(
                {
                    "role": "user",
                    "content": f"{agent_name} Agent Observation:\n{observation}",
                }
            )
            await add_message_to_history(
                "orchestrator",
                "user",
                f"{agent_name} Agent Observation:\n{observation}",
            )
            return observation
        except Exception as e:
            logger.error("Error executing agent: %s", str(e), exc_info=True)
    async def agent_registry_tool(self, available_tools: dict[str, Any]) -> str:
        """
        This function is used to create a tool that will return the agent registry
        """
        agent_registries = []
        for server_name, tools in available_tools.items():
            agent_entry = {
                "agent_name": server_name,
                "agent_description": self.agent_registry[server_name],
                "capabilities": [],
            }
            for tool in tools:
                name = (
                    str(tool.name)
                    if tool.name
                    else "No Name available"
                )
                agent_entry["capabilities"].append(name)
            agent_registries.append(agent_entry)
        return "\n".join([
            "### Agent Registry",
            "| Agent Name     | Description                         | Capabilities                     |",
            "|----------------|-------------------------------------|----------------------------------|",
            *[
                f"| {entry['agent_name']} | {entry['agent_description']} | {', '.join(entry['capabilities'])} |"
                for entry in agent_registries
            ]
        ])

    async def run(
        self,
        sessions: dict,
        query: str,
        add_message_to_history: Callable[[str, str, Optional[dict]], Any],
        llm_connection: Callable,
        available_tools: dict[str, Any],
        message_history: Callable[[], Any],
        episodic_memory: str,
        orchestrator_system_prompt: str,
        debug: bool = False,
    ) -> Optional[str]:
        """Execute ReAct loop with JSON communication"""
        # Initialize messages with system prompt
        self.orchestrator_messages = [
            {"role": "system", "content": orchestrator_system_prompt}
        ]

        # Add initial user message to message history
        await add_message_to_history("orchestrator", "user", query)
        await self.update_llm_working_memory(message_history)
        agent_registry_output = await self.agent_registry_tool(available_tools)
        self.orchestrator_messages.append(
            {
                "role": "assistant",
                "content": f"The following is an observation of the currently available agents and their capabilities. Use this list to choose the most suitable agent for the user's task.\n\n{agent_registry_output}",
            }
        )
        current_steps = 0
        while current_steps < self.max_steps:
            current_steps += 1
            try:
                logger.info(
                    f"Sending Orchestrator messages to LLM: {(self.orchestrator_messages)}"
                )
                response = await llm_connection.llm_call(
                    self.orchestrator_messages
                )
                if response:
                    response = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error("API error: %s", str(e))
                return None

            parsed_response = self.parse_response(response, debug)
            # check for final answer
            logger.info(f"current steps: {current_steps}")
            if "answer" in parsed_response:
                # add the final answer to the message history and the messages that will be sent to LLM
                self.orchestrator_messages.append(
                    {
                        "role": "assistant",
                        "content": parsed_response["answer"],
                    }
                )
                await add_message_to_history(
                    "orchestrator", "assistant", parsed_response["answer"]
                )
                # reset the steps
                current_steps = 0
                return parsed_response["answer"]

            elif "action" in parsed_response and parsed_response["action"]:
                logger.info(f"parsed_response: {parsed_response}")
                await self.act(
                    sessions=sessions,
                    agent_name=parsed_response["agent_name"],
                    task=parsed_response["task"],
                    add_message_to_history=add_message_to_history,
                    llm_connection=llm_connection,
                    available_tools=available_tools,
                    message_history=message_history,
                    episodic_memory=episodic_memory,
                )
                continue
            elif "error" in parsed_response:
                # get the error message from the parsed response
                error_message = parsed_response["error"]
            else:
                # append the invalid response to the messages and the message history
                error_message = "Invalid response format. Please use the correct required format"
            self.orchestrator_messages.append(
                {
                    "role": "user",
                    "content": f"{parsed_response["agent_name"]} Agent Observation:\n{error_message}",
                }
            )
            await add_message_to_history("orchestrator", "user", error_message)
