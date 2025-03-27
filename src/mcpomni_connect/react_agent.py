import json
import logging
import os
import re
import uuid
from typing import Any, Callable, Dict, Optional

from mcpomni_connect.utils import logger


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
        max_iterations: int = 10,
    ):
        self.max_iterations = max_iterations

    def _first_json_match(
        self, response: str, available_tools: dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            logger.info("First JSON match")
            # Extract JSON between "Action:" and "PAUSE" with a better regex pattern
            # This regex finds the text starting with "Action:" followed by "{" and ending with "}" before "PAUSE"
            action_match = re.search(
                r"Action:\s*(\{.*?\})\s*PAUSE", response, re.DOTALL
            )
            if action_match:
                # Get the JSON string
                json_str = action_match.group(1)

                # Remove any comments (// style comments)
                json_str = re.sub(
                    r"//.*?(\n|$)", "", json_str, flags=re.MULTILINE
                )

                logger.debug("Extracted JSON: %s", json_str)

                # Parse the JSON
                action = json.loads(json_str)

                # Normalize tool name (case insensitive)
                tool_name = (
                    action["tool"].lower() if "tool" in action else None
                )
                tool_args = (
                    action["parameters"] if "parameters" in action else None
                )
                # if tool_name is None or tool_args is None, return an error
                if tool_name is None or tool_args is None:
                    return {"error": "Invalid JSON format", "action": False}

                # Validate JSON structure and tool exists
                if "tool" in action and "parameters" in action:
                    for server_name, tools in available_tools.items():
                        tool_names = [tool.name.lower() for tool in tools]
                        logger.info("Available tool names: %s", tool_names)
                        logger.info("Looking for tool: %s", tool_name)
                        if tool_name in tool_names:
                            logger.info("Tool found: %s", tool_name)
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
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", str(e))
            return {"error": f"Invalid JSON format: {str(e)}", "action": False}
        except Exception as e:
            logger.error("Error parsing response: %s", str(e), exc_info=True)
            return {"error": str(e), "action": False}

    def _second_json_match(
        self, response: str, available_tools: dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            logger.info("Second JSON match")
            action_start = response.find("Action:")
            if action_start != -1:
                # Get everything after "Action:"
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
                                    tool_names = [tool.name.lower() for tool in tools]
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
        self, response: str, available_tools: dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse model response to extract actions or final answers"""
        try:
            # First, check for a final answer
            if "Final Answer:" in response or "Answer:" in response:
                # Split the response at "Final Answer:" or "Answer:"
                parts = re.split(
                    r"(?:Final Answer:|Answer:)", response, flags=re.IGNORECASE
                )
                if len(parts) > 1:
                    # Take everything after the "Final Answer:" or "Answer:"
                    answer = parts[-1].strip()
                    return {"answer": answer}

            if "PAUSE" in response and "Action" in response:
                # Check for JSON action
                first_json_match = self._first_json_match(
                    response, available_tools
                )
                if (
                    first_json_match
                    and first_json_match.get("action") is False
                ):
                    second_json_match = self._second_json_match(
                        response, available_tools
                    )
                    if second_json_match and second_json_match.get("action"):
                        return second_json_match
                if first_json_match and first_json_match.get("action"):
                    return first_json_match
                return {"error": "No valid action or answer found in response"}
            else:
                # its a normal response
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
        sessions: dict,
        server_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        add_message_to_history: Callable[[str, str, Optional[dict]], Any],
    ) -> str:
        """Execute tool and return JSON-formatted observation"""
        logger.info("Executing %s with parameters: %s", tool_name, tool_args)
        try:
            result = await sessions[server_name]["session"].call_tool(
                tool_name, tool_args
            )
            logger.info("Tool result: %s", result)
            
            # Handle dictionary response
            if isinstance(result, dict):
                if result.get("status") == "success":
                    tool_result = result.get("data", str(result))
                else:
                    tool_result = json.dumps(result)
            # Handle content-based response
            elif hasattr(result, "content"):
                tool_content = result.content
                tool_result = tool_content[0].text if isinstance(tool_content, list) else str(tool_content)
            else:
                tool_result = str(result)
                
            logger.info("Tool content: %s", tool_result)
            tool_metadata = {
                "tool_call_id": str(uuid.uuid4()),
                "tool": tool_name,
                "args": tool_args,
            }
            await add_message_to_history("tool", tool_result, tool_metadata)
            return tool_result
        except Exception as e:
            return json.dumps(
                {"status": "error", "data": None, "message": str(e)}
            )

    async def run(
        self,
        sessions: dict,
        system_prompt: str,
        query: str,
        llm_connection: Callable,
        available_tools: dict[str, Any],
        add_message_to_history: Callable[[str, str, Optional[dict]], Any],
        message_history: list[dict[str, Any]],
    ) -> Optional[str]:
        """Execute ReAct loop with JSON communication"""
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # add initial user message to message history
        await add_message_to_history("user", query)

        # Track assistant with tool calls and pending tool responses
        assistant_with_tool_calls = None
        pending_tool_responses = []

        # Process message history in order that will be sent to LLM
        for _, message in enumerate(message_history):
            if message["role"] == "user":
                # First flush any pending tool responses if needed
                if assistant_with_tool_calls and pending_tool_responses:
                    messages.append(assistant_with_tool_calls)
                    messages.extend(pending_tool_responses)
                    assistant_with_tool_calls = None
                    pending_tool_responses = []

                # append all the user messages in the message history to the messages that will be sent to LLM
                messages.append(
                    {"role": "user", "content": message["content"]}
                )

            elif message["role"] == "assistant":
                # Check if the assistant has tool calls
                metadata = message.get("metadata", {})
                if metadata.get("has_tool_calls", False):
                    # If we already have a pending assistant with tool calls, flush it
                    if assistant_with_tool_calls:
                        messages.append(assistant_with_tool_calls)
                        messages.extend(pending_tool_responses)
                        pending_tool_responses = []

                    # Store this assistant message for later (until we collect all tool responses)
                    assistant_with_tool_calls = {
                        "role": "assistant",
                        "content": message["content"],
                        "tool_calls": metadata.get("tool_calls", []),
                    }
                else:
                    # Regular assistant message without tool calls
                    # First flush any pending tool calls
                    if assistant_with_tool_calls:
                        messages.append(assistant_with_tool_calls)
                        messages.extend(pending_tool_responses)
                        assistant_with_tool_calls = None
                        pending_tool_responses = []

                    # Add all the assistant messages in the message history to the messages that will be sent to LLM
                    messages.append(
                        {"role": "assistant", "content": message["content"]}
                    )

            elif message["role"] == "tool" and "tool_call_id" in message.get(
                "metadata", {}
            ):
                # Collect tool responses
                # Only add if we have a preceding assistant message with tool calls
                if assistant_with_tool_calls:
                    pending_tool_responses.append(
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
                messages.append(
                    {"role": "system", "content": message["content"]}
                )

        # Flush any remaining pending tool calls at the end
        if assistant_with_tool_calls:
            messages.append(assistant_with_tool_calls)
            messages.extend(pending_tool_responses)

        logger.info("Running ReAct agent")
        for iteration in range(1, self.max_iterations + 1):
            logger.info("Iteration %d/%d", iteration, self.max_iterations)

            try:
                logger.info(f"Sending messages to LLM: {len(messages)}")
                logger.info(f"Messages: {messages}")
                response = await llm_connection.llm_call(messages)
                if response:
                    response = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error("API error: %s", str(e))
                return None

            parsed_response = self.parse_response(response, available_tools)
            # add the assistant message to the messages that will be sent to LLM
            messages.append({"role": "assistant", "content": response})
            # add the assistant message to the message history which will be sent to LLM later
            await add_message_to_history("assistant", response)
            if "answer" in parsed_response:
                logger.info("Final answer: %s", parsed_response["answer"])
                # add the final answer to the message history which will be sent to LLM later
                await add_message_to_history(
                    "assistant", parsed_response["answer"]
                )
                return parsed_response["answer"]

            if "action" in parsed_response and parsed_response["action"]:
                # generate the tool call metadata
                tool_calls_metadata = {
                    "has_tool_calls": True,
                    "tool_calls": [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": parsed_response["tool_name"],
                                "arguments": json.dumps(
                                    parsed_response["tool_args"]
                                ),
                            },
                        }
                    ],
                }
                # add the assistant message to the message history which will be sent to LLM later
                # this is when the assistant call any tool
                await add_message_to_history(
                    "assistant", response, tool_calls_metadata
                )
                # execute the tool
                observation = await self._execute_tool(
                    sessions,
                    parsed_response["server_name"],
                    parsed_response["tool_name"],
                    parsed_response["tool_args"],
                    add_message_to_history,
                )
                # add the observation to the messages that will be sent to LLM
                messages.append(
                    {"role": "user", "content": f"Observation:\n{observation}"}
                )
                # add the observation to the message history which will be sent to LLM later
                await add_message_to_history(
                    "user", f"Observation:\n{observation}"
                )
                continue
            # add the invalid response to the messages that will be sent to LLM
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid response format. Please use JSON for actions.",
                }
            )
            # add the invalid response to the message history which will be sent to LLM later
            await add_message_to_history(
                "user", "Invalid response format. Please use JSON for actions."
            )

        logger.warning("Max iterations reached")
        # add the max iterations reached to the message history which will be sent to LLM later
        await add_message_to_history("user", "Max iterations reached")
        messages.append({"role": "user", "content": "Max iterations reached"})
        # do we return None or a message?
        # return None
        return "Max iterations reached"
