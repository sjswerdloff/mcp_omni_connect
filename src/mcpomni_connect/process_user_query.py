import json
from typing import Any, Callable, Optional

from mcpomni_connect.utils import logger


# process a query using LLM and available tools
async def process_query(
    query: str,
    system_prompt: str,
    llm_connection: Callable[[], Any],
    sessions: dict[str, Any],
    server_names: list[str],
    tools_list: list[dict[str, Any]],
    available_tools: dict[str, Any],
    add_message_to_history: Callable[[str, str, Optional[dict]], Any],
    message_history: list[dict[str, Any]],
    debug: bool = False,
) -> str:
    """Process a query using LLM and available tools"""

    # add user query to history
    await add_message_to_history("user", query)

    # prepare messages for LLM
    messages = []

    # add system prompt and user query to messages
    messages.append({"role": "system", "content": system_prompt})
    # track assistant with tool calls and pending tool responses
    assistant_with_tool_calls = None
    pending_tool_responses = []

    # process message history in order
    for _, message in enumerate(message_history):
        if message["role"] == "user":
            # First flush any pending tool responses if needed
            if assistant_with_tool_calls and pending_tool_responses:
                messages.append(assistant_with_tool_calls)
                messages.extend(pending_tool_responses)
                assistant_with_tool_calls = None
                pending_tool_responses = []

            # then add user message to messages that will be sent to LLM
            messages.append({"role": "user", "content": message["content"]})

        elif message["role"] == "assistant":
            # check if the assistant with tool call
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

                # add the regular assistant message to messages
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
                        "tool_call_id": message["metadata"]["tool_call_id"],
                    }
                )

        elif message["role"] == "system":
            # add system message to messages
            messages.append({"role": "system", "content": message["content"]})

    # Flush any remaining pending tool calls at the end
    if assistant_with_tool_calls:
        messages.append(assistant_with_tool_calls)
        messages.extend(pending_tool_responses)

    if debug:
        logger.info(f"Prepared {len(messages)} messages for LLM")
        for i, message in enumerate(messages):
            role = message["role"]
            has_tool_calls = "tool_calls" in message
            preview = (
                message["content"][:50] + "..." if message["content"] else ""
            )
            logger.info(
                f"Message {i}: {role} {'with tool_calls' if has_tool_calls else ''} - {preview}"
            )

    # list available tools
    all_available_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools_list
    ]

    if debug:
        tool_names = [tool["function"]["name"] for tool in all_available_tools]
        logger.info(f"Available tools for query: {tool_names}")
        logger.info(f"Sending {len(messages)} messages to LLM")

    try:
        # Initial LLM API call
        response = await llm_connection.llm_call(
            messages=messages, tools=all_available_tools
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_message = f"Error processing query: {e}"
        return error_message

    # Process response and handle tool calls
    assistant_message = response.choices[0].message
    initial_response = assistant_message.content or ""

    # Process tool calls
    final_text = []
    tool_results = []

    tool_calls_metadata = {}
    if assistant_message.tool_calls:
        tool_calls_metadata = {
            "has_tool_calls": True,
            "tool_calls": assistant_message.tool_calls,
        }
        if debug:
            logger.info(
                f"Processing {len(assistant_message.tool_calls)} tool calls"
            )

        # Properly append assistant message with tool calls
        messages.append(
            {
                "role": "assistant",
                "content": initial_response,  # This should be a string or null
                "tool_calls": assistant_message.tool_calls,
            }
        )
    # add the assistant message to history with tool calls metadata
    await add_message_to_history(
        "assistant", initial_response, tool_calls_metadata
    )
    final_text.append(initial_response)
    if assistant_message.tool_calls:
        if debug:
            logger.info(
                f"Processing {len(assistant_message.tool_calls)} tool calls"
            )

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            # execute tool call
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse tool arguments for {tool_name}: {tool_args}"
                    )
                    tool_args = {}
            if debug:
                logger.info(
                    f"Processing tool call: {tool_name} with args {tool_args}"
                )
            # execute tool call on the server
            try:
                tool_content = None
                if debug:
                    logger.info(f"Looking for tool {tool_name} in available tools")
                
                for server_name, tools in available_tools.items():
                    # Get tool names, handling both Mock objects and regular tools
                    tool_names = []
                    for tool in tools:
                        if hasattr(tool, 'name'):
                            if debug:
                                logger.info(f"Found tool with name attribute: {tool.name}")
                            tool_names.append(tool.name)
                        elif isinstance(tool, str):
                            if debug:
                                logger.info(f"Found tool with string name: {tool}")
                            tool_names.append(tool)
                    
                    if debug:
                        logger.info(f"Available tool names in {server_name}: {tool_names}")
                    
                    if tool_name in tool_names:
                        if debug:
                            logger.info(f"Found matching tool {tool_name} in {server_name}")
                        result = await sessions[server_name][
                            "session"
                        ].call_tool(tool_name, tool_args)
                        tool_content = (
                            result.content
                            if hasattr(result, "content")
                            else str(result)
                        )
                        break
                
                if tool_content is None:
                    raise Exception(f"Tool {tool_name} not found in any server")

                # Handle the result content appropriately
                if (
                    hasattr(tool_content, "__getitem__")
                    and len(tool_content) > 0
                    and hasattr(tool_content[0], "text")
                ):
                    tool_content = tool_content[0].text
                else:
                    tool_content = tool_content
                tool_results.append(
                    {"call": tool_name, "result": tool_content}
                )
                if debug:
                    result_preview = (
                        tool_content[:200] + "..."
                        if len(str(tool_content)) > 200
                        else str(tool_content)
                    )
                    logger.info(f"Tool result preview: {result_preview}")

                # add the tool result to the messages
                messages.append(
                    {
                        "role": "tool",
                        "content": str(
                            tool_content
                        ),  # Ensure content is a string
                        "tool_call_id": tool_call.id,
                    }
                )
                # add message to history
                await add_message_to_history(
                    "tool",
                    str(tool_content),
                    {
                        "tool_call_id": tool_call.id,
                        "tool": tool_name,
                        "args": tool_args,
                    },
                )
            except Exception as e:
                error_message = f"Error executing tool call {tool_name}: {e}"
                logger.error(error_message)
                # append the message regardless of error
                messages.append(
                    {
                        "role": "tool",
                        "content": error_message,
                        "tool_call_id": tool_call.id,
                    }
                )
                # add error message to history
                await add_message_to_history(
                    "tool",
                    error_message,
                    {
                        "tool_call_id": tool_call.id,
                        "tool": tool_name,
                        "args": tool_args,
                        "error": True,
                    },
                )
                final_text.append(
                    f"\n[Error executing tool call {tool_name}: {error_message}]"
                )
        if debug:
            logger.info("Getting final response from llm with tool results")

        try:
            second_response = await llm_connection.llm_call(
                messages=messages,
            )
            final_assistant_message = second_response.choices[0].message
            response_content = final_assistant_message.content or ""
            await add_message_to_history("assistant", response_content)
            final_text.append(response_content)
        except Exception as e:
            error_message = f"Error getting final response from llm: {e}"
            logger.error(error_message)
            await add_message_to_history(
                "assistant", error_message, {"error": True}
            )
            final_text.append(
                f"\n[Error getting final response from llm: {e}]"
            )

    return "\n".join(final_text)
