from typing import Any, Callable, Optional

from mcpomni_connect.utils import logger

async def list_prompts(server_names: list[str], sessions: dict[str, dict[str, Any]]):
    """List all prompts"""
    prompts = []
    for server_name in server_names:
        if sessions[server_name]["connected"]:
            try:
                prompts_response = await sessions[server_name]["session"].list_prompts()
                prompts.extend(prompts_response.prompts)
            except Exception:
                logger.info(f"{server_name} Does not support prompts")
    return prompts


async def find_prompt_server(
    name: str, available_prompts: dict[str, list[dict | object]]
) -> tuple[str, bool]:
    """Find which server has the prompt

    Returns:
        tuple[str, bool]: (server_name, found)
    """
    logger.info(f"Finding prompt: {name} in {available_prompts}")
    for server_name, prompts in available_prompts.items():
        prompt_names = [
            prompt.name if hasattr(prompt, "name") else prompt["name"]
            for prompt in prompts
        ]
        if name in prompt_names:
            return server_name, True
    return "", False


async def get_prompt(
    sessions: dict[str, dict[str, Any]],
    system_prompt: str,
    add_message_to_history: Callable[[str, str], dict[str, Any]],
    llm_call: Callable[[list[dict[str, Any]]], dict[str, Any]],
    debug: bool,
    available_prompts: dict[str, list[str]],
    name: str,
    arguments: Optional[dict] = None,
):
    """Get a prompt"""
    server_name, found = await find_prompt_server(name, available_prompts)
    if debug:
        logger.info(f"Getting prompt: {name} from {server_name}")
    if not found:
        error_message = f"Prompt not found: {name}"
        await add_message_to_history(
            "user", error_message, {"prompt_name": name, "error": True}
        )
        logger.error(error_message)
        return error_message
    try:
        # add the first message to the history to help the llm to know when to use all available tools directly
        await add_message_to_history("user", f"Getting prompt: {name}")
        prompt_response = await sessions[server_name]["session"].get_prompt(
            name, arguments
        )
        if prompt_response:
            if len(prompt_response.messages) == 0:
                error_message = "Error: Prompt returned empty messages list"
                await add_message_to_history(
                    "user", error_message, {"prompt_name": name, "error": True}
                )
                logger.error(error_message)
                return error_message

            message = prompt_response.messages[0]
            message_content = None
            user_role = message.role or "user" if hasattr(message, "role") else None
            if hasattr(message, "content"):
                if hasattr(message.content, "text"):
                    message_content = message.content.text
                else:
                    message_content = str(message.content)

            if debug:
                logger.info(f"LLM processing {user_role} prompt: {message_content}")
            logger.info(f"System prompt: {system_prompt}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": user_role, "content": message_content},
            ]
            llm_response = await llm_call(
                messages=messages,
            )
            if llm_response:
                if hasattr(llm_response, "choices"):
                    response_content = llm_response.choices[0].message 
                elif hasattr(llm_response, "message"):
                    response_content = llm_response.message
            # adding the message to history helps the llm to know when to use all available tools directly
            await add_message_to_history("assistant", response_content)
            return response_content
    except Exception as e:
        error_message = f"Error getting prompt: {e}"
        await add_message_to_history(
            "user", error_message, {"prompt_name": name, "error": True}
        )
        logger.error(error_message)
        return error_message


async def get_prompt_with_react_agent(
    sessions: dict[str, dict[str, Any]],
    system_prompt: str,
    add_message_to_history: Callable[[str, str], dict[str, Any]],
    debug: bool,
    available_prompts: dict[str, list[str]],
    name: str,
    arguments: Optional[dict] = None,
):
    """Get a prompt with the react agent"""
    server_name, found = await find_prompt_server(name, available_prompts)
    if debug:
        logger.info(f"Getting prompt: {name} from {server_name}")
    if not found:
        error_message = f"Prompt not found: {name}"
        await add_message_to_history(
            "user", error_message, {"prompt_name": name, "error": True}
        )
        logger.error(error_message)
        return error_message
    try:
        # add the first message to the history to help the llm to know when to use all available tools directly
        await add_message_to_history("user", f"Getting prompt: {name}")
        prompt_response = await sessions[server_name]["session"].get_prompt(
            name, arguments
        )
        if prompt_response:
            if not prompt_response.messages:  # Check for empty list
                error_message = (
                    "Error getting prompt: Prompt returned empty messages list"
                )
                await add_message_to_history(
                    "user", error_message, {"prompt_name": name, "error": True}
                )
                logger.error(error_message)
                return error_message

            message = prompt_response.messages[0]
            message_content = None
            user_role = message.role or "user" if hasattr(message, "role") else None
            if hasattr(message, "content"):
                if hasattr(message.content, "text"):
                    message_content = message.content.text
                else:
                    message_content = str(message.content)
            if message_content is None:
                error_message = (
                    "Error getting prompt: Message content is missing or invalid"
                )
                await add_message_to_history(
                    "user", error_message, {"prompt_name": name, "error": True}
                )
                logger.error(error_message)
                return error_message
            if debug:
                logger.info(f"LLM processing {user_role} prompt: {message_content}")
            return message_content
    except Exception as e:
        error_message = f"Error getting prompt: {e}"
        await add_message_to_history(
            "user", error_message, {"prompt_name": name, "error": True}
        )
        logger.error(error_message)
        return error_message
