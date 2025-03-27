from typing import Any, Callable

from mcpomni_connect.utils import logger


# list all resources in mcp server
async def list_resources(
    server_names: list[str], sessions: dict[str, dict[str, Any]]
):
    """List all resources"""
    resources = []
    for server_name in server_names:
        if sessions[server_name]["connected"]:
            try:
                resources_response = await sessions[server_name][
                    "session"
                ].list_resources()
                resources.extend(resources_response.resources)
            except Exception as e:
                logger.info(f"{server_name} Does not support resources")
    return resources


async def find_resource_server(
    uri: str, available_resources: dict[str, list[str]]
) -> tuple[str, bool]:
    """Find which server has the resource

    Returns:
        tuple[str, bool]: (server_name, found)
    """
    for server_name, resources in available_resources.items():
        resource_uris = [str(res.uri) for res in resources]
        if uri in resource_uris:
            return server_name, True
    return "", False


# read a resource from mcp server
async def read_resource(
    uri: str,
    sessions: dict[str, dict[str, Any]],
    available_resources: dict[str, list[str]],
    add_message_to_history: Callable[[str, str], dict[str, Any]],
    llm_call: Callable[[list[dict[str, Any]]], dict[str, Any]],
    debug: bool = False,
):
    """Read a resource"""
    if debug:
        logger.info(f"Reading resource: {uri}")
    # add the first message to the history
    await add_message_to_history("user", f"Reading resource: {uri}")
    server_name, found = await find_resource_server(uri, available_resources)
    if not found:
        error_message = f"Resource not found: {uri}"
        logger.error(error_message)
        # add the error message to the history
        await add_message_to_history(
            "user", error_message, {"resource_uri": uri, "error": True}
        )
        return error_message
    logger.info(f"Resource found in {server_name}")
    try:
        resource_response = await sessions[server_name][
            "session"
        ].read_resource(uri)
        if debug:
            logger.info("LLM processing resource")
        llm_response = await llm_call(
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the document content and provide a clear, concise summary that captures all essential information. Focus on key points, main concepts, and critical details that give the user a complete understanding without reading the entire document. Present your summary using bullet points for main ideas followed by a brief paragraph for context when needed. Include any technical terms, specifications, instructions, or warnings that are vital to proper understanding. Do not include phrases like 'here is your summary' or 'in summary' - deliver only the informative content directly.",
                },
                {"role": "user", "content": str(resource_response)},
            ]
        )
        response_content = llm_response.choices[0].message.content or ""
        if not response_content:
            response_content = "No content found for resource"
        # add the response from the LLM to the history
        await add_message_to_history("assistant", response_content)
        return response_content
    except Exception as e:
        error_message = f"Error reading resource: {e}"
        logger.error(error_message)
        # add the error message to the history
        await add_message_to_history(
            "user", error_message, {"resource_uri": uri, "error": True}
        )
        return error_message
