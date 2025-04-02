from typing import Any, Dict, List

from mcpomni_connect.utils import logger

async def refresh_capabilities(
    sessions: Dict[str, Any],
    server_names: List[str],
    available_tools: Dict[str, Any],
    available_resources: Dict[str, Any],
    available_prompts: Dict[str, Any],
    debug: bool,
) -> None:
    """Refresh the capabilities of the server and update system prompt"""
    for server_name in server_names:
        if not sessions.get(server_name, {}).get("connected", False):
            raise ValueError(f"Not connected to server: {server_name}")
        
        session = sessions[server_name].get("session")
        if not session:
            logger.warning(f"No session found for server: {server_name}")
            continue

        # List all tools
        try:
            tools_response = await session.list_tools()
            available_tools[server_name] = tools_response.tools if tools_response else []
        except Exception:
            logger.info(f"{server_name} does not support tools")
            available_tools[server_name] = []

        # List all resources
        try:
            resources_response = await session.list_resources()
            available_resources[server_name] = resources_response.resources if resources_response else []
        except Exception:
            logger.info(f"{server_name} does not support resources")
            available_resources[server_name] = []

        # List all prompts
        try:
            prompts_response = await session.list_prompts()
            available_prompts[server_name] = prompts_response.prompts if prompts_response else []
        except Exception:
            logger.info(f"{server_name} does not support prompts")
            available_prompts[server_name] = []

    if debug:
        logger.info(f"Refreshed capabilities for {server_names}")
        
        for category, data in {
            "Tools": available_tools,
            "Resources": available_resources,
            "Prompts": available_prompts,
        }.items():
            logger.info(f"Available {category.lower()} by server:")
            for server_name, items in data.items():
                logger.info(f"  {server_name}:")
                for item in items:
                    logger.info(f"    - {item.name}")

    if debug:
        logger.info("Updated system prompt with new capabilities")