# from typing import Any, Callable, Dict, List

# from mcpomni_connect.utils import logger


# async def refresh_capabilities(
#     sessions: Dict[str, Any],
#     server_names: List[str],
#     available_tools: Dict[str, Any],
#     available_resources: Dict[str, Any],
#     available_prompts: Dict[str, Any],
#     debug: bool,
# ) -> None:
#     """Refresh the capabilities of the server and update system prompt"""
#     for server_name in server_names:
#         if not sessions[server_name]["connected"]:
#             raise ValueError("Not connected to a server")
#         # list all tools
#         try:
#             tools_response = await sessions[server_name][
#                 "session"
#             ].list_tools()
#             if tools_response:
#                 available_tools[server_name] = tools_response.tools
#         except Exception as e:
#             logger.info(f"{server_name} Does not support tools")
#             available_tools[server_name] = []
#         # list all resources
#         try:
#             resources_response = await sessions[server_name][
#                 "session"
#             ].list_resources()
#             if resources_response:
#                 available_resources[server_name] = resources_response.resources
#         except Exception as e:
#             logger.info(f"{server_name} Does not support resources")
#             available_resources[server_name] = []
#         # list all prompts
#         try:
#             prompts_response = await sessions[server_name][
#                 "session"
#             ].list_prompts()
#             if prompts_response:
#                 available_prompts[server_name] = prompts_response.prompts
#         except Exception as e:
#             logger.info(f"{server_name} Does not support prompts")
#             available_prompts[server_name] = []

#     if debug:
#         logger.info(f"Refreshed capabilities for {server_names}")
#         # Create a clean dictionary of server names and their tool names
#         tools_by_server = {
#             server_name: [tool.name for tool in tools]
#             for server_name, tools in available_tools.items()
#         }
#         logger.info("Available tools by server:")
#         for server_name, tool_names in tools_by_server.items():
#             logger.info(f"  {server_name}:")
#             for tool_name in tool_names:
#                 logger.info(f"    - {tool_name}")
#         # Create a clean dictionary of server names and their resource names
#         resources_by_server = {
#             server_name: [resource.name for resource in resources]
#             for server_name, resources in available_resources.items()
#         }
#         logger.info("Available resources by server:")
#         for server_name, resource_names in resources_by_server.items():
#             logger.info(f"  {server_name}:")
#             for resource_name in resource_names:
#                 logger.info(f"    - {resource_name}")
#         # Create a clean dictionary of server names and their prompt names
#         prompts_by_server = {
#             server_name: [prompt.name for prompt in prompts]
#             for server_name, prompts in available_prompts.items()
#         }
#         logger.info("Available prompts by server:")
#         for server_name, prompt_names in prompts_by_server.items():
#             logger.info(f"  {server_name}:")
#             for prompt_name in prompt_names:
#                 logger.info(f"    - {prompt_name}")

#     # Update system prompt with new capabilities
#     # self.system_prompt = self.generate_system_prompt()

#     if debug:
#         logger.info("Updated system prompt with new capabilities")
#         # logger.info(f"System prompt:\n{system_prompt}")


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