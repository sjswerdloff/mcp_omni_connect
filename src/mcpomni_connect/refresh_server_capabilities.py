from mcpomni_connect.utils import logger
from typing import Dict, Any, Callable, List
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
        if not sessions[server_name]["connected"]:
            raise ValueError("Not connected to a server")
        # list all tools
        try:
            tools_response = await sessions[server_name]["session"].list_tools()
            if tools_response:
                available_tools[server_name] = tools_response.tools
        except Exception as e:
            logger.info(f"{server_name} Does not support tools")
            available_tools[server_name] = []
        # list all resources
        try:
            resources_response = await sessions[server_name]["session"].list_resources()
            if resources_response:
                available_resources[server_name] = resources_response.resources
        except Exception as e:
            logger.info(f"{server_name} Does not support resources")
            available_resources[server_name] = []
        # list all prompts
        try:
            prompts_response = await sessions[server_name]["session"].list_prompts()
            if prompts_response:
                available_prompts[server_name] = prompts_response.prompts
        except Exception as e:
            logger.info(f"{server_name} Does not support prompts")
            available_prompts[server_name] = []
    
    if debug:
        logger.info(f"Refreshed capabilities for {server_names}")
        # Create a clean dictionary of server names and their tool names
        tools_by_server = {
            server_name: [tool.name for tool in tools]
            for server_name, tools in available_tools.items()
        }
        logger.info("Available tools by server:")
        for server_name, tool_names in tools_by_server.items():
            logger.info(f"  {server_name}:")
            for tool_name in tool_names:
                logger.info(f"    - {tool_name}")
        # Create a clean dictionary of server names and their resource names
        resources_by_server = {
            server_name: [resource.name for resource in resources]
            for server_name, resources in available_resources.items()
        }
        logger.info("Available resources by server:")
        for server_name, resource_names in resources_by_server.items():
            logger.info(f"  {server_name}:")
            for resource_name in resource_names:
                logger.info(f"    - {resource_name}")
        # Create a clean dictionary of server names and their prompt names
        prompts_by_server = {
            server_name: [prompt.name for prompt in prompts]
            for server_name, prompts in available_prompts.items()
        }
        logger.info("Available prompts by server:")
        for server_name, prompt_names in prompts_by_server.items():
            logger.info(f"  {server_name}:")
            for prompt_name in prompt_names:
                logger.info(f"    - {prompt_name}")
    
    # Update system prompt with new capabilities
    #self.system_prompt = self.generate_system_prompt()
    
    if debug:
        logger.info("Updated system prompt with new capabilities")
        #logger.info(f"System prompt:\n{system_prompt}")
