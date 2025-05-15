import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any
import anyio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from mcpomni_connect.llm import LLMConnection
from mcpomni_connect.notifications import handle_notifications
from mcpomni_connect.refresh_server_capabilities import refresh_capabilities
from mcpomni_connect.sampling import samplingCallback
from mcpomni_connect.system_prompts import generate_react_agent_role_prompt
from mcpomni_connect.utils import logger


@dataclass
class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    llm_api_key: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.llm_api_key = os.getenv("LLM_API_KEY")

        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    def load_config(self, file_path: str) -> dict:
        """Load server configuration from JSON file."""
        config_path = Path(file_path)
        logger.info(f"Loading configuration from: {config_path.name}")
        if config_path.name.lower() != "servers_config.json":
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}, it should be 'servers_config.json'"
            )
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)


class MCPClient:
    def __init__(self, config: dict[str, Any], debug: bool = False):
        # Initialize session and client objects
        self.config = config
        self.sessions = {}
        self._cleanup_lock = asyncio.Lock()
        self.available_tools = {}
        self.available_resources = {}
        self.available_prompts = {}
        self.server_names = []
        self.added_servers_names = {}  # this to map the name used in the config and the actual server name gotten after initialization
        self.debug = debug
        self.system_prompt = None
        self.llm_connection = LLMConnection(self.config)
        self.sampling_callback = samplingCallback()
        self.tasks = {}

    async def connect_to_servers(self):
        """Connect to an MCP server"""
        server_config = self.config.load_config("servers_config.json")
        servers = [
            {"name": name, "srv_config": srv_config}
            for name, srv_config in server_config["mcpServers"].items()
        ]
        try:
            async with anyio.create_task_group() as tg:
                for server in servers:
                    server_added_name = server["name"]
                    tg.start_soon(
                        self._connect_to_single_server, server, server_added_name
                    )
        except Exception as e:
            logger.info(f"start servers task error: {e}")

        # start the notification stream with an asyncio task
        asyncio.create_task(
            handle_notifications(
                sessions=self.sessions,
                debug=self.debug,
                server_names=self.server_names,
                available_tools=self.available_tools,
                available_resources=self.available_resources,
                available_prompts=self.available_prompts,
                refresh_capabilities=refresh_capabilities,
                llm_connection=self.llm_connection,
                generate_react_agent_role_prompt=generate_react_agent_role_prompt,
            )
        )

    async def _connect_to_single_server(self, server, server_added_name):
        try:
            # create AsyncExitStack per mcp server to ensure we can remove it safely without cancelling all tasks
            stack = AsyncExitStack()
            transport_type = server["srv_config"].get("transport_type", "stdio")
            read_stream = None
            write_stream = None
            url = server["srv_config"].get("url", "")
            headers = server["srv_config"].get("headers", {})
            timeout = server["srv_config"].get("timeout", 60)
            sse_read_timeout = server["srv_config"].get("sse_read_timeout", 120)
            if transport_type.lower() == "sse":
                if self.debug:
                    logger.info(f"SSE connection to {url} with timeout {timeout}")
                transport = await stack.enter_async_context(
                    sse_client(
                        url=url,
                        headers=headers,
                        timeout=timeout,
                        sse_read_timeout=sse_read_timeout,
                    )
                )
                read_stream, write_stream = transport
            elif transport_type.lower() == "streamable_http":
                if self.debug:
                    logger.info(
                        f"Streamable HTTP connection to {url} with timeout {timeout}"
                    )
                timeout = timedelta(seconds=int(timeout))
                sse_read_timeout = timedelta(seconds=int(sse_read_timeout))
                transport = await stack.enter_async_context(
                    streamablehttp_client(
                        url=url,
                        headers=headers,
                        timeout=timeout,
                        sse_read_timeout=sse_read_timeout,
                    )
                )
                read_stream, write_stream, _ = transport
            else:
                # stdio connection (default)
                args = server["srv_config"]["args"]
                command = server["srv_config"]["command"]
                env = (
                    {**os.environ, **server["srv_config"]["env"]}
                    if server["srv_config"].get("env")
                    else None
                )
                server_params = StdioServerParameters(
                    command=command, args=args, env=env
                )
                transport = await stack.enter_async_context(stdio_client(server_params))

                read_stream, write_stream = transport

            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    sampling_callback=self.sampling_callback._sampling,
                    read_timeout_seconds=timedelta(seconds=300),  # 5 minutes timeout
                )
            )
            init_result = await session.initialize()
            server_name = init_result.serverInfo.name
            capabilities = init_result.capabilities
            if server_name in self.server_names:
                error_message = (
                    f"{server_name} is already connected. disconnect it and try again"
                )
                if self.debug:
                    logger.error(error_message)
                await stack.aclose()
                return error_message
            self.server_names.append(server_name)
            server_name_data = {server_added_name: server_name}
            self.added_servers_names.update(server_name_data)
            self.sessions[server_name] = {
                "session": session,
                "read_stream": read_stream,
                "write_stream": write_stream,
                "connected": True,
                "capabilities": capabilities,
                "transport_type": transport_type,
                "stack": stack,
            }
            if self.debug:
                logger.info(
                    f"Successfully connected to {server_name} via {transport_type}"
                )
            # refresh capabilities to ensure we have the latest tools, resources, and prompts
            await refresh_capabilities(
                sessions=self.sessions,
                server_names=self.server_names,
                available_tools=self.available_tools,
                available_resources=self.available_resources,
                available_prompts=self.available_prompts,
                debug=self.debug,
                server_name=server_name,
                llm_connection=self.llm_connection,
                generate_react_agent_role_prompt=generate_react_agent_role_prompt,
            )

            return f"{server_name} connected succesfully"
        except Exception as e:
            error_message = f"Failed to connect to server: {str(e)}"
            logger.error(error_message)
            return error_message

    async def add_servers(self, config_file: Path) -> None:
        """Dynamically add servers at runtime."""
        with open(config_file, "r") as f:
            server_config = json.load(f)

        servers = [
            {"name": name, "srv_config": srv_config}
            for name, srv_config in server_config["mcpServers"].items()
        ]
        errors = []
        servers_connected_response = []
        try:
            server_added_name = None
            async with anyio.create_task_group() as tg:
                for server in servers:
                    server_added_name = server["name"]
                    tg.start_soon(
                        self._connect_to_single_server, server, server_added_name
                    )
                    servers_connected_response.append(
                        f"{server_added_name} connected succesfully"
                    )
        except Exception as e:
            logger.error(f"Failed to add server '{server_added_name}': {e}")
            errors.append((server_added_name, str(e)))
        if errors:
            return errors
        return servers_connected_response

    async def remove_server(self, name: str) -> None:
        """Disconnect and remove a server by name."""
        try:
            old_name = name
            if name not in self.added_servers_names.keys():
                raise ValueError(f"Server '{name}' not found.")
            if len(self.sessions) == 1:
                return (
                    f"Cannot remove {name}: at least one server must remain connected."
                )
            for server_added_name, server_name in self.added_servers_names.items():
                if name.lower() == server_added_name.lower():
                    name = server_name
            session_info = self.sessions[name]
            await self._close_session_resources(
                server_name=old_name, session_info=session_info
            )
        except ValueError as e:
            error_message = f"Error removing server: {str(e)}"
            logger.error(error_message)
            return error_message
        except Exception as e:
            error_message = f"Error cleaning up server '{name}': {e}"
            logger.error(error_message)
            return error_message

        self.sessions.pop(name, None)
        self.server_names.remove(name)
        self.added_servers_names = {
            k: v for k, v in self.added_servers_names.items() if v != name
        }
        self.available_tools.pop(name, None)
        self.available_resources.pop(name, None)
        self.available_prompts.pop(name, None)

        return f"{name} diconnected succesfully"

        logger.info(f"Server '{name}' removed successfully.")

    async def _close_session_resources(self, server_name: str, session_info: dict):
        """Tear down the per-server context stack, which closes streams and session."""

        stack: AsyncExitStack = session_info.get("stack")
        if not stack:
            logger.warning(f"No context stack found for {server_name}")
            return
        try:
            logger.info(f"Closing context stack for {server_name}")
            # Ensure transport and session resources are cleaned up
            await stack.aclose()
            logger.info(f"Server {server_name} has been disconnected and removed.")
        except RuntimeError as e:
            if "cancel scope" in str(e).lower():
                logger.warning(
                    f"Cancel scope error during disconnect from {server_name}, Ignored context task mismatch"
                )
            else:
                raise e
        except Exception as e:
            logger.error(f"Error closing context stack for {server_name}: {e}")
            return e

    async def clean_up_server(self):
        """Clean up server connections individually"""
        for server_name in list(self.server_names):
            try:
                if (
                    server_name in self.sessions
                    and self.sessions[server_name]["connected"]
                ):
                    session_info = self.sessions[server_name]
                    await self._close_session_resources(server_name, session_info)

                    if self.debug:
                        logger.info(f"Cleaned up server: {server_name}")

            except Exception as e:
                logger.error(f"Error cleaning up server {server_name}: {e}")

    async def cleanup(self):
        """Clean up all resources"""
        try:
            logger.info("Starting client shutdown...")
            try:
                async with asyncio.timeout(
                    60.0
                ):  # 60 second timeout for server cleanup
                    await self.clean_up_server()
            except asyncio.TimeoutError:
                logger.warning("Server cleanup timed out")
            except Exception as e:
                logger.error(f"Error during server cleanup: {e}")

            # Clear any remaining data structures
            self.server_names.clear()
            self.added_servers_names.clear()
            self.sessions.clear()
            self.available_tools.clear()
            self.available_resources.clear()
            self.available_prompts.clear()

            logger.info("All resources cleared")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
