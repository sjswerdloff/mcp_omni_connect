import asyncio
import json
import os
import platform
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.websocket import websocket_client
from mcp.types import (
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
)
from openai import OpenAI
from dataclasses import dataclass, field
from mcpomni_connect.refresh_server_capabilities import refresh_capabilities
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
        with open(config_path, "r", encoding="utf-8") as f:
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
        self.message_history = []
        self.debug = debug
        self.system_prompt = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_servers(self):
        """Connect to an MCP server"""
        server_config = self.config.load_config("servers_config.json")
        servers = [
            {"name": name, "srv_config": srv_config}
            for name, srv_config in server_config["mcpServers"].items()
        ]

        successful_connections = 0
        failed_connections = []

        logger.info(f"Attempting to connect to {len(servers)} servers")

        for server in servers:
            try:
                await self._connect_to_single_server(server)
                successful_connections += 1
                logger.info(
                    f"Successfully connected to server: {server.get('name', 'Unknown')}"
                )
            except Exception as e:
                failed_server = server.get("name", "Unknown")
                error_msg = (
                    f"Failed to connect to server {failed_server}: {str(e)}"
                )
                logger.error(error_msg)
                failed_connections.append((failed_server, str(e)))
                continue  # Continue with next server

        # Log summary of connections
        logger.info(
            f"MCP Servers Connection summary: {successful_connections} servers connected, {len(failed_connections)} servers failed to connect"
        )
        if failed_connections:
            logger.info("Failed connections:")
            for server, error in failed_connections:
                logger.info(f"  - {server}: {error}")

        if successful_connections == 0:
            raise RuntimeError(
                "No servers could be connected. All connection attempts failed."
            )

        return successful_connections

    def _validate_and_convert_url(self, url: str, connection_type: str) -> str:
        """Validate and convert URL based on connection type."""
        if connection_type == "sse":
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"Invalid SSE URL: {url}. Must start with http:// or https://"
                )
            return url
        elif connection_type == "websocket":
            if not url.startswith(("ws://", "wss://")):
                raise ValueError(
                    f"Invalid WebSocket URL: {url}. Must start with ws:// or wss://"
                )
            return url
        else:
            raise ValueError(
                f"Invalid connection type: {connection_type}. Must be sse or websocket"
            )

    async def _connect_to_single_server(self, server):
        try:
            connection_type = server["srv_config"].get("type", "stdio")
            logger.info(f"connection_type: {connection_type}")
            if connection_type == "sse":
                url = self._validate_and_convert_url(
                    server["srv_config"]["url"], "sse"
                )
                headers = server["srv_config"].get("headers", {})
                timeout = server["srv_config"].get("timeout", 5)
                sse_read_timeout = server["srv_config"].get(
                    "sse_read_timeout", 300
                )

                if self.debug:
                    logger.info(
                        f"SSE connection to {url} with timeout {timeout}"
                    )

                transport = await self.exit_stack.enter_async_context(
                    sse_client(
                        url,
                        headers=headers,
                        timeout=timeout,
                        sse_read_timeout=sse_read_timeout,
                    )
                )

            elif connection_type == "websocket":
                url = self._validate_and_convert_url(
                    server["srv_config"]["url"], "websocket"
                )
                logger.info(f"WebSocket connection to {url}")
                if self.debug:
                    logger.info(f"WebSocket connection to {url}")

                transport = await self.exit_stack.enter_async_context(
                    websocket_client(url)
                )

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

                transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )

            read_stream, write_stream = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            init_result = await session.initialize()
            server_name = init_result.serverInfo.name
            capabilities = init_result.capabilities
            self.server_names.append(server_name)

            self.sessions[server_name] = {
                "session": session,
                "read_stream": read_stream,
                "write_stream": write_stream,
                "connected": True,
                "capabilities": capabilities,
                "type": connection_type,
            }

            if self.debug:
                logger.info(
                    f"Successfully connected to {server_name} via {connection_type}"
                )
            # refresh capabilities to ensure we have the latest tools, resources, and prompts
            await refresh_capabilities(
                sessions=self.sessions,
                server_names=self.server_names,
                available_tools=self.available_tools,
                available_resources=self.available_resources,
                available_prompts=self.available_prompts,
                debug=self.debug,
            )
        except Exception as e:
            if self.debug:
                logger.error(f"Failed to connect to server: {str(e)}")
            raise

    async def clean_up_server(self):
        """Clean up server connections individually"""
        for server_name in list(self.server_names):
            try:
                if (
                    server_name in self.sessions
                    and self.sessions[server_name]["connected"]
                ):
                    session_info = self.sessions[server_name]
                    try:
                        if (
                            session_info["write_stream"]
                            and not session_info["write_stream"]._closed
                        ):
                            await session_info["write_stream"].aclose()
                            if self.debug:
                                logger.info(
                                    f"Closed write stream for {server_name}"
                                )

                        if (
                            session_info["read_stream"]
                            and not session_info["read_stream"]._closed
                        ):
                            await session_info["read_stream"].aclose()
                            if self.debug:
                                logger.info(
                                    f"Closed read stream for {server_name}"
                                )

                        if session_info["session"]:
                            close_method = getattr(
                                session_info["session"], "close", None
                            )
                            if close_method and callable(close_method):
                                await close_method()
                                if self.debug:
                                    logger.info(
                                        f"Closed session for {server_name}"
                                    )

                    except Exception as e:
                        logger.error(
                            f"Error during stream closure for {server_name}: {e}"
                        )

                    # Mark as disconnected and clear references
                    self.sessions[server_name]["connected"] = False
                    self.sessions[server_name]["session"] = None
                    self.sessions[server_name]["read_stream"] = None
                    self.sessions[server_name]["write_stream"] = None

                    if self.debug:
                        logger.info(f"Cleaned up server: {server_name}")
            except Exception as e:
                logger.error(f"Error cleaning up server {server_name}: {e}")

    async def cleanup(self):
        """Clean up all resources"""
        try:
            logger.info("Starting client shutdown...")

            # First make sure all servers are properly shut down
            try:
                async with asyncio.timeout(
                    10.0
                ):  # 10 second timeout for server cleanup
                    await self.clean_up_server()
            except asyncio.TimeoutError:
                logger.warning("Server cleanup timed out")
            except Exception as e:
                logger.error(f"Error during server cleanup: {e}")

            try:
                await self.clean_up_server()
                await self.exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error closing exit stack: {e}")

            # Clear any remaining data structures
            self.server_names.clear()
            self.sessions.clear()
            self.available_tools.clear()
            self.available_resources.clear()
            self.available_prompts.clear()

            logger.info("All resources cleared")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    # add a message to the message history
    async def add_message_to_history(
        self, role: str, content: str, metadata: Optional[dict] = None
    ):
        """Add a message to the message history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": asyncio.get_running_loop().time(),
            "metadata": metadata or {},
        }
        self.message_history.append(message)
        if self.debug:
            logger.info(f"Added message to history: {role} - {content[:100]}")

    async def show_history(self):
        """Show the message history"""
        for i, message in enumerate(self.message_history):
            logger.info(
                f"Message {i}: {message['role']} - {message['content']}"
            )

    async def clear_history(self):
        """Clear the message history"""
        self.message_history = []
        if self.debug:
            logger.info("Message history cleared")

    async def save_message_history_to_file(self, file_path: str):
        """Save the message history to a file"""
        with open(file_path, "w") as f:
            for message in self.message_history:
                f.write(f"{message['role']}: {message['content']}\n")
        if self.debug:
            logger.info(f"Message history saved to {file_path}")
