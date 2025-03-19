import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from openai import OpenAI
from utils import logger
import sys
from dotenv import load_dotenv
import json


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def openai_api_key(self) -> str:
        """Get the OpenAI API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return self.api_key

class MCPClient:
    def __init__(self, config: dict[str, Any], debug: bool = False):
        # Initialize session and client objects
        self.config = config
        self.sessions = {}
        self._cleanup_lock = asyncio.Lock()
        self.llm_config = None
        self.openai = OpenAI(api_key=self.config.openai_api_key)
        self.available_tools = {}
        self.available_resources = {}
        self.server_names = []
        self.message_history = []
        self.debug = debug
        self.system_prompt = None
        self.exit_stack = AsyncExitStack()
        
    async def llm_configs(self):
        """Load the LLM configuration"""
        llm_config = self.config.load_config("servers_config.json")["LLM"]
        try:
            model = llm_config.get("model", "gpt-4o-mini")
            temperature = llm_config.get("temperature", 0.5)
            max_tokens = llm_config.get("max_tokens", 5000)
            top_p = llm_config.get("top_p", 0)
            self.llm_config = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            return self.llm_config
        except Exception as e:
            logger.error(f"Error loading LLM configuration: {e}")
            return None

    async def connect_to_servers(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py)
        """
        server_config = self.config.load_config("servers_config.json")
        servers = [{"name": name, "srv_config": srv_config} for name, srv_config in server_config["mcpServers"].items()]
        # create tasks for connecting to each server
        # issue using for loops works and exist_stack closed but using asyncio.gather does not work
        for server in servers:
            await self._connect_to_single_server(server)
        # connections_tasks = []
        # for server in servers:
        #     task = asyncio.create_task(self._connect_to_single_server(server))
        #     connections_tasks.append(task)
        # # wait for all connections to complete
        # await asyncio.gather(*connections_tasks)
        
    async def _connect_to_single_server(self, server):
        try:
            # initialize server parameters
            args = server["srv_config"]["args"]
            command = server["srv_config"]["command"]
            env = {**os.environ, **server["srv_config"]["env"]} if server["srv_config"].get("env") else None
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )

            # initialize stdio transport
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

            # initialize the session
            init_result = await session.initialize()
            server_name = init_result.serverInfo.name
            capabilities = init_result.capabilities
            self.server_names.append(server_name)
            # store the session
            self.sessions[server_name] = {
                "session": session,
                "stdio": stdio,
                "write": write,
                "connected": True,
                "capabilities": capabilities
            }
            if self.debug:
                logger.info(f"Connected to server: {server_name} v{init_result.serverInfo.version}")
            # refresh capabilities and cache available tools
            await self.refresh_capabilities()
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
           
    async def clean_up_server(self):
        """Clean up server connections individually"""
        for server_name in list(self.server_names):
            async with self._cleanup_lock:
                if server_name in self.sessions and self.sessions[server_name]["connected"]:
                    await self.exit_stack.aclose()
                    # Mark as disconnected and clear references
                    self.sessions[server_name]["connected"] = False
                    self.sessions[server_name]["session"] = None
                    self.sessions[server_name]["stdio"] = None
                    self.sessions[server_name]["write"] = None
            
    

    async def cleanup(self):
        """Clean up all resources"""
        try:
            # First make sure all servers are properly shut down
            await self.clean_up_server()
            # Clear any remaining data structures
            self.server_names = []
            self.available_tools = {}
            self.available_resources = {}
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            logger.info("Client shutdown complete")

    async def refresh_capabilities(self):
        """Refresh the capabilities of the server and update system prompt"""
        for server_name in self.server_names:
            if not self.sessions[server_name]["connected"]:
                raise ValueError("Not connected to a server")
            # list all tools
            tools_response = await self.sessions[server_name]["session"].list_tools()
            available_tools = tools_response.tools
            self.available_tools[server_name] = available_tools
            # list all resources
            resources_response = await self.sessions[server_name]["session"].list_resources()
            available_resources = resources_response.resources
            self.available_resources[server_name] = available_resources
        
        if self.debug:
            logger.info(f"Refreshed capabilities for {self.server_names}")
            # Create a clean dictionary of server names and their tool names
            tools_by_server = {
                server_name: [tool.name for tool in tools]
                for server_name, tools in self.available_tools.items()
            }
            logger.info("Available tools by server:")
            for server_name, tool_names in tools_by_server.items():
                logger.info(f"  {server_name}:")
                for tool_name in tool_names:
                    logger.info(f"    - {tool_name}")
            # Create a clean dictionary of server names and their resource names
            resources_by_server = {
                server_name: [resource.name for resource in resources]
                for server_name, resources in self.available_resources.items()
            }
            logger.info("Available resources by server:")
            for server_name, resource_names in resources_by_server.items():
                logger.info(f"  {server_name}:")
                for resource_name in resource_names:
                    logger.info(f"    - {resource_name}")
            
        # Update system prompt with new capabilities
        self.system_prompt = self.generate_system_prompt()
        # update llm config
        await self.llm_configs()
        if self.debug:
            logger.info("Updated system prompt with new capabilities")
            logger.info(f"System prompt:\n{self.system_prompt}")

    # list all resources in mcp server
    async def list_resources(self):
        """List all resources"""
        resources = []
        for server_name in self.server_names:
            if self.sessions[server_name]["connected"]:
                resources_response = await self.sessions[server_name]["session"].list_resources()
                resources.extend(resources_response.resources)
        return resources
    
    async def find_resource_server(self, uri: str) -> tuple[str, bool]:
        """Find which server has the resource
        
        Returns:
            tuple[str, bool]: (server_name, found)
        """
        for server_name, resources in self.available_resources.items():
            resource_uris = [str(res.uri) for res in resources]
            if uri in resource_uris:
                return server_name, True
        return "", False
    
    # read a resource from mcp server
    async def read_resource(self, uri: str):
        """Read a resource"""
        if self.debug:
            logger.info(f"Reading resource: {uri}")
        # add the first message to the history
        await self.add_message_to_history("user", f"Reading resource: {uri}")
        server_name, found = await self.find_resource_server(uri)
        if not found:
            error_message = f"Resource not found: {uri}"
            logger.error(error_message)
            await self.add_message_to_history("user", error_message, {"resource_uri": uri, "error": True})
            return error_message
        logger.info(f"Resource found in {server_name}")
        try:
            resource_response = await self.sessions[server_name]["session"].read_resource(uri)
            if self.debug:
                logger.info("LLM processing resource")
            llm_response = self.openai.chat.completions.create(
                model=self.llm_config["model"],
                max_tokens=self.llm_config["max_tokens"],
                temperature=self.llm_config["temperature"],
                top_p=self.llm_config["top_p"],
                messages=[
                    {
                    "role": "system", 
                    "content": "Analyze the document content and provide a clear, concise summary that captures all essential information. Focus on key points, main concepts, and critical details that give the user a complete understanding without reading the entire document. Present your summary using bullet points for main ideas followed by a brief paragraph for context when needed. Include any technical terms, specifications, instructions, or warnings that are vital to proper understanding. Do not include phrases like 'here is your summary' or 'in summary' - deliver only the informative content directly."
                },
                {"role": "user", "content": str(resource_response)}
            ]
        )
            response_content = llm_response.choices[0].message.content or ""
            if not response_content:
                response_content = "No content found for resource"
            await self.add_message_to_history("assistant", response_content)
            return response_content
        except Exception as e:
            error_message = f"Error reading resource: {e}"
            logger.error(error_message)
            await self.add_message_to_history("user", error_message, {"resource_uri": uri, "error": True})
            return error_message
    
    async def list_tools(self):
        """List all tools"""
        tools = []
        for server_name in self.server_names:
            if self.sessions[server_name]["connected"]:
                tools_response = await self.sessions[server_name]["session"].list_tools()
                tools.extend(tools_response.tools)
        return tools
    
    
    # add a message to the message history
    async def add_message_to_history(self, role: str, content: str, metadata: Optional[dict] = None):
        """Add a message to the message history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": asyncio.get_running_loop().time(),
            "metadata": metadata or {}
        }
        self.message_history.append(message)
        if self.debug:
            logger.info(f"Added message to history: {role} - {content[:100]}")

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        # Ensure system prompt is updated
        if not self.system_prompt:
            self.system_prompt = self.generate_system_prompt()
            

        # add user query to history
        await self.add_message_to_history("user", query)

        # prepare messages for OpenAI
        messages = []

        # add system prompt and user query to messages
        messages.append({
                "role": "system",
                "content": self.system_prompt
        })

        # track assistant with tool calls and pending tool responses
        assistant_with_tool_calls = None
        pending_tool_responses = []
        
        # process message history in order
        for _, message in enumerate(self.message_history):
            if message["role"] == "user":
                # First flush any pending tool responses if needed
                if assistant_with_tool_calls and pending_tool_responses:
                    messages.append(assistant_with_tool_calls)
                    messages.extend(pending_tool_responses)
                    assistant_with_tool_calls = None
                    pending_tool_responses = []

                # then add user message to messages that will be sent to OpenAI
                messages.append({
                "role": "user",
                    "content": message["content"]
                })

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
                        "tool_calls": metadata.get("tool_calls", [])
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
                    messages.append({
                        "role": "assistant",
                        "content": message["content"]
                    })
            elif message["role"] == "tool" and "tool_call_id" in message.get("metadata", {}):
                # Collect tool responses
                # Only add if we have a preceding assistant message with tool calls
                if assistant_with_tool_calls:
                    pending_tool_responses.append({
                        "role": "tool",
                        "content": message["content"],
                        "tool_call_id": message["metadata"]["tool_call_id"]
                    })
            
            elif message["role"] == "system":
                # add system message to messages
                messages.append({
                    "role": "system",
                    "content": message["content"]
                })
        
        # Flush any remaining pending tool calls at the end
        if assistant_with_tool_calls:
            messages.append(assistant_with_tool_calls)
            messages.extend(pending_tool_responses)

        
        if self.debug:
            logger.info(f"Prepared {len(messages)} messages for OpenAI")
            for i, message in enumerate(messages):
                role = message['role']
                has_tool_calls = 'tool_calls' in message
                preview = message['content'][:50] + "..." if message['content'] else ""
                logger.info(f"Message {i}: {role} {'with tool_calls' if has_tool_calls else ''} - {preview}")

        # make sure we have the latest tools
        if not self.available_tools:
            await self.refresh_capabilities()

        # list available tools
        tools = await self.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
            "name": tool.name,
            "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in tools]

        if self.debug:
            tool_names = [tool["function"]["name"] for tool in available_tools]
            logger.info(f"Available tools for query: {tool_names}")
            logger.info(f"Sending {len(messages)} messages to OpenAI")

        try:
            # Initial OpenAI API call
            response = self.openai.chat.completions.create(
                model=self.llm_config["model"],
                max_tokens=self.llm_config["max_tokens"],
                temperature=self.llm_config["temperature"],
                top_p=self.llm_config["top_p"],
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
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
                "tool_calls": assistant_message.tool_calls
            }
            if self.debug:
                logger.info(f"Processing {len(assistant_message.tool_calls)} tool calls")
                
            # Properly append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": initial_response,  # This should be a string or null
                "tool_calls": assistant_message.tool_calls
            })
        # add the assistant message to history with tool calls metadata
        await self.add_message_to_history("assistant", initial_response, tool_calls_metadata)
        final_text.append(initial_response)
        if assistant_message.tool_calls:
            if self.debug:
                logger.info(f"Processing {len(assistant_message.tool_calls)} tool calls")
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                # execute tool call
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse tool arguments for {tool_name}: {tool_args}")
                        tool_args = {}
                if self.debug:
                    logger.info(f"Processing tool call: {tool_name} with args {tool_args}")
                # execute tool call on the server
                try:
                    for server_name, tools in self.available_tools.items():
                        tool_names = [tool.name for tool in tools]
                        if tool_name in tool_names:
                            result = await self.sessions[server_name]["session"].call_tool(tool_name, tool_args)
                            tool_content = result.content if hasattr(result, 'content') else str(result)
                            break
                    
                    # Handle the result content appropriately
                    if hasattr(tool_content, '__getitem__') and len(tool_content) > 0 and hasattr(tool_content[0], 'text'):
                        tool_content = tool_content[0].text
                    else:
                        tool_content = tool_content
                    tool_results.append(
                        {
                            "call": tool_name,
                            "result": tool_content
                        }
                    )
                    if self.debug:
                        result_preview = tool_content[:200] + "..." if len(str(tool_content)) > 200 else str(tool_content)
                        logger.info(f"Tool result preview: {result_preview}")
                    
                    # add the tool result to the messages
                    messages.append({
                        "role": "tool",
                        "content": str(tool_content),  # Ensure content is a string
                        "tool_call_id": tool_call.id
                    })
                    # add message to history
                    await self.add_message_to_history("tool", str(tool_content), {
                        "tool_call_id": tool_call.id,
                        "tool": tool_name,
                        "args": tool_args
                    })
                except Exception as e:
                    error_message = f"Error executing tool call {tool_name}: {e}"
                    logger.error(error_message)
                    # append the message regardless of error
                    messages.append({
                        "role": "tool",
                        "content": error_message,
                        "tool_call_id": tool_call.id
                    })
                    # add error message to history
                    await self.add_message_to_history("tool", error_message, {
                        "tool_call_id": tool_call.id,
                        "tool": tool_name,
                        "args": tool_args,
                        "error": True
                    })
                    final_text.append(f"\n[Error executing tool call {tool_name}: {error_message}]")
            if self.debug:
                logger.info("Getting final response from OpenAi with tool results")
            
            try:
                second_response = self.openai.chat.completions.create(
                    model=self.llm_config["model"],
                    max_tokens=self.llm_config["max_tokens"],
                    temperature=self.llm_config["temperature"],
                    top_p=self.llm_config["top_p"],
                    messages=messages
                )
                final_assistant_message = second_response.choices[0].message
                response_content = final_assistant_message.content or ""
                await self.add_message_to_history("assistant", response_content)
                final_text.append(response_content)
            except Exception as e:
                error_message = f"Error getting final response from OpenAI: {e}"
                logger.error(error_message)
                await self.add_message_to_history("assistant", error_message, {"error": True})
                final_text.append(f"\n[Error getting final response from OpenAI: {e}]")

        return "\n".join(final_text)

    def generate_system_prompt(self) -> str:
        """Generate a dynamic system prompt based on available tools and capabilities"""
        
        # Base prompt that's server-agnostic
        base_prompt = """You are an intelligent assistant with access to various tools and resources through the Model Context Protocol (MCP).

Your capabilities:
1. You can understand and process user queries
2. You can use available tools to fetch information and perform actions
3. You can access and summarize resources when needed

Guidelines:
1. Always verify tool availability before attempting to use them
2. Ask clarifying questions if the user's request is unclear
3. Explain your thought process before using any tools
4. If a requested capability isn't available, explain what's possible with current tools
5. Provide clear, concise responses focusing on the user's needs

Available Tools by Server:
"""

        # Add available tools dynamically
        tools_section = []
        for server_name, tools in self.available_tools.items():
            tools_section.append(f"\n[{server_name}]")
            for tool in tools:
                tool_desc = f"• {tool.name}: {tool.description}"
                # Add parameters if they exist
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    params = tool.inputSchema.get('properties', {})
                    if params:
                        tool_desc += "\n  Parameters:"
                        for param_name, param_info in params.items():
                            param_desc = param_info.get('description', 'No description')
                            param_type = param_info.get('type', 'any')
                            tool_desc += f"\n    - {param_name} ({param_type}): {param_desc}"
                tools_section.append(tool_desc)

        interaction_guidelines = """
Before using any tool:
1. Analyze the user's request carefully
2. Check if the required tool is available in the current toolset
3. If unclear about the request or tool choice:
   - Ask for clarification from the user
   - Explain what information you need
   - Suggest available alternatives if applicable

When using tools:
1. Explain which tool you're going to use and why
2. Verify all required parameters are available
3. Handle errors gracefully and inform the user
4. Provide context for the results

Remember:
- Only use tools that are listed above
- Don't assume capabilities that aren't explicitly listed
- Be transparent about limitations
- Maintain a helpful and professional tone
"""

        # Combine all sections
        full_prompt = (
            base_prompt +
            "\n".join(tools_section) +
            interaction_guidelines
        )

        return full_prompt
