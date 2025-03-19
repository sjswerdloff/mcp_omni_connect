# MCP Connect Client CLI

MCP Connect is a versatile command-line interface (CLI) client designed to connect to various Model Context Protocol (MCP) servers using stdio transport. It provides seamless integration with OpenAI models and supports dynamic tool and resource management across multiple servers.

## Features

- **Stdio Transport**: Connects to MCP servers using efficient stdio transport.
- **OpenAI Integration**: Leverages OpenAI models for advanced processing.
- **Multi-Server Support**: Connects to multiple MCP servers simultaneously.
- **Dynamic Tool Discovery**: Automatically discovers and lists available tools.
- **Resource Management**: Accesses and manages resources across servers.

## Architecture Overview

### Client-Side Components

- **Command Parser**: Interprets user commands and routes them to the appropriate handlers.
- **Tool Manager**: Manages tool execution requests and interactions with servers.
- **Resource Manager**: Handles resource access requests.
- **OpenAI Integration**: Processes data using OpenAI models for enhanced capabilities.
- **Response Formatter**: Formats responses for user-friendly output.

### Server-Side Components

- **MCP Servers**: Each server provides specific tools and resources, accessible via stdio transport.

## Getting Started

### Prerequisites

- Python 3.12 or later
- Virtual environment setup
- OpenAI API Key

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/mcp-connect.git
   cd mcp-connect
   ```

2. **Set Up Virtual Environment**:
   ```bash
   uv venv or python -m venv .venv
   source .venv/bin/activate
   ```

3. **Sync Dependencies with `uv`**:
   ```bash
   uv sync
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory with the following content:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Update the `servers_config.json`**:
   Ensure your `servers_config.json` is correctly configured to define the MCP servers you wish to connect to and the LLM configuration. Here is an example configuration:
   ```json
   {
       "LLM": {
           "model": "gpt-4o-mini",
           "temperature": 0.5,
           "max_tokens": 1000,
           "top_p": 0,
           "frequency_penalty": 0,
           "presence_penalty": 0
       },
       "mcpServers": {
           "server-name": {
               "command": "python",
               "args": ["mcp-server.py"],
               "env": {
                   "API_KEY": "value"
               }
           }
       }
   }
   ```

### Running the Client

1. **Start the MCP Servers**:
   Use `uv` to run your MCP servers. For example:
   ```bash
   uv run main.py
   ```

2. **Run the MCP Connect Client**:
   ```bash
   python mcp_client.py
   ```

### Usage

- **List Available Tools**: `/tools`
- **List Available Resources**: `/resources`
- **Read a Resource**: `/resource:<uri>`
- **Toggle Debug Mode**: `/debug`
- **Refresh Capabilities**: `/refresh`
- **Exit the Application**: `quit`

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact [abioladedayo1993@gmail.com] or open an issue on GitHub.
