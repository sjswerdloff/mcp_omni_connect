# 🚀 MCPOmni Connect - Universal Gateway to MCP Servers

MCPOmni Connect is a powerful, universal command-line interface (CLI) that serves as your gateway to the Model Context Protocol (MCP) ecosystem. It seamlessly integrates multiple MCP servers, AI models, and various transport protocols into a unified, intelligent interface.

## ✨ Key Features

### 🔌 Universal Connectivity
- **Multi-Protocol Support**
  - Native support for stdio transport
  - Server-Sent Events (SSE) for real-time communication
  - Docker container integration
  - NPX package execution
  - Extensible transport layer for future protocols

### 🧠 AI-Powered Intelligence
- **Advanced LLM Integration**
  - Seamless OpenAI model integration
  - Dynamic system prompts based on available capabilities
  - Intelligent context management
  - Automatic tool selection and chaining

### 💬 Prompt Management
- **Advanced Prompt Handling**
  - Dynamic prompt discovery across servers
  - Flexible argument parsing (JSON and key-value formats)
  - Cross-server prompt coordination
  - Intelligent prompt validation
  - Context-aware prompt execution
  - Real-time prompt responses
  - Support for complex nested arguments
  - Automatic type conversion and validation

### 🛠️ Tool Orchestration
- **Dynamic Tool Discovery & Management**
  - Automatic tool capability detection
  - Cross-server tool coordination
  - Intelligent tool selection based on context
  - Real-time tool availability updates

### 📦 Resource Management
- **Universal Resource Access**
  - Cross-server resource discovery
  - Unified resource addressing
  - Automatic resource type detection
  - Smart content summarization

### 🔄 Server Management
- **Advanced Server Handling**
  - Multiple simultaneous server connections
  - Automatic server health monitoring
  - Graceful connection management
  - Dynamic capability updates

## 🏗️ Architecture

### Core Components
```
MCPOmni Connect
├── Transport Layer
│   ├── Stdio Transport
│   ├── SSE Transport
│   └── Docker Integration
├── Session Management
│   ├── Multi-Server Orchestration
│   └── Connection Lifecycle Management
├── Tool Management
│   ├── Dynamic Tool Discovery
│   ├── Cross-Server Tool Routing
│   └── Tool Execution Engine
└── AI Integration
    ├── LLM Processing
    ├── Context Management
    └── Response Generation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- OpenAI API key
- UV package manager (recommended)

### Quick Start

1. **Installation**
   ```bash
   # Clone the repository
   git clone https://github.com/Abiorh001/mcp_connect.git
   cd mcp_connect

   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate

   # Install dependencies
   uv sync
   ```

2. **Configuration**
   ```bash
   # Set up environment variables
   echo "OPENAI_API_KEY=your_key_here" > .env

   # Configure your servers in servers_config.json
   ```

### Server Configuration Examples

```json
{   
    "LLM": {
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 5000,
        "top_p": 0
    },
    "mcpServers": {
        "filesystem-server": {
            "command": "npx",
            "args": [
                "@modelcontextprotocol/server-filesystem",
                "/path/to/files"
            ]
        },
        "sse-server": {
            "type": "sse",
            "url": "http://localhost:3000/mcp",
            "headers": {
                "Authorization": "Bearer token"
            },
        },
        "docker-server": {
            "command": "docker",
            "args": ["run", "-i", "--rm", "mcp/server"]
        }
    }
}
```

## 🎯 Usage

### Interactive Commands
- `/tools` - List all available tools across servers
- `/prompts` - View available prompts
- `/prompt:<name>/<args>` - Execute a prompt with arguments
  ```
  # Example: Weather prompt
  /prompt:weather/location=tokyo/units=metric
  
  # Alternative JSON format
  /prompt:weather/{"location":"tokyo","units":"metric"}
  ```
- `/resources` - List available resources
- `/resource:<uri>` - Access and analyze a resource
- `/debug` - Toggle debug mode
- `/refresh` - Update server capabilities

### Prompt Management
```bash
# List all available prompts
/prompts

# Basic prompt usage
/prompt:weather/location=tokyo

# Prompt with multiple arguments depends on the server prompt arguments requirements
/prompt:travel-planner/from=london/to=paris/date=2024-03-25

# JSON format for complex arguments
/prompt:analyze-data/{
    "dataset": "sales_2024",
    "metrics": ["revenue", "growth"],
    "filters": {
        "region": "europe",
        "period": "q1"
    }
}

# Nested argument structures
/prompt:market-research/target=smartphones/criteria={
    "price_range": {"min": 500, "max": 1000},
    "features": ["5G", "wireless-charging"],
    "markets": ["US", "EU", "Asia"]
}
```

### Advanced Prompt Features
- **Argument Validation**: Automatic type checking and validation
- **Default Values**: Smart handling of optional arguments
- **Context Awareness**: Prompts can access previous conversation context
- **Cross-Server Execution**: Seamless execution across multiple MCP servers
- **Error Handling**: Graceful handling of invalid arguments with helpful messages
- **Dynamic Help**: Detailed usage information for each prompt

### AI-Powered Interactions
The client intelligently:
- Chains multiple tools together
- Provides context-aware responses
- Automatically selects appropriate tools
- Handles errors gracefully
- Maintains conversation context

## 🔧 Advanced Features

### Tool Orchestration
```python
# Example of automatic tool chaining if the tool is available in the servers connected
User: "Find charging stations near Silicon Valley and check their current status"

# Client automatically:
1. Uses Google Maps API to locate Silicon Valley
2. Searches for charging stations in the area
3. Checks station status through EV network API
4. Formats and presents results
```

### Resource Analysis
```python
# Automatic resource processing
User: "Analyze the contents of /path/to/document.pdf"

# Client automatically:
1. Identifies resource type
2. Extracts content
3. Processes through LLM
4. Provides intelligent summary
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact & Support

- **Author**: Abiola Adeshina
- **Email**: abioladedayo1993@gmail.com
- **GitHub Issues**: [Report a bug](https://github.com/Abiorh001/mcp_connect/issues)

---

<p align="center">Built with ❤️ by the MCPOmni Connect Team</p> 