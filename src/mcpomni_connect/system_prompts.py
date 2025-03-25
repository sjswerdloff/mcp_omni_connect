from typing import Any, Callable
from mcpomni_connect.utils import logger

def generate_concise_prompt(available_tools: dict[str, list[dict[str, Any]]]) -> str:
    """Generate a concise prompt for LLMs that accept tools in input"""
    prompt = """You are a helpful AI assistant with access to various tools to help users with their tasks.
Your responses should be clear, concise, and focused on the user's needs.

Available tools:
"""
    # Add tool descriptions without full schemas
    for server_name, tools in available_tools.items():
        prompt += f"\n[{server_name}]"
        for tool in tools:
            # Only show name and description
            description = tool.description.split('\n')[0] if tool.description else "No description available"
            prompt += f"\n• {tool.name}: {description}"

    prompt += """
When using tools:
1. Use them only when necessary to answer the user's question
2. Provide clear explanations of what you're doing
3. Handle errors gracefully and inform the user if something goes wrong
4. If a tool call fails, try alternative approaches or explain why it's not possible

Remember to:
- Be direct and concise in your responses
- Focus on the user's specific needs
- Explain your reasoning when using tools
- Handle errors gracefully
- Provide clear next steps when appropriate
"""
    return prompt

def generate_detailed_prompt(available_tools: dict[str, list[dict[str, Any]]]) -> str:
    """Generate a detailed prompt for LLMs that don't accept tools in input"""
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
    for server_name, tools in available_tools.items():
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

def generate_system_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
    llm_connection: Callable[[], Any]
) -> str:
    """Generate a dynamic system prompt based on available tools and capabilities"""
    
    # Set of providers that accept tools in input
    TOOL_ACCEPTING_PROVIDERS = {"groq", "openai"}
    
    # Get current provider from LLM config
    current_provider = llm_connection.llm_config.get("provider", "").lower()
    
    # Choose appropriate prompt based on provider
    if current_provider in TOOL_ACCEPTING_PROVIDERS:
        return generate_concise_prompt(available_tools)
    else:
        return generate_detailed_prompt(available_tools)
