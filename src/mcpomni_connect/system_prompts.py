from typing import Any, Callable

from mcpomni_connect.constants import TOOL_ACCEPTING_PROVIDERS
from mcpomni_connect.utils import logger


def generate_concise_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
) -> str:
    """Generate a concise prompt for LLMs that accept tools in input"""
    prompt = """You are a helpful AI assistant with access to various tools to help users with their tasks. Your responses should be clear, concise, and focused on the user's needs. 

Before performing any action or using any tool, you must:
1. Explicitly ask the user for permission.
2. Clearly explain what data will be accessed or what action will be taken, including any potential sensitivity of the data or operation.
3. Ensure the user understands the implications and has given explicit consent.
4. Avoid sharing or transmitting any information that is not directly relevant to the user's request.

If a task involves using a tool or accessing sensitive data:
- Provide a detailed description of the tool's purpose and behavior.
- Confirm with the user before proceeding.
- Log the user's consent and the action performed for auditing purposes.


Available tools:
"""
    # Add tool descriptions without full schemas
    # In generate_concise_prompt
    for server_name, tools in available_tools.items():
        prompt += f"\n[{server_name}]"
        for tool in tools:
            # Explicitly convert name and description to strings
            tool_name = str(tool.name)
            tool_description = str(tool.description).split("\n")[0] if tool.description else "No description available"
            prompt += f"\n• {tool_name}: {tool_description}"

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

If a task involves using a tool or accessing sensitive data, describe the tool's purpose and behavior, and confirm with the user before proceeding. Always prioritize user consent, data privacy, and safety.
"""
    return prompt


def generate_detailed_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
) -> str:
    """Generate a detailed prompt for LLMs that don't accept tools in input"""
    base_prompt = """You are an intelligent assistant with access to various tools and resources through the Model Context Protocol (MCP).

Before performing any action or using any tool, you must:
1. Explicitly ask the user for permission.
2. Clearly explain what data will be accessed or what action will be taken, including any potential sensitivity of the data or operation.
3. Ensure the user understands the implications and has given explicit consent.
4. Avoid sharing or transmitting any information that is not directly relevant to the user's request.

If a task involves using a tool or accessing sensitive data:
- Provide a detailed description of the tool's purpose and behavior.
- Confirm with the user before proceeding.
- Log the user's consent and the action performed for auditing purposes.

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
            # Explicitly convert name and description to strings
            tool_name = str(tool.name)
            tool_description = str(tool.description)
            tool_desc = f"• {tool_name}: {tool_description}"
            # Add parameters if they exist
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                params = tool.inputSchema.get("properties", {})
                if params:
                    tool_desc += "\n  Parameters:"
                    for param_name, param_info in params.items():
                        param_desc = param_info.get("description", "No description")
                        param_type = param_info.get("type", "any")
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

If a task involves using a tool or accessing sensitive data, describe the tool's purpose and behavior, and confirm with the user before proceeding. Always prioritize user consent, data privacy, and safety.
"""
    return base_prompt + "".join(tools_section) + interaction_guidelines


def generate_system_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
    llm_connection: Callable[[], Any],
) -> str:
    """Generate a dynamic system prompt based on available tools and capabilities"""

    # Get current provider from LLM config
    if hasattr(llm_connection, 'llm_config'):
        current_provider = llm_connection.llm_config.get("provider", "").lower()
    else:
        current_provider = ""

    # Choose appropriate prompt based on provider
    if current_provider in TOOL_ACCEPTING_PROVIDERS:
        return generate_concise_prompt(available_tools)
    else:
        return generate_detailed_prompt(available_tools)


def generate_react_agent_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
) -> str:
    """Generate prompt for ReAct agent"""
    base_prompt = """You are an agent, designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.
You run in a loop of Thought, Action, PAUSE, Observation.
When you have the answer, output it as "Answer: [your answer]"

Process:
1. Thought: Use this to reason about the problem and determine what action to take.
2. Action: Execute one of the available tools, then return PAUSE.
3. Observation: You will receive the result of your action.
4. Repeat until you have enough information to provide a final answer.

"""
    # Add available tools dynamically
    tools_section = []
    for server_name, tools in available_tools.items():
        tools_section.append(f"\n[{server_name}]")
        for tool in tools:
            # Explicitly convert name and description to strings
            tool_name = str(tool.name)
            tool_description = str(tool.description)
            tool_desc = f"• {tool_name}: {tool_description}"
            # Add parameters if they exist
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                params = tool.inputSchema.get("properties", {})
                if params:
                    tool_desc += "\n  Parameters:"
                    for param_name, param_info in params.items():
                        param_desc = param_info.get("description", "No description")
                        param_type = param_info.get("type", "any")
                        tool_desc += f"\n    - {param_name} ({param_type}): {param_desc}"
            tools_section.append(tool_desc)

    example = """
Example 1:
Question: What is my account balance?
Thought: I need to check the account balance. I'll use the get_account_balance tool.
Action: {
  "tool": tool_name,
  "parameters": {
    "name": "John"
  }
}
PAUSE

Observation: {
  "status": "success",
  "data": 1000
}

Thought: I have found the account balance.
Answer: John has 1000 dollars in his account.
"""

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
    return base_prompt + "".join(tools_section) + example + interaction_guidelines
