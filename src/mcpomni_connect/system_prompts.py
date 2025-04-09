from typing import Any, Callable, List, Dict

from mcpomni_connect.constants import TOOL_ACCEPTING_PROVIDERS
from mcpomni_connect.utils import logger


def generate_concise_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
    episodic_memory: List[Dict[str, Any]],
) -> str:
    """Generate a concise system prompt for LLMs that accept tools in input, with structured episodic memory."""
    prompt = """You are a helpful AI assistant with access to various tools to help users with their tasks.


Your behavior should reflect the following:
- Be clear, concise, and focused on the user's needs
- Always ask for consent before using tools or accessing sensitive data
- Explain your reasoning and tool usage clearly
- Clearly explain what data will be accessed or what action will be taken, including any potential sensitivity of the data or operation.
- Ensure the user understands the implications and has given explicit consent.
- Prioritize user preferences, previously known friction points, and successful strategies from memory
- If you recognize similar contexts from past conversations, adapt your approach accordingly
- Do not mention this memory directly in conversation. Use it as a guide to shape your behavior, personalize responses, and anticipate user needs.

---

📘 [EPISODIC MEMORY]
You recall the following relevant past experiences with the user:

"""

    for i, memory in enumerate(episodic_memory, 1):
        prompt += f"\n--- Memory #{i} ---"
        context_tags = ", ".join(memory.get("context_tags", [])) or "N/A"
        conversation_summary = memory.get("conversation_summary", "N/A")
        user_intent = memory.get("user_intent", "N/A")
        effective_strategies = memory.get("effective_strategies", "N/A")
        key_topics = ", ".join(memory.get("key_topics", [])) or "N/A"
        user_preferences = memory.get("user_preferences", "N/A")
        friction_points = memory.get("friction_points", "N/A")
        follow_up = ", ".join(memory.get("follow_up_potential", [])) or "N/A"

        prompt += (
            f"\n• Context Tags: {context_tags}"
            f"\n• Summary: {conversation_summary}"
            f"\n• User Intent: {user_intent}"
            f"\n• Effective Strategies: {effective_strategies}"
            f"\n• Key Topics: {key_topics}"
            f"\n• Preferences: {user_preferences}"
            f"\n• Friction Points: {friction_points}"
            f"\n• Follow-Up Potential: {follow_up}"
        )

    prompt += """
\nWhen helping the user, use this memory to interpret intent, reduce friction, and personalize your response. Memory is crucial — always reference relevant entries when applicable.

---

🧰 [AVAILABLE TOOLS]
You have access to the following tools grouped by server. Use them only when necessary:

"""

    for server_name, tools in available_tools.items():
        prompt += f"\n[{server_name}]"
        for tool in tools:
            tool_name = str(tool.name)
            tool_description = (
                str(tool.description)
                if tool.description
                else "No description available"
            )
            prompt += f"\n• {tool_name}: {tool_description}"

    prompt += """

---

🔐 [TOOL USAGE RULES]
- Always ask the user for consent before using a tool
- Explain what the tool does and what data it accesses
- Inform the user of potential sensitivity or privacy implications
- Log consent and action taken
- If tool call fails, explain and consider alternatives
- If a task involves using a tool or accessing sensitive data:
- Provide a detailed description of the tool's purpose and behavior.
- Confirm with the user before proceeding.
- Log the user's consent and the action performed for auditing purposes.
---

💡 [GENERAL GUIDELINES]
- Be direct and concise
- Explain your reasoning clearly
- Prioritize user-specific needs
- Use memory as guidance
- Offer clear next steps


If a task involves using a tool or accessing sensitive data, describe the tool's purpose and behavior, and confirm with the user before proceeding. Always prioritize user consent, data privacy, and safety.
"""
    return prompt


# def generate_concise_prompt(
#     available_tools: dict[str, list[dict[str, Any]]],
#     episodic_memory: List[Dict[str, Any]],
# ) -> str:
#     """Generate a concise prompt for LLMs that accept tools in input"""
#     prompt = """You are a helpful AI assistant with access to various tools to help users with their tasks. Your responses should be clear, concise, and focused on the user's needs.

# Before performing any action or using any tool, you must:
# 1. Explicitly ask the user for permission.
# 2. Clearly explain what data will be accessed or what action will be taken, including any potential sensitivity of the data or operation.
# 3. Ensure the user understands the implications and has given explicit consent.
# 4. Avoid sharing or transmitting any information that is not directly relevant to the user's request.

# If a task involves using a tool or accessing sensitive data:
# - Provide a detailed description of the tool's purpose and behavior.
# - Confirm with the user before proceeding.
# - Log the user's consent and the action performed for auditing purposes.


# [EPISODIC MEMORY]
# You recall similar conversations with the user, here are the details:

# """
#     # Inject relevant episodic memories based on conversation context
#     for memory in episodic_memory:
#         relevant_context_tags = memory.get("context_tags", [])
#         conversation_summary = memory.get("conversation_summary", "")
#         user_intent = memory.get("user_intent", "")
#         effective_strategies = memory.get("effective_strategies", "")
#         key_topics = memory.get("key_topics", [])
#         user_preferences = memory.get("user_preferences", "")
#         friction_points = memory.get("friction_points", "")
#         follow_up_potential = memory.get("follow_up_potential", [])
#         # Add key memory points (you can adjust this selection as needed)
#         if relevant_context_tags and relevant_context_tags != "N/A":
#             prompt += f"\nContext Tags: {', '.join(relevant_context_tags)}"
#         if conversation_summary and conversation_summary != "N/A":
#             prompt += f"\nConversation Summary: {conversation_summary}"
#         if user_intent and user_intent != "N/A":
#             prompt += f"\nUser Intent: {user_intent}"
#         if effective_strategies and effective_strategies != "N/A":
#             prompt += f"\nEffective Strategies: {effective_strategies}"
#         # Optionally include user preferences and friction points if relevant
#         if key_topics and key_topics != "N/A":
#             prompt += f"\nKey Topics: {', '.join(key_topics)}"
#         if user_preferences and user_preferences != "N/A":
#             prompt += f"\nUser Preferences: {user_preferences}"
#         if friction_points and friction_points != "N/A":
#             prompt += f"\nFriction Points: {friction_points}"

#         # Add potential follow-up topics if relevant
#         if follow_up_potential and follow_up_potential != "N/A":
#             prompt += f"\nFollow-Up Potential: {', '.join(follow_up_potential)}"

#     prompt += """Use these memories as context for your response to the user. [END OF EPISODIC MEMORY]"""
#     prompt += """
# Available tools:
# """
#     # Add tool descriptions without full schemas
#     for server_name, tools in available_tools.items():
#         prompt += f"\n[{server_name}]"
#         for tool in tools:
#             # Explicitly convert name and description to strings
#             tool_name = str(tool.name)
#             tool_description = str(tool.description).split("\n")[0] if tool.description else "No description available"
#             prompt += f"\n• {tool_name}: {tool_description}"

#     prompt += """

# When using tools:
# 1. Use them only when necessary to answer the user's question
# 2. Provide clear explanations of what you're doing
# 3. Handle errors gracefully and inform the user if something goes wrong
# 4. If a tool call fails, try alternative approaches or explain why it's not possible

# Remember to:
# - Be direct and concise in your responses
# - Focus on the user's specific needs
# - Explain your reasoning when using tools
# - Handle errors gracefully
# - Provide clear next steps when appropriate

# If a task involves using a tool or accessing sensitive data, describe the tool's purpose and behavior, and confirm with the user before proceeding. Always prioritize user consent, data privacy, and safety.
# """
#     return prompt


def generate_detailed_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
    episodic_memory: List[Dict[str, Any]],
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

You recall similar conversations with the user, here are the details:
{episodic_memory}

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
                        param_desc = param_info.get(
                            "description", "No description"
                        )
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
    episodic_memory: List[Dict[str, Any]],
) -> str:
    """Generate a dynamic system prompt based on available tools and capabilities"""

    # Get current provider from LLM config
    if hasattr(llm_connection, "llm_config"):
        current_provider = llm_connection.llm_config.get(
            "provider", ""
        ).lower()
    else:
        current_provider = ""

    # Choose appropriate prompt based on provider
    if current_provider in TOOL_ACCEPTING_PROVIDERS:
        return generate_concise_prompt(available_tools, episodic_memory)
    else:
        return generate_detailed_prompt(available_tools, episodic_memory)


# def generate_react_agent_prompt(
#     available_tools: dict[str, list[dict[str, Any]]],
#     episodic_memory: List[Dict[str, Any]],
# ) -> str:
#     """Generate prompt for ReAct agent"""
#     prompt = """You are an agent, designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.
# You run in a loop of Thought, Action, Observation, until you reach an answer.
# [IMPORTANT]
# - differentiate between the user request and when to use a tool
# - do not use a tool if it is not necessary to answer the user request
# - do not hallucinate tools that are not available
# - If you dont have enough information to answer the user request, say so and ask for more information dont use any tool ask for more information from the user

# Process:
# 1. Thought: Use this to reason about the problem and determine what action to take next.
# 2. Action: Execute one of the available tools by outputting a valid JSON object with "tool" and "parameters" fields.
# 3. After each Action, the system will automatically pause execution and process your request.
# 4. Observation: The system will return the result of your action.
# 5. Repeat steps 1-4 until you have enough information to provide a final answer.
# 6. When you have the answer, output it as "Final Answer: [your answer]"

# ---

# 📘 [EPISODIC MEMORY]
# - Prioritize user preferences, previously known friction points, and successful strategies from memory
# - If you recognize similar contexts from past conversations, adapt your approach accordingly
# - Do not mention this memory directly in conversation. Use it as a guide to shape your behavior, personalize responses, and anticipate user needs.
# You recall the following relevant past experiences with the user:

# """

#     for i, memory in enumerate(episodic_memory, 1):
#         prompt += f"\n--- Memory #{i} ---"
#         context_tags = ", ".join(memory.get("context_tags", [])) or "N/A"
#         conversation_summary = memory.get("conversation_summary", "N/A")
#         user_intent = memory.get("user_intent", "N/A")
#         effective_strategies = memory.get("effective_strategies", "N/A")
#         key_topics = ", ".join(memory.get("key_topics", [])) or "N/A"
#         user_preferences = memory.get("user_preferences", "N/A")
#         friction_points = memory.get("friction_points", "N/A")
#         follow_up = ", ".join(memory.get("follow_up_potential", [])) or "N/A"

#         prompt += (
#             f"\n• Context Tags: {context_tags}"
#             f"\n• Summary: {conversation_summary}"
#             f"\n• User Intent: {user_intent}"
#             f"\n• Effective Strategies: {effective_strategies}"
#             f"\n• Key Topics: {key_topics}"
#             f"\n• Preferences: {user_preferences}"
#             f"\n• Friction Points: {friction_points}"
#             f"\n• Follow-Up Potential: {follow_up}"
#         )

#     prompt += """
# \nWhen helping the user, use this memory to interpret intent, reduce friction, and personalize your response. Memory is crucial — always reference relevant entries when applicable.

# ---

# 🧰 [AVAILABLE TOOLS]
# """
#     # Add available tools dynamically
#     tools_section = []
#     for server_name, tools in available_tools.items():
#         tools_section.append(f"\n[{server_name}]")
#         for tool in tools:
#             # Explicitly convert name and description to strings
#             tool_name = str(tool.name)
#             tool_description = str(tool.description)
#             tool_desc = f"• {tool_name}: {tool_description}"
#             # Add parameters if they exist
#             if hasattr(tool, "inputSchema") and tool.inputSchema:
#                 params = tool.inputSchema.get("properties", {})
#                 if params:
#                     tool_desc += "\n  Parameters:"
#                     for param_name, param_info in params.items():
#                         param_desc = param_info.get("description", "No description")
#                         param_type = param_info.get("type", "any")
#                         tool_desc += f"\n    - {param_name} ({param_type}): {param_desc}"
#             tools_section.append(tool_desc)

#     example = """
# Example 1:
# Question: What is my account balance?

# Thought: I need to check the account balance. I'll use the get_account_balance tool.
# Action: {
#   "tool": "get_account_balance",
#   "parameters": {
#     "name": "John"
#   }
# }

# Observation: {
#   "status": "success",
#   "data": 1000
# }

# Thought: I have found the account balance.
# Final Answer: John has 1000 dollars in his account.

# Example 2:
# Question: What's the weather like in New York and should I bring an umbrella?

# Thought: I need to check the current weather in New York. I'll use the weather_check tool.
# Action: {
#   "tool": "weather_check",
#   "parameters": {
#     "location": "New York"
#   }
# }

# Observation: {
#   "status": "success",
#   "data": {
#     "temperature": 65,
#     "conditions": "Light rain",
#     "precipitation_chance": 70
#   }
# }

# Thought: The weather in New York shows light rain with a 70% chance of precipitation. This suggests bringing an umbrella would be advisable.
# Final Answer: The weather in New York is currently 65°F with light rain. There's a 70% chance of precipitation, so yes, you should bring an umbrella.
# """

#     interaction_guidelines = """
# Before using any tool:
# 1. Analyze the user's request carefully
# 2. Check if the required tool is available in the current toolset
# 3. If unclear about the request or tool choice:
#    - Ask for clarification from the user
#    - Explain what information you need
#    - Suggest available alternatives if applicable

# Remember:
# - Only use tools that are listed above
# - Don't assume capabilities that aren't explicitly listed
# - Be transparent about limitations
# - Maintain a helpful and professional tone
# - Always follow the Thought -> Action -> Observation pattern
# - End with a Final Answer once you have all needed information
# """
#     return prompt + "".join(tools_section) + example + interaction_guidelines


def generate_react_agent_prompt(
    available_tools: dict[str, list[dict[str, Any]]],
    episodic_memory: List[Dict[str, Any]],
) -> str:
    """Generate prompt for ReAct agent"""
    prompt = """You are an agent, designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

[UNDERSTANDING USER REQUESTS - CRITICAL]
- FIRST, always carefully analyze the user's request to determine if you fully understand what they're asking
- If the request is unclear, vague, or missing key information, DO NOT use any tools - instead, ask clarifying questions
- Only proceed to the ReAct framework (Thought -> Action -> Observation) if you fully understand the request

[IMPORTANT FORMATTING RULES]
- NEVER use markdown formatting, asterisks, or bold in your responses
- Always use plain text format exactly as shown in the examples
- The exact format and syntax shown in examples must be followed precisely
- CRITICALLY IMPORTANT: Always close JSON objects properly

[IMPORTANT RULES]
- If the user's question can be answered directly without tools, do so without using any tools
- Only use tools when necessary to fulfill the user's request
- Never hallucinate tools that aren't explicitly listed in the available tools section
- If you don't have enough information or the right tools to answer, politely explain your limitations

[REACT PROCESS]
When you understand the request and need to use tools, you run in a loop of:
1. Thought: Use this to understand the problem and plan your approach. then start immediately with the action
2. Action: Execute one of the available tools by outputting a valid JSON object with EXACTLY this format:
   Action: {
     "tool": "tool_name",
     "parameters": {
       "param1": "value1",
       "param2": "value2"
     }
   }
3. After each Action, the system will automatically process your request.
4. Observation: The system will return the result of your action.
5. Repeat steps 1-4 until you have enough information to provide a final answer.
6. When you have the answer, output it as "Final Answer: [your answer]"

---

📘 [EPISODIC MEMORY]
- Prioritize user preferences, previously known friction points, and successful strategies from memory
- If you recognize similar contexts from past conversations, adapt your approach accordingly
- Do not mention this memory directly in conversation. Use it as a guide to shape your behavior, personalize responses, and anticipate user needs.
You recall the following relevant past experiences with the user:

"""

    for i, memory in enumerate(episodic_memory, 1):
        prompt += f"\n--- Memory #{i} ---"
        context_tags = ", ".join(memory.get("context_tags", [])) or "N/A"
        conversation_summary = memory.get("conversation_summary", "N/A")
        user_intent = memory.get("user_intent", "N/A")
        effective_strategies = memory.get("effective_strategies", "N/A")
        key_topics = ", ".join(memory.get("key_topics", [])) or "N/A"
        user_preferences = memory.get("user_preferences", "N/A")
        friction_points = memory.get("friction_points", "N/A")
        follow_up = ", ".join(memory.get("follow_up_potential", [])) or "N/A"

        prompt += (
            f"\n• Context Tags: {context_tags}"
            f"\n• Summary: {conversation_summary}"
            f"\n• User Intent: {user_intent}"
            f"\n• Effective Strategies: {effective_strategies}"
            f"\n• Key Topics: {key_topics}"
            f"\n• Preferences: {user_preferences}"
            f"\n• Friction Points: {friction_points}"
            f"\n• Follow-Up Potential: {follow_up}"
        )

    prompt += """
\nWhen helping the user, use this memory to interpret intent, reduce friction, and personalize your response. Memory is crucial — always reference relevant entries when applicable.

---

🧰 [AVAILABLE TOOLS]
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
                        param_desc = param_info.get(
                            "description", "No description"
                        )
                        param_type = param_info.get("type", "any")
                        tool_desc += f"\n    - {param_name} ({param_type}): {param_desc}"
            tools_section.append(tool_desc)

    example = """
Example 1: Tool usage when needed
Question: What is my account balance?

Thought: This request is asking for account balance information. To answer this, I'll need to query the system using the get_account_balance tool.
Action: {
  "tool": "get_account_balance",
  "parameters": {
    "name": "John"
  }
}

Observation: {
  "status": "success",
  "data": 1000
}

Thought: I have found the account balance.
Final Answer: John has 1000 dollars in his account.

Example 2: Direct answer when no tool is needed
Question: What is the capital of France?

Thought: This is a simple factual question that I can answer directly without using any tools.
Final Answer: The capital of France is Paris.

Example 3: Asking for clarification
Question: Can you check that for me?

Thought: This request is vague and doesn't specify what the user wants me to check. Before using any tools, I should ask for clarification.
Final Answer: I'd be happy to help check something for you, but I need more information. Could you please specify what you'd like me to check?

Example 4: Multiple tool usage
Question: What's the weather like in New York and should I bring an umbrella?

Thought: This request asks about the current weather in New York and advice about bringing an umbrella. I'll need to check the weather information first using a tool.
Action: {
  "tool": "weather_check",
  "parameters": {
    "location": "New York"
  }
}

Observation: {
  "status": "success",
  "data": {
    "temperature": 65,
    "conditions": "Light rain",
    "precipitation_chance": 70
  }
}

Thought: The weather in New York shows light rain with a 70% chance of precipitation. This suggests bringing an umbrella would be advisable.
Final Answer: The weather in New York is currently 65°F with light rain. There's a 70% chance of precipitation, so yes, you should bring an umbrella.
"""

    interaction_guidelines = """
[COMMON ERROR SCENARIOS TO AVOID]
1. Incorrect JSON formatting:
   WRONG: **Action**: {
   WRONG: Action: {
     "tool": "tool_name",
     "parameters": {
       "param1": "value1"
     }
   
   CORRECT: Action: {
     "tool": "tool_name",
     "parameters": {
       "param1": "value1"
     }
   }

2. Using markdown/styling:
   WRONG: **Thought**: I need to check...
   CORRECT: Thought: I need to check...

3. Incomplete steps:
   WRONG: [Skipping directly to Action without Thought]
   CORRECT: Always include Thought before Action

4. Adding extra text:
   WRONG: Action: Let me search for this. {
   CORRECT: Action: {

[DECISION PROCESS]
1. First, verify if you clearly understand the user's request
   - If unclear, ask for clarification without using any tools
   - If clear, proceed to step 2

2. Determine if tools are necessary
   - Can you answer directly with your knowledge? If yes, provide a direct answer
   - Do you need external data or computation? If yes, proceed to step 3

3. When using tools:
   - Select the appropriate tool based on the request
   - Format the Action JSON exactly as shown in the examples
   - Process the observation before deciding next steps
   - Continue until you have enough information

Remember:
- Only use tools that are listed in the available tools section
- Don't assume capabilities that aren't explicitly listed
- Always maintain a helpful and professional tone
- Always focus on addressing the user's actual question
"""
    return prompt + "".join(tools_section) + example + interaction_guidelines
