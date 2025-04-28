from unittest.mock import Mock
from mcpomni_connect.system_prompts import (
    generate_concise_prompt,
    generate_system_prompt,
)

# Mock data
MOCK_TOOLS = {
    "server1": [
        Mock(
            name="tool1",
            description="Test tool 1",
            inputSchema={
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "First parameter",
                    }
                }
            },
        ),
        Mock(
            name="tool2",
            description="Test tool 2",
            inputSchema={
                "properties": {
                    "param2": {
                        "type": "integer",
                        "description": "Second parameter",
                    }
                }
            },
        ),
    ],
    "server2": [
        Mock(
            name="tool3",
            description="Test tool 3",
            inputSchema={
                "properties": {
                    "param3": {
                        "type": "boolean",
                        "description": "Third parameter",
                    }
                }
            },
        )
    ],
}


def test_generate_concise_prompt():
    """Test generation of concise prompt"""
    # Create mock tools with proper attributes
    mock_tools = {"server1": [Mock(), Mock()], "server2": [Mock()]}

    # Explicitly set name and description attributes
    mock_tools["server1"][0].name = "tool1"
    mock_tools["server1"][0].description = "Test tool 1"
    mock_tools["server1"][1].name = "tool2"
    mock_tools["server1"][1].description = "Test tool 2"
    mock_tools["server2"][0].name = "tool3"
    mock_tools["server2"][0].description = "Test tool 3"

    prompt = generate_concise_prompt(mock_tools)

    # Check basic structure
    assert "You are a helpful AI assistant" in prompt
    assert "AVAILABLE TOOLS" in prompt

    # Check tool descriptions
    assert "[server1]" in prompt
    assert "[server2]" in prompt
    assert "• tool1: Test tool 1" in prompt
    assert "• tool2: Test tool 2" in prompt
    assert "• tool3: Test tool 3" in prompt

    # Check guidelines
    assert "If a task involves using a tool" in prompt
    assert "confirm with the user before proceeding" in prompt


# def test_generate_detailed_prompt():
#     """Test generation of detailed prompt"""
#     prompt = generate_detailed_prompt(MOCK_TOOLS)

#     # Check basic structure
#     assert "You are an intelligent assistant" in prompt
#     assert "Available Tools by Server:" in prompt

#     # Check tool descriptions with parameters
#     assert "[server1]" in prompt
#     assert "[server2]" in prompt
#     assert "• tool1: Test tool 1" in prompt
#     assert "Parameters:" in prompt
#     assert "- param1 (string): First parameter" in prompt
#     assert "- param2 (integer): Second parameter" in prompt
#     assert "- param3 (boolean): Third parameter" in prompt

#     # Check guidelines
#     assert "Before using any tool:" in prompt
#     assert "When using tools:" in prompt
#     assert "Remember:" in prompt


def test_generate_system_prompt():
    """Test generation of system prompt based on provider"""
    # Mock LLM connection for tool-accepting provider
    mock_llm_connection_tool = Mock()
    mock_llm_connection_tool.llm_config = {"provider": "openai"}

    # Mock LLM connection for non-tool-accepting provider
    mock_llm_connection_no_tool = Mock()
    mock_llm_connection_no_tool.llm_config = {"provider": "other"}

    # Test with tool-accepting provider
    prompt_tool = generate_system_prompt(MOCK_TOOLS, mock_llm_connection_tool)
    assert "You are a helpful AI assistant" in prompt_tool  # Concise prompt

    # Test with non-tool-accepting provider
    prompt_no_tool = generate_system_prompt(MOCK_TOOLS, mock_llm_connection_no_tool)
    assert "You are an intelligent assistant" in prompt_no_tool  # Detailed prompt


# def test_generate_react_agent_prompt():
#     """Test generation of ReAct agent prompt"""
#     prompt = generate_react_agent_prompt(MOCK_TOOLS)

#     # Check basic structure
#     assert "You are an agent" in prompt
#     assert "Process:" in prompt
#     assert "Available Tools by Server:" in prompt or "[server1]" in prompt

#     # Check example
#     assert "Example 1:" in prompt
#     assert "Question: What is my account balance?" in prompt
#     assert "Thought:" in prompt
#     assert "Action:" in prompt
#     assert "Observation:" in prompt
#     assert "Answer:" in prompt

#     # Check tool descriptions with parameters
#     assert "[server1]" in prompt
#     assert "[server2]" in prompt
#     assert "• tool1: Test tool 1" in prompt
#     assert "Parameters:" in prompt
#     assert "- param1 (string): First parameter" in prompt
#     assert "- param2 (integer): Second parameter" in prompt
#     assert "- param3 (boolean): Third parameter" in prompt

#     # Check guidelines
#     assert "Before using any tool:" in prompt
#     assert "When using tools:" in prompt
#     assert "Remember:" in prompt
