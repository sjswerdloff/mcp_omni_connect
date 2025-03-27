import pytest
from unittest.mock import AsyncMock
from mcpomni_connect.prompts import (
    find_prompt_server,
    get_prompt,
    get_prompt_with_react_agent,
    list_prompts,
)
from mcpomni_connect.utils import logger

# Mock data for testing
MOCK_AVAILABLE_PROMPTS = {
    "server1": [
        {
            "name": "test-prompt",
            "description": "Test prompt description",
            "arguments": [
                {
                    "name": "arg1",
                    "description": "First argument",
                    "required": True
                }
            ]
        }
    ],
    "server2": [
        {
            "name": "another-prompt",
            "description": "Another test prompt",
            "arguments": []
        }
    ]
}

@pytest.mark.asyncio
async def test_find_prompt_server():
    """Test finding a prompt server"""
    # Test existing prompt
    server_name, found = await find_prompt_server("test-prompt", MOCK_AVAILABLE_PROMPTS)
    assert found is True
    assert server_name == "server1"

    # Test non-existing prompt
    server_name, found = await find_prompt_server("non-existent", MOCK_AVAILABLE_PROMPTS)
    assert found is False
    assert server_name == ""


@pytest.mark.asyncio
async def test_list_prompts():
    """Test listing prompts"""

    # Separate mock session objects for each server
    mock_session_1 = AsyncMock()
    mock_session_1.list_prompts = AsyncMock(return_value=AsyncMock(prompts=[
        {"name": "test-prompt", "description": "Test prompt description"},
    ]))

    mock_session_2 = AsyncMock()
    mock_session_2.list_prompts = AsyncMock(return_value=AsyncMock(prompts=[
        {"name": "another-prompt", "description": "Another test prompt"},
    ]))

    # Mock sessions dictionary with distinct mock sessions
    mock_sessions = {
        "server1": {"session": mock_session_1, "connected": True},
        "server2": {"session": mock_session_2, "connected": True},
    }

    # Call list_prompts with the mock sessions
    prompts = await list_prompts(server_names=["server1", "server2"], sessions=mock_sessions)

    print("Retrieved prompts:", prompts)  # Debugging

    assert len(prompts) == 2  # Now matches expected output
    assert {"name": "test-prompt", "description": "Test prompt description"} in prompts
    assert {"name": "another-prompt", "description": "Another test prompt"} in prompts

@pytest.mark.asyncio
async def test_get_prompt():
    """Test getting a specific prompt"""
    # Mock sessions
    mock_sessions = {
        "server1": {"session": None, "connected": True},
        "server2": {"session": None, "connected": True}
    }

    async def mock_add_message_to_history(*args):
        return {}

    # Test getting existing prompt
    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=lambda *args: "Test LLM response",
        debug=False,
        available_prompts=MOCK_AVAILABLE_PROMPTS,
        name="test-prompt",
        arguments={"arg1": "test_value"}
    )
    assert content is not None

    # Test getting non-existing prompt
    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=lambda *args: "Test LLM response",
        debug=False,
        available_prompts=MOCK_AVAILABLE_PROMPTS,
        name="non-existent",
        arguments={}
    )
    assert "Prompt not found" in content

@pytest.mark.asyncio
async def test_get_prompt_with_react_agent():
    """Test getting a prompt with ReAct agent"""
    # Mock sessions
    mock_sessions = {
        "server1": {"session": None, "connected": True},
        "server2": {"session": None, "connected": True}
    }

    async def mock_add_message_to_history(*args):
        return {}

    # Test getting existing prompt
    content = await get_prompt_with_react_agent(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        debug=False,
        available_prompts=MOCK_AVAILABLE_PROMPTS,
        name="test-prompt",
        arguments={"arg1": "test_value"}
    )
    assert content is not None

    # Test getting non-existing prompt
    content = await get_prompt_with_react_agent(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        debug=False,
        available_prompts=MOCK_AVAILABLE_PROMPTS,
        name="non-existent",
        arguments={}
    )
    assert "Prompt not found" in content
