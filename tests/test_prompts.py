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
                    "required": True,
                }
            ],
        }
    ],
    "server2": [
        {
            "name": "another-prompt",
            "description": "Another test prompt",
            "arguments": [],
        }
    ],
}


@pytest.mark.asyncio
async def test_find_prompt_server():
    """Test finding a prompt server"""
    # Test existing prompt
    server_name, found = await find_prompt_server(
        "test-prompt", MOCK_AVAILABLE_PROMPTS
    )
    assert found is True
    assert server_name == "server1"

    # Test non-existing prompt
    server_name, found = await find_prompt_server(
        "non-existent", MOCK_AVAILABLE_PROMPTS
    )
    assert found is False
    assert server_name == ""


@pytest.mark.asyncio
async def test_list_prompts():
    """Test listing prompts"""

    # Separate mock session objects for each server
    mock_session_1 = AsyncMock()
    mock_session_1.list_prompts = AsyncMock(
        return_value=AsyncMock(
            prompts=[
                {
                    "name": "test-prompt",
                    "description": "Test prompt description",
                },
            ]
        )
    )

    mock_session_2 = AsyncMock()
    mock_session_2.list_prompts = AsyncMock(
        return_value=AsyncMock(
            prompts=[
                {
                    "name": "another-prompt",
                    "description": "Another test prompt",
                },
            ]
        )
    )

    # Mock sessions dictionary with distinct mock sessions
    mock_sessions = {
        "server1": {"session": mock_session_1, "connected": True},
        "server2": {"session": mock_session_2, "connected": True},
    }

    # Call list_prompts with the mock sessions
    prompts = await list_prompts(
        server_names=["server1", "server2"], sessions=mock_sessions
    )

    print("Retrieved prompts:", prompts)  # Debugging

    assert len(prompts) == 2  # Now matches expected output
    assert {
        "name": "test-prompt",
        "description": "Test prompt description",
    } in prompts
    assert {
        "name": "another-prompt",
        "description": "Another test prompt",
    } in prompts


@pytest.mark.asyncio
async def test_get_prompt():
    """Test getting a specific prompt"""
    # Mock sessions
    mock_sessions = {
        "server1": {"session": None, "connected": True},
        "server2": {"session": None, "connected": True},
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
        arguments={"arg1": "test_value"},
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
        arguments={},
    )
    assert "Prompt not found" in content


@pytest.mark.asyncio
async def test_get_prompt_with_react_agent():
    """Test getting a prompt with ReAct agent"""
    # Mock sessions
    mock_sessions = {
        "server1": {"session": None, "connected": True},
        "server2": {"session": None, "connected": True},
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
        arguments={"arg1": "test_value"},
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
        arguments={},
    )
    assert "Prompt not found" in content


@pytest.mark.asyncio
async def test_list_prompts_edge_cases():
    """Test listing prompts with edge cases"""
    # Test disconnected server
    mock_sessions = {
        "server1": {"session": AsyncMock(), "connected": False},
    }
    prompts = await list_prompts(["server1"], mock_sessions)
    assert len(prompts) == 0

    # Test server throwing exception
    mock_session = AsyncMock()
    mock_session.list_prompts = AsyncMock(
        side_effect=Exception("Server error")
    )
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }
    prompts = await list_prompts(["server1"], mock_sessions)
    assert len(prompts) == 0

    # Test empty prompts response
    mock_session = AsyncMock()
    mock_session.list_prompts = AsyncMock(return_value=AsyncMock(prompts=[]))
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }
    prompts = await list_prompts(["server1"], mock_sessions)
    assert len(prompts) == 0


@pytest.mark.asyncio
async def test_find_prompt_server_edge_cases():
    """Test finding prompt server with edge cases"""

    # Test with object-style prompts
    class PromptObj:
        def __init__(self, name):
            self.name = name

    mixed_prompts = {
        "server1": [{"name": "dict-prompt"}],
        "server2": [PromptObj("obj-prompt")],
    }

    server_name, found = await find_prompt_server("dict-prompt", mixed_prompts)
    assert found is True
    assert server_name == "server1"

    server_name, found = await find_prompt_server("obj-prompt", mixed_prompts)
    assert found is True
    assert server_name == "server2"

    # Test with empty prompts
    empty_prompts = {
        "server1": [],
        "server2": [],
    }
    server_name, found = await find_prompt_server("any-prompt", empty_prompts)
    assert found is False
    assert server_name == ""

    # Test with duplicate prompt names
    duplicate_prompts = {
        "server1": [{"name": "common-prompt"}],
        "server2": [{"name": "common-prompt"}],
    }
    server_name, found = await find_prompt_server(
        "common-prompt", duplicate_prompts
    )
    assert found is True
    # Should return the first server that has the prompt
    assert server_name == "server1"


@pytest.mark.asyncio
async def test_get_prompt_advanced():
    """Test get_prompt with advanced scenarios"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    async def mock_add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    async def mock_llm_call(messages):
        return AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="LLM Response"))]
        )

    # Test with object-style message content
    class MessageContent:
        def __init__(self, text):
            self.text = text

    mock_session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[
                AsyncMock(role="user", content=MessageContent("Test message"))
            ]
        )
    )

    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=mock_llm_call,
        debug=True,  # Test debug mode
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert content == "LLM Response"

    # Test LLM call failure
    async def failing_llm_call(messages):
        raise Exception("LLM Error")

    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=failing_llm_call,
        debug=False,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert "Error getting prompt" in content


@pytest.mark.asyncio
async def test_get_prompt_with_react_agent_advanced():
    """Test get_prompt_with_react_agent with advanced scenarios"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    async def mock_add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    # Test with different message content formats
    class ComplexContent:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or {}

    mock_session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[
                AsyncMock(
                    role="assistant",
                    content=ComplexContent("Test message", {"key": "value"}),
                )
            ]
        )
    )

    content = await get_prompt_with_react_agent(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert content == "Test message"

    # Test with invalid message structure
    mock_session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[AsyncMock(role="user")]
        )  # Missing content
    )

    content = await get_prompt_with_react_agent(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert "Error getting prompt" in content


@pytest.mark.asyncio
async def test_list_prompts_additional_cases():
    """Test additional cases for listing prompts"""
    # Test multiple disconnected servers
    mock_sessions = {
        "server1": {"session": AsyncMock(), "connected": False},
        "server2": {"session": AsyncMock(), "connected": False},
    }
    prompts = await list_prompts(["server1", "server2"], mock_sessions)
    assert len(prompts) == 0

    # Test mixed server states
    mock_session_connected = AsyncMock()
    mock_session_connected.list_prompts = AsyncMock(
        return_value=AsyncMock(
            prompts=[{"name": "test-prompt", "description": "Test"}]
        )
    )
    mock_sessions = {
        "server1": {"session": mock_session_connected, "connected": True},
        "server2": {"session": AsyncMock(), "connected": False},
    }
    prompts = await list_prompts(["server1", "server2"], mock_sessions)
    assert len(prompts) == 1
    assert prompts[0]["name"] == "test-prompt"

    # Test malformed prompt response
    mock_session_malformed = AsyncMock()
    mock_session_malformed.list_prompts = AsyncMock(
        return_value=AsyncMock(prompts=[{"invalid_key": "value"}])
    )
    mock_sessions = {
        "server1": {"session": mock_session_malformed, "connected": True},
    }
    prompts = await list_prompts(["server1"], mock_sessions)
    assert len(prompts) == 1  # Should still return the malformed prompt


@pytest.mark.asyncio
async def test_get_prompt_empty_messages():
    """Test get_prompt with empty message list"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    async def mock_add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    # Test empty messages list
    mock_session.get_prompt = AsyncMock(return_value=AsyncMock(messages=[]))

    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=AsyncMock(),
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert content is not None


@pytest.mark.asyncio
async def test_get_prompt_content_types():
    """Test get_prompt with different content types"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    async def mock_add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    async def mock_llm_call(messages):
        return AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="LLM Response"))]
        )

    # Test dict content
    mock_session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[
                AsyncMock(
                    role="user",
                    content={"text": "Dict message", "metadata": {}},
                )
            ]
        )
    )

    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=mock_llm_call,
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert content == "LLM Response"

    # Test missing role attribute
    mock_session.get_prompt = AsyncMock(
        return_value=AsyncMock(messages=[AsyncMock(content="No role message")])
    )

    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=mock_llm_call,
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert content == "LLM Response"


@pytest.mark.asyncio
async def test_get_prompt_arguments_validation():
    """Test get_prompt arguments validation"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    async def mock_add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    # Test with invalid arguments
    mock_session.get_prompt = AsyncMock(
        side_effect=Exception("Invalid arguments")
    )

    content = await get_prompt(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        llm_call=AsyncMock(),
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
        arguments={"invalid_arg": "value"},
    )
    assert "Error getting prompt" in content


@pytest.mark.asyncio
async def test_get_prompt_with_react_agent_empty_messages():
    """Test get_prompt_with_react_agent with empty message list"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    async def mock_add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    # Test empty messages list
    mock_session.get_prompt = AsyncMock(return_value=AsyncMock(messages=[]))

    content = await get_prompt_with_react_agent(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )
    assert content is not None


@pytest.mark.asyncio
async def test_get_prompt_with_react_agent_debug_logging():
    """Test get_prompt_with_react_agent debug logging"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    async def mock_add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    # Test debug logging
    mock_session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[AsyncMock(role="user", content="Debug test message")]
        )
    )

    with pytest.LogCaptureFixture() as log_capture:
        content = await get_prompt_with_react_agent(
            sessions=mock_sessions,
            system_prompt="Test system prompt",
            add_message_to_history=mock_add_message_to_history,
            debug=True,
            available_prompts={"server1": [{"name": "test-prompt"}]},
            name="test-prompt",
        )
        assert "Debug test message" in str(log_capture)


@pytest.mark.asyncio
async def test_get_prompt_with_react_agent_error_metadata():
    """Test get_prompt_with_react_agent error metadata handling"""
    mock_session = AsyncMock()
    mock_sessions = {
        "server1": {"session": mock_session, "connected": True},
    }

    metadata_captured = {}

    async def mock_add_message_to_history(role, content, metadata=None):
        nonlocal metadata_captured
        metadata_captured = metadata or {}
        return {"role": role, "content": content}

    # Test error with metadata
    mock_session.get_prompt = AsyncMock(side_effect=Exception("Test error"))

    content = await get_prompt_with_react_agent(
        sessions=mock_sessions,
        system_prompt="Test system prompt",
        add_message_to_history=mock_add_message_to_history,
        debug=True,
        available_prompts={"server1": [{"name": "test-prompt"}]},
        name="test-prompt",
    )

    assert "Error getting prompt" in content
    assert metadata_captured.get("error") is True
    assert metadata_captured.get("prompt_name") == "test-prompt"
