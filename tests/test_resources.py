import pytest
from unittest.mock import AsyncMock
from mcpomni_connect.resources import (
    list_resources,
    find_resource_server,
    read_resource,
)

# Mock data for testing
class MockResource:
    def __init__(self, uri, name):
        self.uri = uri
        self.name = name

MOCK_RESOURCES = {
    "server1": [
        MockResource("resource1", "Resource 1"),
        MockResource("resource2", "Resource 2"),
    ],
    "server2": [
        MockResource("resource3", "Resource 3"),
    ],
}

MOCK_SESSIONS = {
    "server1": {
        "session": None,
        "connected": True,
    },
    "server2": {
        "session": None,
        "connected": True,
    },
}

@pytest.mark.asyncio
async def test_list_resources():
    """Test listing resources from servers"""
    # Mock the list_resources method
    async def mock_list_resources():
        class MockResponse:
            def __init__(self, resources):
                self.resources = resources  # Use MockResource objects directly
        return MockResponse(MOCK_RESOURCES["server1"])

    # Update mock sessions with mock method
    test_sessions = MOCK_SESSIONS.copy()
    test_sessions["server1"]["session"] = AsyncMock()
    test_sessions["server1"]["session"].list_resources.side_effect = mock_list_resources

    resources = await list_resources(
        server_names=["server1", "server2"],
        sessions=test_sessions
    )
    
    # Assertions
    assert len(resources) == 2
    assert all(hasattr(res, "uri") and hasattr(res, "name") for res in resources)
    assert {res.uri for res in resources} == {"resource1", "resource2"}

@pytest.mark.asyncio
async def test_find_resource_server():
    """Test finding a resource server"""
    # Test existing resource
    server_name, found = await find_resource_server("resource1", MOCK_RESOURCES)
    assert found is True
    assert server_name == "server1"

    # Test non-existing resource
    server_name, found = await find_resource_server("non-existent", MOCK_RESOURCES)
    assert found is False
    assert server_name == ""

@pytest.mark.asyncio
async def test_read_resource():
    """Test reading a resource"""
    # Mock the read_resource method
    async def mock_read_resource(*args, **kwargs):
        uri = args[0]  # Extract the URI from the arguments
        return f"Content of {uri}"

    # Mock the LLM call
    async def mock_llm_call(messages):
        return type("MockResponse", (), {
            "choices": [
                type("MockChoice", (), {
                    "message": type("MockMessage", (), {
                        "content": "Processed content"
                    })()
                })()
            ]
        })()

    # Mock the add_message_to_history function
    async def mock_add_message_to_history(role, content, metadata=None):
        pass

    # Update mock sessions with mock method
    test_sessions = MOCK_SESSIONS.copy()
    test_sessions["server1"]["session"] = type("MockSession", (), {
        "read_resource": mock_read_resource
    })()

    # Test successful resource read
    content = await read_resource(
        uri="resource1",
        sessions=test_sessions,
        available_resources=MOCK_RESOURCES,
        add_message_to_history=mock_add_message_to_history,
        llm_call=mock_llm_call,
        debug=False
    )
    assert content == "Processed content"

    # Test non-existing resource
    content = await read_resource(
        uri="non-existent",
        sessions=test_sessions,
        available_resources=MOCK_RESOURCES,
        add_message_to_history=mock_add_message_to_history,
        llm_call=mock_llm_call,
        debug=False
    )
    assert "Resource not found" in content