import pytest
from unittest.mock import Mock, AsyncMock
from mcpomni_connect.tool_calling_agent import tool_calling_agent


class MockTool:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.inputSchema = parameters


MOCK_TOOLS = {
    "server1": [
        MockTool(
            name="tool1",
            description="Test tool 1",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Test parameter 1",
                    }
                },
            },
        ),
        Mock(name="tool2", description="Test tool 2"),
    ],
    "server2": [
        Mock(name="tool3", description="Test tool 3"),
    ],
}


@pytest.fixture
def mock_sessions():
    mock_session = Mock()
    mock_session.call_tool = AsyncMock(return_value="Tool result")
    return {"server1": {"session": mock_session}}


@pytest.fixture
def mock_llm_connection():
    mock_conn = Mock()
    mock_conn.llm_call = AsyncMock(
        return_value=Mock(
            choices=[Mock(message=Mock(content="Test response", tool_calls=None))]
        )
    )
    return mock_conn


@pytest.fixture
def mock_add_message_to_history():
    return AsyncMock()


@pytest.fixture
def mock_message_history():
    return AsyncMock(return_value=[])


@pytest.mark.asyncio
async def test_process_query_without_agent(
    mock_sessions,
    mock_llm_connection,
    mock_add_message_to_history,
    mock_message_history,
):
    system_prompt = "You are a helpful assistant"
    query = "Test query"

    result = await tool_calling_agent(
        query=query,
        system_prompt=system_prompt,
        llm_connection=mock_llm_connection,
        sessions=mock_sessions,
        server_names=["server1"],
        tools_list=[
            MockTool(
                name="tool1",
                description="Test tool 1",
                parameters={
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "Test parameter 1",
                        }
                    },
                },
            )
        ],
        available_tools=MOCK_TOOLS,
        add_message_to_history=mock_add_message_to_history,
        message_history=mock_message_history,
        debug=False,
    )

    assert result == "Test response"
    assert mock_add_message_to_history.call_count == 2
    mock_add_message_to_history.assert_any_call(role="user", content="Test query")
    mock_add_message_to_history.assert_any_call("assistant", "Test response", {})


@pytest.mark.asyncio
async def test_process_query_with_tool_calls(
    mock_sessions,
    mock_llm_connection,
    mock_add_message_to_history,
    mock_message_history,
):
    system_prompt = "You are a helpful assistant"
    query = "Test query"

    mock_tool = MockTool(
        name="test_tool",
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Test parameter"}
            },
        },
    )

    tool_call = Mock()
    tool_call.id = "call1"
    tool_call.function = Mock()
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = '{"param1": "test_value"}'

    mock_llm_connection.llm_call.side_effect = [
        Mock(
            choices=[
                Mock(message=Mock(content="I'll use a tool", tool_calls=[tool_call]))
            ]
        ),
        Mock(choices=[Mock(message=Mock(content="Tool result processed"))]),
    ]

    result = await tool_calling_agent(
        query=query,
        system_prompt=system_prompt,
        llm_connection=mock_llm_connection,
        sessions=mock_sessions,
        server_names=["server1"],
        tools_list=[mock_tool],
        available_tools={"server1": [mock_tool]},
        add_message_to_history=mock_add_message_to_history,
        message_history=mock_message_history,
        debug=False,
    )

    assert result.startswith("I'll use a tool")
    assert "Tool result processed" in result
    mock_sessions["server1"]["session"].call_tool.assert_called_once_with(
        "test_tool", {"param1": "test_value"}
    )


@pytest.mark.asyncio
async def test_process_query_with_error(
    mock_sessions,
    mock_llm_connection,
    mock_add_message_to_history,
    mock_message_history,
):
    system_prompt = "You are a helpful assistant"
    query = "Test query"

    mock_llm_connection.llm_call.side_effect = Exception("Test error")

    result = await tool_calling_agent(
        query=query,
        system_prompt=system_prompt,
        llm_connection=mock_llm_connection,
        sessions=mock_sessions,
        server_names=["server1"],
        tools_list=[
            tool
            for tools in MOCK_TOOLS.values()
            for tool in tools
            if hasattr(tool, "name")
        ],
        available_tools=MOCK_TOOLS,
        add_message_to_history=mock_add_message_to_history,
        message_history=mock_message_history,
        debug=False,
    )

    assert result == "Error processing query: Test error"
    mock_add_message_to_history.assert_awaited_once_with(
        role="user", content="Test query"
    )
