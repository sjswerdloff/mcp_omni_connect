import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from mcpomni_connect.react_agent import ReActAgent
import asyncio


# Mock data
class MockTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class TestSession:
    async def call_tool(self, tool_name, tool_args):
        return {"status": "success", "data": "Tool result", "message": None}


MOCK_TOOLS = {
    "server1": [
        MockTool("tool1", "Test tool 1"),
        MockTool("tool2", "Test tool 2"),
    ],
    "server2": [
        MockTool("tool3", "Test tool 3"),
    ],
}


@pytest.fixture
def mock_sessions():
    """Create mock sessions"""
    return {"server1": {"session": TestSession()}}


@pytest.fixture
def mock_llm_connection():
    """Create mock LLM connection"""
    mock = Mock()
    mock.llm_call = AsyncMock(
        return_value=Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )
    )
    return mock


@pytest_asyncio.fixture
async def mock_add_message_to_history():
    """Create mock add_message_to_history function"""

    async def add_message_to_history(role, content, metadata=None):
        pass

    return add_message_to_history


class TestReActAgent:
    def test_init(self):
        """Test ReActAgent initialization"""
        agent = ReActAgent(max_iterations=5)
        assert agent.max_iterations == 5

    def test_first_json_match_success(self):
        """Test successful first JSON match"""
        agent = ReActAgent()
        response = """
        Thought: I need to use a tool
        Action: {"tool": "tool1", "parameters": {"param1": "value1"}}
        PAUSE
        """
        result = agent._first_json_match(response, MOCK_TOOLS)
        assert result["action"] is True
        assert result["tool_name"] == "tool1"
        assert result["tool_args"] == {"param1": "value1"}
        assert result["server_name"] == "server1"

    def test_first_json_match_invalid_json(self):
        """Test first JSON match with invalid JSON"""
        agent = ReActAgent()
        response = """
        Thought: I need to use a tool
        Action: {invalid json}
        PAUSE
        """
        result = agent._first_json_match(response, MOCK_TOOLS)
        assert result["action"] is False
        assert "error" in result

    def test_second_json_match_success(self):
        """Test successful second JSON match"""
        agent = ReActAgent()
        response = """
        Thought: I need to use a tool
        Action: {"tool": "tool2", "parameters": {"param2": "value2"}}
        PAUSE
        """
        result = agent._second_json_match(response, MOCK_TOOLS)
        assert result["action"] is True
        assert result["tool_name"] == "tool2"
        assert result["tool_args"] == {"param2": "value2"}
        assert result["server_name"] == "server1"

    def test_parse_response_with_answer(self):
        """Test parsing response with final answer"""
        agent = ReActAgent()
        response = "Final Answer: This is the answer"
        result = agent.parse_response(response, MOCK_TOOLS)
        assert "answer" in result
        assert result["answer"] == "This is the answer"

    def test_parse_response_with_action(self):
        """Test parsing response with action"""
        agent = ReActAgent()
        response = """
        Thought: I need to use a tool
        Action: {"tool": "tool1", "parameters": {"param1": "value1"}}
        PAUSE
        """
        result = agent.parse_response(response, MOCK_TOOLS)
        assert result["action"] is True
        assert result["tool_name"] == "tool1"

    def test_parse_response_normal(self):
        """Test parsing normal response"""
        agent = ReActAgent()
        response = "This is a normal response"
        result = agent.parse_response(response, MOCK_TOOLS)
        assert "answer" in result
        assert result["answer"] == "This is a normal response"

    @pytest.mark.asyncio
    async def test_execute_tool(
        self, mock_sessions, mock_add_message_to_history
    ):
        """Test tool execution"""
        agent = ReActAgent()
        result = await agent._execute_tool(
            mock_sessions,
            "server1",
            "tool1",
            {"param1": "value1"},
            mock_add_message_to_history,
        )
        assert result == "Tool result"

    @pytest.mark.asyncio
    async def test_run_success(
        self, mock_sessions, mock_llm_connection, mock_add_message_to_history
    ):
        """Test successful run"""
        agent = ReActAgent()
        system_prompt = "You are a helpful assistant"
        query = "Test query"
        message_history = []

        result = await agent.run(
            mock_sessions,
            system_prompt,
            query,
            mock_llm_connection,
            MOCK_TOOLS,
            mock_add_message_to_history,
            message_history,
        )

        assert result == "Test response"
        mock_llm_connection.llm_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_tool_calls(
        self, mock_sessions, mock_llm_connection, mock_add_message_to_history
    ):
        """Test run with tool calls"""
        agent = ReActAgent()
        system_prompt = "You are a helpful assistant"
        query = "Test query"
        message_history = []

        # Mock LLM response with tool calls
        mock_llm_connection.llm_call.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="I'll use a tool",
                        tool_calls=[
                            Mock(
                                id="call1",
                                function=Mock(
                                    name="tool1",
                                    arguments='{"param1": "value1"}',
                                ),
                            )
                        ],
                    )
                )
            ]
        )

        result = await agent.run(
            mock_sessions,
            system_prompt,
            query,
            mock_llm_connection,
            MOCK_TOOLS,
            mock_add_message_to_history,
            message_history,
        )

        assert result == "I'll use a tool"
        mock_llm_connection.llm_call.assert_called_once()
