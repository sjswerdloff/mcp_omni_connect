import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
from mcpomni_connect.react_agent import ReActAgent, AgentState
import asyncio
import json
import logging


# Mock data
class MockTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description


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
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(
        return_value={"status": "success", "data": "Tool result"}
    )
    return {"server1": {"session": mock_session, "connected": True}}


@pytest.fixture
def mock_llm_connection():
    """Create mock LLM connection"""
    mock = AsyncMock()
    mock.llm_call = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(content="Test response"))])
    )
    return mock


@pytest_asyncio.fixture
async def mock_add_message_to_history():
    """Create mock add_message_to_history function"""

    async def add_message_to_history(role, content, metadata=None):
        return {"role": role, "content": content}

    return add_message_to_history


@pytest_asyncio.fixture
async def mock_message_history():
    """Create mock message_history function"""

    async def message_history():
        return [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]

    return message_history


class TestReActAgent:
    def test_init(self):
        """Test ReActAgent initialization"""
        agent = ReActAgent(max_steps=5)
        assert agent.max_steps == 5
        assert agent.state == AgentState.IDLE
        assert len(agent.messages) == 0

    @pytest.mark.asyncio
    async def test_parse_action_valid(self):
        """Test parsing valid action"""
        agent = ReActAgent()
        response = """
        Thought: I need to use tool1
        Action: {"tool": "tool1", "parameters": {"param1": "value1"}}
        """
        result = agent.parse_action(response, MOCK_TOOLS)
        assert result["action"] is True
        assert result["tool_name"] == "tool1"
        assert result["tool_args"] == {"param1": "value1"}
        assert result["server_name"] == "server1"

    @pytest.mark.asyncio
    async def test_parse_action_invalid_json(self):
        """Test parsing invalid JSON action"""
        agent = ReActAgent()
        response = """
        Thought: I need to use a tool
        Action: {invalid json}
        """
        result = agent.parse_action(response, MOCK_TOOLS)
        assert result["action"] is False
        assert "Invalid JSON format" in result["error"]

    @pytest.mark.asyncio
    async def test_parse_action_unknown_tool(self):
        """Test parsing action with unknown tool"""
        agent = ReActAgent()
        response = """
        Action: {"tool": "unknown_tool", "parameters": {}}
        """
        result = agent.parse_action(response, MOCK_TOOLS)
        assert result["action"] is False
        assert "Tool unknown_tool not found" in result["error"]

    @pytest.mark.asyncio
    async def test_parse_response_final_answer(self):
        """Test parsing response with final answer"""
        agent = ReActAgent()
        response = "Final Answer: This is the final answer"
        result = agent.parse_response(response, MOCK_TOOLS)
        assert "answer" in result
        assert result["answer"] == "This is the final answer"

    @pytest.mark.asyncio
    async def test_parse_response_with_action(self):
        """Test parsing response with action"""
        agent = ReActAgent()
        response = """
        Thought: Using tool1
        Action: {"tool": "tool1", "parameters": {"param1": "value1"}}
        """
        result = agent.parse_response(response, MOCK_TOOLS)
        assert result["action"] is True
        assert result["tool_name"] == "tool1"

    @pytest.mark.asyncio
    async def test_execute_tool_success(
        self, mock_sessions, mock_add_message_to_history
    ):
        """Test successful tool execution"""
        agent = ReActAgent()
        tool_call_id = "test-id"
        result = await agent._execute_tool(
            mock_sessions,
            "server1",
            "tool1",
            {"param1": "value1"},
            tool_call_id,
            mock_add_message_to_history,
        )
        assert result == "Tool result"

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, mock_sessions, mock_add_message_to_history):
        """Test tool execution with error"""
        agent = ReActAgent()
        mock_sessions["server1"]["session"].call_tool = AsyncMock(
            side_effect=Exception("Tool error")
        )
        result = await agent._execute_tool(
            mock_sessions,
            "server1",
            "tool1",
            {"param1": "value1"},
            "test-id",
            mock_add_message_to_history,
        )
        assert json.loads(result)["status"] == "error"

    @pytest.mark.asyncio
    async def test_update_llm_working_memory(self, mock_message_history):
        """Test updating LLM working memory"""
        agent = ReActAgent()
        await agent.update_llm_working_memory(mock_message_history)
        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_act_success(self, mock_sessions, mock_add_message_to_history):
        """Test successful act execution"""
        agent = ReActAgent()
        parsed_response = {
            "action": True,
            "tool_name": "tool1",
            "tool_args": {"param1": "value1"},
            "server_name": "server1",
        }
        response = "Using tool1"

        await agent.act(
            parsed_response,
            response,
            mock_add_message_to_history,
            mock_sessions,
            "Test system prompt",
            debug=True,
        )

        assert agent.state == AgentState.OBSERVING

    @pytest.mark.asyncio
    async def test_act_timeout(self, mock_sessions, mock_add_message_to_history):
        """Test act with tool timeout"""
        agent = ReActAgent(tool_call_timeout=0.1)

        # Initialize agent messages with system prompt
        agent.messages = [{"role": "system", "content": "Test system prompt"}]

        # Mock the timeout by making the call_tool function sleep longer than timeout
        async def delayed_call(*args, **kwargs):
            await asyncio.sleep(0.2)  # Sleep longer than timeout
            return {"status": "success", "data": "Should not reach this"}

        mock_sessions["server1"]["session"].call_tool = delayed_call

        parsed_response = {
            "action": True,
            "tool_name": "tool1",
            "tool_args": {"param1": "value1"},
            "server_name": "server1",
        }

        # Set initial state
        async with agent.agent_state_context(AgentState.TOOL_CALLING):
            await agent.act(
                parsed_response,
                "Using tool1",
                mock_add_message_to_history,
                mock_sessions,
                "Test system prompt",
            )

            # Print all messages for debugging
            print("All messages:")
            for msg in agent.messages:
                print(f"Message: {msg}")

            # Verify the timeout message is added correctly
            assert (
                len(agent.messages) >= 2
            )  # Should have system prompt and timeout message
            assert agent.messages[0]["role"] == "system"
            assert agent.messages[-1]["role"] == "user"
            assert "Tool call timed out" in agent.messages[-1]["content"]
            assert agent.state == AgentState.TOOL_CALLING

    @pytest.mark.asyncio
    async def test_run_with_final_answer(
        self,
        mock_sessions,
        mock_llm_connection,
        mock_add_message_to_history,
        mock_message_history,
    ):
        """Test run with immediate final answer"""
        agent = ReActAgent()
        # Ensure agent starts in IDLE state (default state)
        assert agent.state == AgentState.IDLE

        # Setup the LLM response
        mock_llm_connection.llm_call = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="Final Answer: Test complete"))]
            )
        )

        # Capture state changes
        state_changes = []
        original_info = logging.info

        def mock_info(msg):
            if "Agent state changed from" in msg:
                state_changes.append(msg)
            original_info(msg)

        logging.info = mock_info

        try:
            result = await agent.run(
                mock_sessions,
                "Test system prompt",
                "Test query",
                mock_llm_connection,
                MOCK_TOOLS,
                mock_add_message_to_history,
                mock_message_history,
                debug=True,
            )

            # Verify the result
            assert result == "Test complete"

            # Verify state transitions
            assert len(state_changes) > 0
            assert "RUNNING to FINISHED" in state_changes[-1]

            # Verify the message flow
            assert (
                len(agent.messages) >= 2
            )  # Should have at least system prompt and final answer
            assert agent.messages[0]["role"] == "system"
            assert agent.messages[-1]["role"] == "assistant"
            assert agent.messages[-1]["content"] == "Test complete"

        finally:
            # Restore original logger
            logging.info = original_info

    # @pytest.mark.asyncio
    # async def test_run_with_tool_chain(
    #     self,
    #     mock_sessions,
    #     mock_llm_connection,
    #     mock_add_message_to_history,
    #     mock_message_history,
    # ):
    #     """Test run with tool chain execution"""
    #     agent = ReActAgent(max_steps=2)

    #     async with agent.agent_state_context(AgentState.RUNNING):
    #         mock_llm_connection.llm_call = AsyncMock(
    #             side_effect=[
    #                 Mock(
    #                     choices=[
    #                         Mock(
    #                             message=Mock(
    #                                 content="""
    #                                 Thought: Using tool1
    #                                 Action: {"tool": "tool1", "parameters": {"param1": "value1"}}
    #                                 """
    #                             )
    #                         )
    #                     ]
    #                 ),
    #                 Mock(
    #                     choices=[
    #                         Mock(
    #                             message=Mock(
    #                                 content="Final Answer: Tool chain complete"
    #                             )
    #                         )
    #                     ]
    #                 ),
    #             ]
    #         )

    #         result = await agent.run(
    #             mock_sessions,
    #             "Test system prompt",
    #             "Test query",
    #             mock_llm_connection,
    #             MOCK_TOOLS,
    #             mock_add_message_to_history,
    #             mock_message_history,
    #             debug=True,
    #         )

    #         assert result == "Tool chain complete"
    #         assert agent.state == AgentState.FINISHED

    # @pytest.mark.asyncio
    # async def test_agent_state_context(self):
    #     """Test agent state context manager"""
    #     agent = ReActAgent()

    #     async with agent.agent_state_context(AgentState.RUNNING):
    #         assert agent.state == AgentState.RUNNING

    #     assert agent.state == AgentState.IDLE

    #     with pytest.raises(ValueError):
    #         async with agent.agent_state_context("invalid_state"):
    #             pass

    #     try:
    #         async with agent.agent_state_context(AgentState.RUNNING):
    #             raise Exception("Test error")
    #     except Exception:
    #         await asyncio.sleep(0.1)
    #         assert agent.state == AgentState.ERROR

    # @pytest.mark.asyncio
    # async def test_agent_state_transitions(self):
    #     """Test complete agent state transition flow"""
    #     agent = ReActAgent()

    #     assert agent.state == AgentState.IDLE

    #     async with agent.agent_state_context(AgentState.RUNNING):
    #         assert agent.state == AgentState.RUNNING

    #         async with agent.agent_state_context(AgentState.TOOL_CALLING):
    #             assert agent.state == AgentState.TOOL_CALLING

    #             async with agent.agent_state_context(AgentState.OBSERVING):
    #                 assert agent.state == AgentState.OBSERVING

    #             assert agent.state == AgentState.TOOL_CALLING

    #         assert agent.state == AgentState.RUNNING

    #         async with agent.agent_state_context(AgentState.FINISHED):
    #             assert agent.state == AgentState.FINISHED

    #         assert agent.state == AgentState.RUNNING

    #     assert agent.state == AgentState.IDLE
