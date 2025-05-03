# import pytest
# import asyncio
# import json
# from unittest.mock import AsyncMock, MagicMock
# from mcpomni_connect.agents.orchestrator import OrchestratorAgent
# from mcpomni_connect.agents.types import AgentConfig, ParsedResponse

# @pytest.fixture
# def agent_config():
#     return AgentConfig(
#         agent_name="orchestrator",
#         max_steps=5,
#         tool_call_timeout=10,
#         request_limit=100,
#         total_tokens_limit=1000,
#         mcp_enabled=True,
#     )

# @pytest.fixture
# def agents_registry():
#     return {
#         "summary": "You summarize text",
#         "report": "You write reports",
#     }

# @pytest.fixture
# def orchestrator(agent_config, agents_registry):
#     return OrchestratorAgent(
#         config=agent_config,
#         agents_registry=agents_registry,
#         chat_id=123,
#         current_date_time="2025-05-03",
#         debug=True,
#     )

# @pytest.mark.asyncio
# async def test_extract_action_json_valid(orchestrator):
#     response = ParsedResponse(data=json.dumps({"agent_name": "summary", "task": "Summarize this"}))
#     result = await orchestrator.extract_action_json(response)
#     assert result.get("action") is True
#     assert result.get("agent_name") == "summary"
#     assert result.get("task") == "Summarize this"

# @pytest.mark.asyncio
# async def test_extract_action_json_invalid_json(orchestrator):
#     response = ParsedResponse(data="{invalid_json}")
#     result = await orchestrator.extract_action_json(response)
#     assert result.get("action") is False
#     assert result.get("error") == "Invalid JSON format"

# @pytest.mark.asyncio
# async def test_extract_action_json_missing_fields(orchestrator):
#     response = ParsedResponse(data=json.dumps({"foo": "bar"}))
#     result = await orchestrator.extract_action_json(response)
#     assert result.get("action") is False
#     assert "error" in result

# # @pytest.mark.asyncio
# # async def test_create_agent_system_prompt(orchestrator):
# #     available_tools = {}
# #     prompt = await orchestrator.create_agent_system_prompt("summary", available_tools)
# #     assert isinstance(prompt, str)
# #     assert "summarize" in prompt.lower()

# # @pytest.mark.asyncio
# # async def test_update_llm_working_memory(orchestrator):
# #     message_history = AsyncMock(return_value=[
# #         {"role": "user", "content": "What is the summary?"},
# #         {"role": "assistant", "content": "Here is the summary..."},
# #         {"role": "system", "content": "System initialized."},
# #     ])
# #     await orchestrator.update_llm_working_memory(message_history)
# #     assert orchestrator.orchestrator_messages == await message_history()

# @pytest.mark.asyncio
# async def test_act_success(orchestrator):
#     llm_connection = AsyncMock(return_value="Here is the output.")
#     add_message_to_history = AsyncMock()
#     message_history = AsyncMock(return_value=[])

#     result = await orchestrator.act(
#         sessions={},
#         agent_name="summary",
#         task="Summarize this",
#         add_message_to_history=add_message_to_history,
#         llm_connection=llm_connection,
#         available_tools={},
#         message_history=message_history,
#         tool_call_timeout=5,
#         max_steps=2,
#         request_limit=100,
#         total_tokens_limit=500,
#     )
#     assert isinstance(result, str)
#     assert "output" in result.lower() or "observation" in result.lower()

# @pytest.mark.asyncio
# async def test_agent_registry_tool(orchestrator):
#     result = await orchestrator.agent_registry_tool(available_tools={"tool_a": "desc"})
#     assert isinstance(result, str)
#     assert "tool" in result.lower()
