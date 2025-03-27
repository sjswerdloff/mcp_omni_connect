import pytest
from unittest.mock import Mock, patch
from mcpomni_connect.llm import LLMConnection

# Mock configuration
MOCK_CONFIG = {
    "openai_api_key": "test-openai-key",
    "groq_api_key": "test-groq-key",
    "openrouter_api_key": "test-openrouter-key",
    "load_config": Mock(return_value={
        "LLM": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9
        }
    })
}

@pytest.fixture
def mock_llm_connection():
    """Create a mock LLM connection"""
    with patch("mcpomni_connect.llm.OpenAI") as mock_openai, \
         patch("mcpomni_connect.llm.Groq") as mock_groq:
        mock_openai.return_value = Mock()
        mock_groq.return_value = Mock()
        connection = LLMConnection(Mock(**MOCK_CONFIG))
        return connection

class TestLLMConnection:
    def test_init(self, mock_llm_connection):
        """Test LLMConnection initialization"""
        assert mock_llm_connection.config is not None
        assert mock_llm_connection.llm_config is not None
        assert mock_llm_connection.llm_config["provider"] == "openai"
        assert mock_llm_connection.llm_config["model"] == "gpt-4"

    def test_llm_configuration(self, mock_llm_connection):
        """Test LLM configuration loading"""
        config = mock_llm_connection.llm_configuration()
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4"
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 1000
        assert config["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_llm_call_openai(self, mock_llm_connection):
        """Test LLM call with OpenAI"""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test_tool", "description": "Test tool"}]
        
        mock_response = Mock()
        mock_llm_connection.openai.chat.completions.create.return_value = mock_response
        
        response = await mock_llm_connection.llm_call(messages, tools)
        
        assert response == mock_response
        mock_llm_connection.openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

    @pytest.mark.asyncio
    async def test_llm_call_groq(self, mock_llm_connection):
        """Test LLM call with Groq"""
        # Update config to use Groq
        mock_llm_connection.llm_config["provider"] = "groq"
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test_tool", "description": "Test tool"}]
        
        mock_response = Mock()
        mock_llm_connection.groq.chat.completions.create.return_value = mock_response
        
        response = await mock_llm_connection.llm_call(messages, tools)
        
        assert response == mock_response
        mock_llm_connection.groq.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

    @pytest.mark.asyncio
    async def test_llm_call_openrouter(self, mock_llm_connection):
        """Test LLM call with OpenRouter"""
        # Update config to use OpenRouter
        mock_llm_connection.llm_config["provider"] = "openrouter"
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test_tool", "description": "Test tool"}]
        
        mock_response = Mock()
        mock_llm_connection.openrouter.chat.completions.create.return_value = mock_response
        
        response = await mock_llm_connection.llm_call(messages, tools)
        
        assert response == mock_response
        mock_llm_connection.openrouter.chat.completions.create.assert_called_once_with(
            extra_body={
                "order": ["openai", "anthropic", "groq"],
                "allow_fallback": True,
                "require_provider": True,
            },
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

    def test_truncate_messages_for_groq(self, mock_llm_connection):
        """Test message truncation for Groq"""
        messages = [
            {"role": "system", "content": "x" * 2000},  # Should be truncated to 1000
            *[{"role": "user", "content": "x" * 600} for _ in range(5)],  # Should be kept
            *[{"role": "assistant", "content": "x" * 600} for _ in range(5)],  # Should be kept
            {"role": "user", "content": "x" * 600},  # May be truncated or removed
            {"role": "assistant", "content": "x" * 600},  # May be truncated or removed
        ]

        truncated = mock_llm_connection.truncate_messages_for_groq(messages)

        print("\n=== Debugging Truncated Messages ===")
        print("Original message count:", len(messages))
        print("Truncated message count:", len(truncated))
        for i, msg in enumerate(truncated):
            print(f"Message {i} ({msg['role']}): {len(msg['content'])} characters")

        # Check system message truncation
        assert truncated[0]["role"] == "system"
        assert len(truncated[0]["content"]) == 1000

        if len(messages) > 10:
            assert any(len(msg["content"]) < 600 for msg in truncated), "Expected some messages to be truncated"
        else:
            assert len(truncated) == len(messages), "No truncation expected for ≤10 messages"

