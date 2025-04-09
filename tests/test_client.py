import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mcpomni_connect.client import Configuration, MCPClient

# Mock data for testing
MOCK_SERVER_CONFIG = {
    "mcpServers": {
        "server1": {
            "type": "stdio",
            "command": "mock_command",
            "args": ["arg1", "arg2"],
            "env": {"TEST_ENV": "test"},
        },
        "server2": {
            "type": "sse",
            "url": "http://test.com",
            "headers": {"Authorization": "Bearer test"},
            "timeout": 5,
            "sse_read_timeout": 300,
        },
        "server3": {"type": "websocket", "url": "ws://test.com"},
    }
}


@pytest.fixture
def mock_env():
    """Fixture to set up mock environment variables"""
    with patch.dict(
        os.environ,
        {
            "LLM_API_KEY": "test_llm_key",
        },
    ):
        yield


@pytest.fixture
def mock_config_file(tmp_path):
    """Fixture to create a mock config file"""
    config_file = tmp_path / "servers_config.json"
    config_file.write_text(json.dumps(MOCK_SERVER_CONFIG))
    return str(config_file)


class TestConfiguration:
    def test_init(self, mock_env):
        """Test Configuration initialization"""
        config = Configuration()
        assert config.llm_api_key == "test_llm_key"

    def test_load_config(self, mock_config_file):
        """Test loading configuration from file"""
        config = Configuration()
        loaded_config = config.load_config(mock_config_file)
        assert loaded_config == MOCK_SERVER_CONFIG

    def test_load_config_invalid_file(self, tmp_path):
        """Test loading configuration from invalid file"""
        config = Configuration()
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid json")
        with pytest.raises(FileNotFoundError):
            config.load_config(str(invalid_file))

    def test_llm_api_key(self, mock_env):
        """Test getting LLM API key"""
        config = Configuration()
        assert config.llm_api_key == "test_llm_key"

    def test_llm_api_key_missing(self):
        """Test getting LLM API key when missing"""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(Configuration, "load_env", return_value=None),
        ):
            with pytest.raises(ValueError) as exc_info:
                Configuration()
            assert "LLM_API_KEY not found in environment variables" in str(
                exc_info.value
            )


class TestMCPClient:
    @pytest.fixture
    def mock_client(self, mock_config_file):
        """Fixture to create a mock MCP client"""
        config = Configuration()
        config.load_config = MagicMock(return_value=MOCK_SERVER_CONFIG)
        return MCPClient(config, debug=True)

    @pytest.mark.asyncio
    async def test_connect_to_servers(self, mock_client):
        """Test connecting to servers"""

        # Mock the _connect_to_single_server to append server names on success
        async def connect_side_effect(server):
            if server["name"] == "server2":
                raise Exception("Connection failed")
            else:
                mock_client.server_names.append(server["name"])

        mock_client._connect_to_single_server = AsyncMock(
            side_effect=connect_side_effect
        )

        successful_connections = await mock_client.connect_to_servers()
        assert successful_connections == 2
        assert len(mock_client.server_names) == 2

    def test_validate_and_convert_url(self, mock_client):
        """Test URL validation and conversion"""
        # Test SSE URL
        url = mock_client._validate_and_convert_url("http://test.com", "sse")
        assert url == "http://test.com"

        # Test WebSocket URL
        url = mock_client._validate_and_convert_url(
            "ws://test.com", "websocket"
        )
        assert url == "ws://test.com"

        # Test invalid SSE URL
        with pytest.raises(ValueError):
            mock_client._validate_and_convert_url("invalid", "sse")

        # Test invalid WebSocket URL
        with pytest.raises(ValueError):
            mock_client._validate_and_convert_url("invalid", "websocket")

        # Test invalid connection type
        with pytest.raises(ValueError):
            mock_client._validate_and_convert_url("http://test.com", "invalid")

    @pytest.mark.asyncio
    async def test_message_history(self, mock_client):
        """Test message history operations"""
        # Test adding message
        await mock_client.add_message_to_history("user", "test message")
        assert len(mock_client.message_history) == 1
        assert mock_client.message_history[0]["role"] == "user"
        assert mock_client.message_history[0]["content"] == "test message"

        # Test clearing history
        await mock_client.clear_history()
        assert len(mock_client.message_history) == 0

        # Test showing history (no return value check, just ensure no errors)
        await mock_client.add_message_to_history("user", "test message")
        await mock_client.add_message_to_history("assistant", "test response")
        await mock_client.show_history()  # Just call the method, don't check return value
        assert (
            len(mock_client.message_history) == 2
        )  # Check the attribute directly
