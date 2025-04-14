import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mcpomni_connect.client import Configuration, MCPClient
import asyncio

# Mock data for testing
MOCK_SERVER_CONFIG = {
    "LLM":{
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 1000,
        "temperature": 0.5,
        "max_input_tokens": 1000,
        "top_p": 1,
    },
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

    @pytest.fixture
    def mock_session(self):
        """Fixture to create a mock session"""
        session = AsyncMock()
        server_info = MagicMock()
        server_info.name = "test_server"  # Use string instead of MagicMock
        session.initialize = AsyncMock(return_value=MagicMock(
            serverInfo=server_info,
            capabilities={"tools": [], "resources": [], "prompts": []}
        ))
        return session

    @pytest.mark.asyncio
    async def test_connect_to_single_server_stdio(self, mock_client, mock_session):
        """Test connecting to a stdio server"""
        with patch('mcpomni_connect.client.stdio_client') as mock_stdio_client:
            mock_transport = (AsyncMock(), AsyncMock())
            mock_stdio_client.return_value.__aenter__.return_value = mock_transport
            
            mock_client.exit_stack.enter_async_context = AsyncMock()
            mock_client.exit_stack.enter_async_context.side_effect = [
                mock_transport,
                mock_session
            ]

            server_info = {"name": "server1", "srv_config": MOCK_SERVER_CONFIG["mcpServers"]["server1"]}
            await mock_client._connect_to_single_server(server_info)

            assert mock_client.server_names == ["test_server"]  # Check for the actual server name
            assert mock_client.sessions["test_server"]["type"] == "stdio"

    @pytest.mark.asyncio
    async def test_connect_to_single_server_sse(self, mock_client, mock_session):
        """Test connecting to an SSE server"""
        with patch('mcpomni_connect.client.sse_client') as mock_sse_client:
            mock_transport = (AsyncMock(), AsyncMock())
            mock_sse_client.return_value.__aenter__.return_value = mock_transport
            
            mock_client.exit_stack.enter_async_context = AsyncMock()
            mock_client.exit_stack.enter_async_context.side_effect = [
                mock_transport,
                mock_session
            ]

            server_info = {"name": "server2", "srv_config": MOCK_SERVER_CONFIG["mcpServers"]["server2"]}
            await mock_client._connect_to_single_server(server_info)

            assert mock_client.server_names == ["test_server"]  # Check for the actual server name
            assert mock_client.sessions["test_server"]["type"] == "sse"

    @pytest.mark.asyncio
    async def test_connect_to_single_server_websocket(self, mock_client, mock_session):
        """Test connecting to a WebSocket server"""
        with patch('mcpomni_connect.client.websocket_client') as mock_ws_client:
            mock_transport = (AsyncMock(), AsyncMock())
            mock_ws_client.return_value.__aenter__.return_value = mock_transport
            
            mock_client.exit_stack.enter_async_context = AsyncMock()
            mock_client.exit_stack.enter_async_context.side_effect = [
                mock_transport,
                mock_session
            ]

            server_info = {"name": "server3", "srv_config": MOCK_SERVER_CONFIG["mcpServers"]["server3"]}
            await mock_client._connect_to_single_server(server_info)

            assert mock_client.server_names == ["test_server"]  # Check for the actual server name
            assert mock_client.sessions["test_server"]["type"] == "websocket"

    @pytest.mark.asyncio
    async def test_clean_up_server(self, mock_client):
        """Test cleaning up server connections"""
        # Setup mock server session
        mock_read_stream = AsyncMock()
        mock_read_stream._closed = False
        mock_write_stream = AsyncMock()
        mock_write_stream._closed = False
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        
        mock_client.server_names = ["test_server"]
        mock_client.sessions = {
            "test_server": {
                "session": mock_session,
                "read_stream": mock_read_stream,
                "write_stream": mock_write_stream,
                "connected": True,
                "type": "stdio"
            }
        }

        await mock_client.clean_up_server()

        # Verify cleanup calls
        assert mock_write_stream.aclose.called
        assert mock_read_stream.aclose.called
        assert mock_session.close.called
        assert not mock_client.sessions["test_server"]["connected"]
        assert mock_client.sessions["test_server"]["session"] is None

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_client):
        """Test full client cleanup"""
        mock_client.clean_up_server = AsyncMock()
        mock_client.exit_stack.aclose = AsyncMock()

        await mock_client.cleanup()

        assert mock_client.clean_up_server.called
        assert mock_client.exit_stack.aclose.called
        assert len(mock_client.server_names) == 0
        assert len(mock_client.sessions) == 0
        assert len(mock_client.available_tools) == 0
        assert len(mock_client.available_resources) == 0
        assert len(mock_client.available_prompts) == 0

    @pytest.mark.asyncio
    async def test_cleanup_timeout(self, mock_client):
        """Test cleanup with timeout"""
        # Setup mock server session with timeout behavior
        mock_read_stream = AsyncMock()
        mock_read_stream._closed = False
        mock_write_stream = AsyncMock()
        mock_write_stream._closed = False
        mock_session = AsyncMock()
        
        # Instead of making all operations timeout, let's make only some timeout
        # to better simulate real behavior
        mock_write_stream.aclose = AsyncMock()  # This one succeeds
        mock_read_stream.aclose = AsyncMock(side_effect=asyncio.TimeoutError)  # This one times out
        mock_session.close = AsyncMock()  # This one succeeds
        
        mock_client.server_names = ["test_server"]
        mock_client.sessions = {
            "test_server": {
                "session": mock_session,
                "read_stream": mock_read_stream,
                "write_stream": mock_write_stream,
                "connected": True,
                "type": "stdio"
            }
        }

        # Mock the exit_stack.aclose
        mock_client.exit_stack.aclose = AsyncMock()
        
        # The cleanup should handle the TimeoutError and still proceed
        await mock_client.cleanup()
        
        # Verify that all stream operations were attempted
        assert mock_write_stream.aclose.called, "Write stream close was not called"
        assert mock_read_stream.aclose.called, "Read stream close was not called"
        assert mock_session.close.called, "Session close was not called"
        
        # Verify that exit_stack was closed
        assert mock_client.exit_stack.aclose.called, "Exit stack close was not called"
        
        # Verify that all collections were cleared
        assert len(mock_client.server_names) == 0, "Server names were not cleared"
        assert len(mock_client.sessions) == 0, "Sessions were not cleared"
        assert len(mock_client.available_tools) == 0, "Tools were not cleared"
        assert len(mock_client.available_resources) == 0, "Resources were not cleared"
        assert len(mock_client.available_prompts) == 0, "Prompts were not cleared"

    @pytest.mark.asyncio
    async def test_connect_to_servers_all_failed(self, mock_client):
        """Test behavior when all server connections fail"""
        mock_client._connect_to_single_server = AsyncMock(side_effect=Exception("Connection failed"))

        with pytest.raises(RuntimeError) as exc_info:
            await mock_client.connect_to_servers()
        
        assert "No servers could be connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_clean_up_server_error_handling(self, mock_client):
        """Test error handling during server cleanup"""
        # Setup mock server with problematic streams
        mock_read_stream = AsyncMock()
        mock_read_stream.aclose.side_effect = Exception("Read stream error")
        mock_write_stream = AsyncMock()
        mock_write_stream.aclose.side_effect = Exception("Write stream error")
        mock_session = AsyncMock()
        mock_session.close.side_effect = Exception("Session close error")

        mock_client.server_names = ["test_server"]
        mock_client.sessions = {
            "test_server": {
                "session": mock_session,
                "read_stream": mock_read_stream,
                "write_stream": mock_write_stream,
                "connected": True,
                "type": "stdio"
            }
        }

        # Should not raise exceptions
        await mock_client.clean_up_server()

        # Verify the server is marked as disconnected despite errors
        assert not mock_client.sessions["test_server"]["connected"]
        assert mock_client.sessions["test_server"]["session"] is None

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