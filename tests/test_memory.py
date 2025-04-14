import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from mcpomni_connect.memory import InMemoryShortTermMemory, RedisShortTermMemory

@pytest.mark.asyncio
class TestInMemoryShortTermMemory:

    # This fixture resets the class-level message history before each test
    @pytest.fixture(autouse=True)
    def reset_message_history(self):
        """Reset the shared message_history before each test."""
        InMemoryShortTermMemory.message_history = []
        yield
        InMemoryShortTermMemory.message_history = []

    async def test_store_and_get_messages(self):
        memory = InMemoryShortTermMemory(max_context_tokens=100)
        await memory.store_message("user", "Hello world")
        await memory.store_message("assistant", "Hi there!")

        messages = await memory.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    async def test_truncate_message_history(self):
        memory = InMemoryShortTermMemory(max_context_tokens=2)
        await memory.store_message("user", "one two")
        await memory.store_message("user", "three four")
        await memory.store_message("user", "five six")

        messages = await memory.get_messages()
        assert len(messages) == 1

    async def test_clear_memory(self):
        memory = InMemoryShortTermMemory()
        await memory.store_message("user", "test")
        cleared = await memory.clear_memory()
        assert len(cleared) == 1
        assert await memory.get_messages() == []

@pytest.mark.asyncio
class TestRedisShortTermMemory:

    @pytest.fixture
    def mock_redis(self):
        redis_mock = AsyncMock()
        redis_mock.zadd = AsyncMock()
        redis_mock.set = AsyncMock()
        redis_mock.zrange = AsyncMock(return_value=[])
        redis_mock.delete = AsyncMock()
        redis_mock.get = AsyncMock(return_value="1680000000.0")
        redis_mock.zrem = AsyncMock()
        return redis_mock

    async def test_store_message(self, mock_redis):
        memory = RedisShortTermMemory(redis_client=mock_redis)
        await memory.store_message("user", "hello", metadata={"intent": "greeting"})

        assert mock_redis.zadd.called
        assert mock_redis.set.called

    async def test_get_messages(self, mock_redis):
        stored_msg = json.dumps({
            "role": "user",
            "content": "Hello from Redis!",
            "metadata": json.dumps({"intent": "greet"}),
            "timestamp": 1680000000.0,
        })
        mock_redis.zrange.return_value = [stored_msg]
        memory = RedisShortTermMemory(redis_client=mock_redis)
        messages = await memory.get_messages()

        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        assert messages[0]["metadata"]["intent"] == "greet"

    async def test_get_last_active(self, mock_redis):
        memory = RedisShortTermMemory(redis_client=mock_redis)
        last_active = await memory.get_last_active()
        assert isinstance(last_active, float)
        assert last_active == 1680000000.0

    async def test_clear_memory(self, mock_redis):
        memory = RedisShortTermMemory(redis_client=mock_redis)
        await memory.clear_memory()
        mock_redis.delete.assert_called_once()

    async def test_enforce_short_term_limit(self, mock_redis):
        messages = [
            (json.dumps({"role": "user", "content": "word " * 200}), 1680000000.0),
            (json.dumps({"role": "assistant", "content": "another " * 200}), 1680000001.0),
        ]
        mock_redis.zrange.return_value = messages
        memory = RedisShortTermMemory(redis_client=mock_redis, max_context_tokens=100)
        await memory.enforce_short_term_limit()
        assert mock_redis.zrem.called
