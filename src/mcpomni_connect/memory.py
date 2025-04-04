import json
from mcpomni_connect.utils import logger, CLIENT_MAC_ADDRESS
import redis.asyncio as redis
import time
from typing import Optional
from decouple import config
import asyncio


class InMemoryShortTermMemory:
    """In memory short term memory."""
    def __init__(self, max_context_tokens: int = 30000, debug: bool = False) -> None:
        """Initialize."""
        self.max_context_tokens = max_context_tokens
        self.message_history = []
        self.debug = debug

    # add a message to the message history
    async def store_message(
        self, role: str, content: str, metadata: Optional[dict] = None
    ):
        """Add a message to the message history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": asyncio.get_running_loop().time(),
            "metadata": metadata or {},
        }
        self.message_history.append(message)
        if self.debug:
            logger.info(f"Added message to history: {role} - {content[:100]}")

    async def get_messages(self):
        """Get the message history"""
        return self.message_history

    async def show_history(self):
        """Show the message history"""
        for i, message in enumerate(self.message_history):
            logger.info(
                f"Message {i}: {message['role']} - {message['content']}"
            )

    async def clear_memory(self):
        """Clear the message history"""
        self.message_history = []
        if self.debug:
            logger.info("Message history cleared")

    async def save_message_history_to_file(self, file_path: str):
        """Save the message history to a file"""
        with open(file_path, "w") as f:
            for message in self.message_history:
                f.write(f"{message['role']}: {message['content']}\n")
        if self.debug:
            logger.info(f"Message history saved to {file_path}")

class RedisShortTermMemory:
    """Redis short term memory."""
    REDIS_HOST = config("REDIS_HOST", default="localhost")
    REDIS_PORT = config("REDIS_PORT", default=6379)
    REDIS_DB = config("REDIS_DB", default=0)
    def __init__(self, redis_client: Optional[redis.Redis] = None, max_context_tokens: int = 30000) -> None:
        """Initialize."""
        self._redis_client = redis_client or redis.Redis(
            host=self.REDIS_HOST, port=self.REDIS_PORT, db=self.REDIS_DB, decode_responses=True
        )
        self.SHORT_TERM_LIMIT = int(0.7 * max_context_tokens)
        self.client_id = CLIENT_MAC_ADDRESS
        logger.info(f"Initialized RedisShortTermMemory with client ID: {self.client_id}")

    async def store_message(self, role: str, content: str, metadata: dict = None):
        """Store a message in Redis with a timestamp using the client's MAC address as ID."""
        metadata = metadata or {}
        logger.info(f"Storing message for client {self.client_id}: {content}")

        key = f"mcp_memory:{self.client_id}"
        timestamp = time.time()

        message = {
            "role": role,
            "content": content,
            "metadata": self._serialize(metadata),
            "timestamp": timestamp,
        }

        # Store as a JSON string in Redis
        await self._redis_client.zadd(key, {json.dumps(message): timestamp})
        await self._redis_client.set(f"mcp_last_active:{self.client_id}", timestamp)
        
        # Enforce the short term limit
        await self.enforce_short_term_limit()

    async def get_messages(self):
        """Retrieve messages for this client using the MAC address as ID."""
        key = f"mcp_memory:{self.client_id}"
        # enforce short term limit before retrieving messages
        await self.enforce_short_term_limit()
        messages = await self._redis_client.zrange(key, 0, -1)

        # Deserialize messages and reconstruct tool calls if necessary
        return [self._deserialize(json.loads(msg)) for msg in messages]

    def _serialize(self, data):
        """Convert any non-serializable data into a JSON-compatible format."""
        try:
            return json.dumps(data, default=lambda o: o.__dict__)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            return json.dumps({"error": "Serialization failed"})

    def _deserialize(self, data):
        """Convert stored JSON strings back to their original format if needed."""
        try:
            if "metadata" in data:
                data["metadata"] = json.loads(data["metadata"])
            return data
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return data

    async def get_last_active(self):
        """Get last active timestamp for this client."""
        key = f"mcp_last_active:{self.client_id}"
        last_active = await self._redis_client.get(key)
        return float(last_active) if last_active else None

    async def enforce_short_term_limit(self):
        """Enforce short term limit on the number of tokens in the context window."""
        key = f"mcp_memory:{self.client_id}"
        messages = await self._redis_client.zrange(key, 0, -1, withscores=True)

        total_tokens = sum(len(msg[0].split()) for msg in messages)
        while total_tokens > self.SHORT_TERM_LIMIT and messages:
            oldest_msg = messages.pop(0)
            await self._redis_client.zrem(key, oldest_msg[0])
            total_tokens = sum(len(msg[0].split()) for msg in messages)
            
        logger.debug(f"Enforced short term limit: {total_tokens}/{self.SHORT_TERM_LIMIT} tokens")

    async def clear_memory(self):
        """Clear the memory."""
        key = f"mcp_memory:{self.client_id}"
        await self._redis_client.delete(key)
        logger.info(f"Cleared memory for client {self.client_id}")

    async def save_message_history_to_file(self, file_path: str):
        """Save the message history to a file."""
        key = f"mcp_memory:{self.client_id}"
        messages = await self._redis_client.zrange(key, 0, -1)
        messages = [self._deserialize(json.loads(msg)) for msg in messages]
        with open(file_path, "w") as f:
            for message in messages:
                f.write(f"{message['role']}: {message['content']}\n")
        logger.info(f"Saved message history to {file_path}")