import json
from mcpomni_connect.utils import (
    logger,
    CLIENT_MAC_ADDRESS,
)
import redis.asyncio as redis
import time
from typing import Optional
from decouple import config
import asyncio

# TODO: Add QDRANT DB episodic memory
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import Distance, VectorParams


class InMemoryShortTermMemory:
    """In memory short term memory with support for both single and multi-agent scenarios."""

    # Class-level storage for multi-agent mode
    multi_agent_history = {}
    single_agent_history = []

    def __init__(
        self,
        max_context_tokens: int = 30000,
        debug: bool = False,
        multi_agent: bool = False,
    ) -> None:
        """Initialize memory storage.

        Args:
            max_context_tokens: Maximum tokens to keep in memory
            debug: Enable debug logging
            multi_agent: Whether to use multi-agent memory structure
        """
        self.max_context_tokens = max_context_tokens
        self.debug = debug
        self.multi_agent = multi_agent
        self.short_term_limit = int(0.7 * max_context_tokens)

    async def truncate_message_history(self, agent_name: str = None):
        """Truncate the message history to the max context tokens."""
        try:
            if self.multi_agent:
                if agent_name not in self.multi_agent_history:
                    self.multi_agent_history[agent_name] = []
                    return
                messages = self.multi_agent_history[agent_name]
            else:
                messages = self.single_agent_history

            # Calculate total tokens
            total_tokens = sum(len(str(msg["content"]).split()) for msg in messages)

            # Remove oldest messages until under token limit
            while total_tokens > self.short_term_limit and messages:
                messages.pop(0)
                total_tokens = sum(len(str(msg["content"]).split()) for msg in messages)

        except Exception as e:
            logger.error(f"Failed to truncate message history: {e}")
            if self.multi_agent:
                self.multi_agent_history[agent_name] = []
            else:
                self.single_agent_history = []

    async def store_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
        agent_name: str = None,
    ):
        """Store a message in memory.

        Args:
            role: Message role (e.g., 'user', 'assistant')
            content: Message content
            agent_name: Name of agent (required for multi-agent mode)
            metadata: Optional metadata about the message
        """
        try:
            if self.multi_agent and not agent_name:
                raise ValueError("agent_name is required in multi-agent mode")

            message = {
                "role": role,
                "content": content,
                "timestamp": asyncio.get_running_loop().time(),
                "metadata": metadata or {},
            }

            if self.multi_agent:
                if agent_name not in self.multi_agent_history:
                    self.multi_agent_history[agent_name] = []
                self.multi_agent_history[agent_name].append(message)
            else:
                self.single_agent_history.append(message)

        except Exception as e:
            logger.error(f"Failed to store message: {e}")

    async def get_messages(self, agent_name: str = None):
        """Get messages from memory.

        Args:
            agent_name: Name of agent (required for multi-agent mode)

        Returns:
            List of messages
        """
        try:
            if self.multi_agent:
                if not agent_name:
                    raise ValueError("agent_name is required in multi-agent mode")
                if agent_name not in self.multi_agent_history:
                    self.multi_agent_history[agent_name] = []
                await self.truncate_message_history(agent_name)
                return self.multi_agent_history[agent_name]
            else:
                await self.truncate_message_history()
                return self.single_agent_history

        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []

    async def clear_memory(self, agent_name: str = None):
        """Clear memory for an agent or all memory.

        Args:
            agent_name: Name of agent to clear (required for multi-agent mode)
        """
        try:
            if self.multi_agent:
                if not agent_name:
                    raise ValueError("agent_name is required in multi-agent mode")
                if agent_name in self.multi_agent_history:
                    del self.multi_agent_history[agent_name]
            else:
                self.single_agent_history = []

        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    async def save_message_history_to_file(
        self, file_path: str, agent_name: str = None
    ):
        """Save message history to a file, appending to existing content."""
        try:
            with open(file_path, "a") as f:
                # Add separator if file has content
                if f.tell() > 0:
                    f.write("\n\n")

                if self.multi_agent:
                    if agent_name:
                        messages = self.multi_agent_history.get(agent_name, [])
                        if messages:
                            f.write(f"Agent: {agent_name}\n")
                            for message in messages:
                                f.write(f"{message['role']}: {message['content']}\n")
                    else:
                        logger.info(
                            f"Saving multi-agent messages for all agents: {self.multi_agent_history}"
                        )
                        for (
                            agent,
                            messages,
                        ) in self.multi_agent_history.items():
                            if messages:
                                logger.info(
                                    f"Saving multi-agent messages for agent: {agent}"
                                )
                                f.write(f"Agent: {agent}\n")
                                for message in messages:
                                    f.write(
                                        f"{message['role']}: {message['content']}\n"
                                    )
                                f.write("\n")
                else:
                    for message in self.single_agent_history:
                        f.write(f"{message['role']}: {message['content']}\n")

            if self.debug:
                logger.info(f"Message history saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save message history: {e}")
            raise

    async def load_message_history_from_file(
        self, file_path: str, agent_name: str = None
    ):
        """Load message history from a file and store in in memory short term memory."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

                if "Agent:" in content:
                    # Multi-agent format
                    sections = content.split("Agent:")
                    for section in sections[1:]:
                        lines = section.strip().split("\n")
                        current_agent = lines[0].strip()
                        messages = lines[1:]
                        for msg in messages:
                            if ":" in msg:
                                role, content = msg.split(":", 1)
                                # update the multi-agent to be true
                                self.multi_agent = True
                                await self.store_message(
                                    agent_name=current_agent,
                                    role=role.strip(),
                                    content=content.strip(),
                                )
                else:
                    # Single-agent format
                    messages = content.strip().split("\n")
                    for msg in messages:
                        if ":" in msg:
                            role, content = msg.split(":", 1)
                            await self.store_message(
                                agent_name=agent_name,
                                role=role.strip(),
                                content=content.strip(),
                            )

            if self.debug:
                logger.info(f"Successfully loaded message history from {file_path}")
                if self.multi_agent:
                    logger.info(
                        f"Loaded messages for agents: {list(self.multi_agent_history.keys())}"
                    )
                else:
                    logger.info(f"Loaded {len(self.single_agent_history)} messages")

        except Exception as e:
            logger.error(f"Failed to load message history from file: {e}")


class RedisShortTermMemory:
    """Redis short term memory."""

    REDIS_HOST = config("REDIS_HOST", default="localhost")
    REDIS_PORT = config("REDIS_PORT", default=6379)
    REDIS_DB = config("REDIS_DB", default=0)

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        max_context_tokens: int = 30000,
    ) -> None:
        """Initialize."""
        self._redis_client = redis_client or redis.Redis(
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
            db=self.REDIS_DB,
            decode_responses=True,
        )
        self.SHORT_TERM_LIMIT = int(0.7 * max_context_tokens)
        self.client_id = CLIENT_MAC_ADDRESS
        self.in_memory_short_term_memory = InMemoryShortTermMemory(
            max_context_tokens=max_context_tokens
        )
        logger.info(
            f"Initialized RedisShortTermMemory with client ID: {self.client_id}"
        )

    async def store_message(self, role: str, content: str, metadata: dict = None):
        """Store a message in Redis with a timestamp using the client's MAC address as ID."""
        metadata = metadata or {}
        logger.info(f"Storing message for client {self.client_id}: {content}")

        key = f"mcp_memory:{self.client_id}"
        timestamp = time.time()

        message = {
            "role": role,
            "content": str(content),
            "metadata": self._serialize(metadata),
            "timestamp": timestamp,
        }

        # Store as a JSON string in Redis
        await self._redis_client.zadd(key, {json.dumps(message): timestamp})
        await self._redis_client.set(f"mcp_last_active:{self.client_id}", timestamp)
        # store to the in memory to act as current working memory which will be use for episodic memory
        await self.in_memory_short_term_memory.store_message(role, content, metadata)
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

        logger.debug(
            f"Enforced short term limit: {total_tokens}/{self.SHORT_TERM_LIMIT} tokens"
        )

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


# class ChromaDBMemory:
#     def __init__(self, name: str, description: str):
#         self.chroma_client = chromadb.HttpClient(host="localhost", port=8000)
#         self.collection = self._get_or_create_collection(name, description)

#     def _get_or_create_collection(self, name: str, description: str):
#         """Get or create a collection with default embedding model."""
#         try:

#             logger.info(f"Getting or creating collection: {name}")
#             collection = self.chroma_client.get_or_create_collection(
#                 name=name,
#                 metadata={
#                     "hnsw:space": "cosine",
#                     "description": description

#                 }
#             )
#             logger.info("Successfully initialized ChromaDB collection")
#             return collection
#         except Exception as e:
#             logger.error(f"Failed to initialize ChromaDB collection: {e}")
#             raise

#     def add_to_collection(self, documents: list[str], metadatas: list[dict] = None, ids: list[str] = None):
#         """Add documents to the collection using ChromaDB's internal embedding model.

#         Args:
#             collection: The ChromaDB collection
#             documents: List of text documents to store
#             metadatas: Optional list of metadata dictionaries
#             ids: Optional list of document IDs
#         """
#         try:
#             # Add documents to ChromaDB
#             self.collection.add(
#                 documents=documents,
#                 metadatas=metadatas if metadatas else None,
#                 ids=ids if ids else None
#             )
#             logger.info(f"Successfully added {len(documents)} documents to ChromaDB")
#         except Exception as e:
#             logger.error(f"Failed to add documents to ChromaDB: {e}")
#             raise

#     def query_collection(self, query: str, n_results: int = 5):
#         """Query the collection using ChromaDB's internal embedding model.

#         Args:
#             collection: The ChromaDB collection
#             query: The query text
#             n_results: Number of results to return

#         Returns:
#             dict: Query results containing documents, distances, and metadata
#         """
#         try:
#             results = self.collection.query(
#                 query_texts=[query],
#                 n_results=n_results
#             )
#             logger.debug(f"Retrieved {len(results['documents'][0])} results from ChromaDB")
#             return results
#         except Exception as e:
#             logger.error(f"Failed to query ChromaDB: {e}")
#             raise

#     def delete_from_collection(self, ids: list[str] = None, where: dict = None):
#         """Delete documents from the collection.

#         Args:
#             collection: The ChromaDB collection
#             ids: Optional list of document IDs to delete
#             where: Optional metadata filter for deletion
#         """
#         try:
#             self.collection.delete(
#                 ids=ids if ids else None,
#                 where=where if where else None
#             )
#             logger.info(f"Successfully deleted documents from ChromaDB")
#         except Exception as e:
#             logger.error(f"Failed to delete documents from ChromaDB: {e}")
#             raise

#     def update_collection(self, documents: list[str], metadatas: list[dict] = None, ids: list[str] = None):
#         """Update documents in the collection using ChromaDB's internal embedding model.

#         Args:
#             collection: The ChromaDB collection
#             documents: List of text documents to update
#             metadatas: Optional list of metadata dictionaries
#             ids: List of document IDs to update (required)
#         """
#         if not ids:
#             raise ValueError("IDs are required for updating documents")

#         try:
#             # Update documents in ChromaDB
#             self.collection.update(
#                 documents=documents,
#                 metadatas=metadatas if metadatas else None,
#                 ids=ids
#             )
#             logger.info(f"Successfully updated {len(documents)} documents in ChromaDB")
#         except Exception as e:
#             logger.error(f"Failed to update documents in ChromaDB: {e}")
#             raise


# class QdrantMemory:
#     def __init__(self, name: str, description: str):
#         """Initialize Qdrant memory storage.

#         Args:
#             name: Name of the collection
#             description: Description of the collection
#         """
#         self.client = QdrantClient(host=config("QDRANT_HOST", default="localhost"), port=config("QDRANT_PORT", default=6333))
#         self.collection_name = name
#         self.description = description
#         self._ensure_collection()

#     def _ensure_collection(self):
#         """Ensure the collection exists, create if it doesn't."""
#         try:
#             collections = self.client.get_collections().collections
#             collection_names = [collection.name for collection in collections]

#             if self.collection_name not in collection_names:
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
#                 )
#                 logger.info(f"Created new Qdrant collection: {self.collection_name}")
#             else:
#                 logger.info(f"Using existing Qdrant collection: {self.collection_name}")
#         except Exception as e:
#             logger.error(f"Failed to initialize Qdrant collection: {e}")
#             raise

#     def add_to_collection(self, documents: List[str], conversation: str, metadatas: List[Dict] = None, ids: List[str] = None):
#         """Add documents to the collection.

#         Args:
#             documents: List of text documents to store
#             metadatas: Optional list of metadata dictionaries
#             ids: Optional list of document IDs
#         """
#         try:
#             if not ids:
#                 ids = [str(uuid.uuid4()) for _ in documents]

#             # Convert documents to points
#             points = []
#             for i, (doc, doc_id) in enumerate(zip(documents, ids)):
#                 metadata = metadatas[i] if metadatas else {}
#                 metadata["text"] = doc
#                 metadata["previous_conversation"] = conversation
#                 metadata["timestamp"] = str(datetime.now())

#                 points.append(models.PointStruct(
#                     id=doc_id,
#                     vector=embed_text(doc),
#                     payload=metadata
#                 ))

#             # Upsert points to collection
#             self.client.upsert(
#                 collection_name=self.collection_name,
#                 points=points
#             )
#             logger.info(f"Successfully added {len(documents)} documents to Qdrant")
#         except Exception as e:
#             logger.error(f"Failed to add documents to Qdrant: {e}")
#             raise

#     def query_collection(self, query: str, n_results: int = 5, distance_threshold: float = 0.70) -> Dict[str, Any]:
#         """Query the collection.

#         Args:
#             query: The query text
#             n_results: Number of results to return

#         Returns:
#             Dict containing query results
#         """
#         try:
#             # Search for similar documents
#             search_result = self.client.query_points(
#                 collection_name=self.collection_name,
#                 query=embed_text(query),
#                 limit=n_results,
#                 with_payload=True
#             ).points
#             if hasattr(search_result[0], "payload"):
#                 #logger.info(f"Search result: {search_result}")
#                 # format the results and filter by distance threshold if its greater than or equal to the threshold
#                 results = {
#                     "documents": [[hit.payload["text"] for hit in search_result if hit.score >= distance_threshold]],
#                     "previous_conversation": [[hit.payload["previous_conversation"] for hit in search_result if hit.score >= distance_threshold]],
#                     "distances": [[hit.score for hit in search_result if hit.score >= distance_threshold]],
#                     "metadatas": [[hit.payload for hit in search_result if hit.score >= distance_threshold]]
#                 }
#                 #logger.info(f" results distances: {results['distances']}")
#                 logger.info(f"Retrieved {len(results['documents'])} results from Qdrant")
#                 return results
#             else:
#                 logger.error(f"Failed to retrieve results from Qdrant: {search_result}")
#                 raise Exception(f"Failed to retrieve results from Qdrant: {search_result}")
#         except Exception as e:
#             logger.error(f"Failed to query Qdrant: {e}")
#             raise

#     def delete_from_collection(self, ids: List[str] = None, where: Dict = None):
#         """Delete documents from the collection.

#         Args:
#             ids: Optional list of document IDs to delete
#             where: Optional filter for deletion
#         """
#         try:
#             if ids:
#                 self.client.delete(
#                     collection_name=self.collection_name,
#                     points_selector=models.PointIdsList(
#                         points=ids
#                     )
#                 )
#             elif where:
#                 self.client.delete(
#                     collection_name=self.collection_name,
#                     points_selector=models.FilterSelector(
#                         filter=models.Filter(
#                             must=[
#                                 models.FieldCondition(
#                                     key=key,
#                                     match=models.MatchValue(value=value)
#                                 )
#                                 for key, value in where.items()
#                             ]
#                         )
#                     )
#                 )
#             logger.info("Successfully deleted documents from Qdrant")
#         except Exception as e:
#             logger.error(f"Failed to delete documents from Qdrant: {e}")
#             raise

#     def update_collection(self, documents: List[str], conversation: str, metadatas: List[Dict] = None, ids: List[str] = None):
#         """Update documents in the collection.

#         Args:
#             documents: List of text documents to update
#             metadatas: Optional list of metadata dictionaries
#             ids: List of document IDs to update (required)
#         """
#         if not ids:
#             raise ValueError("IDs are required for updating documents")

#         try:
#             # Convert documents to points
#             points = []
#             for i, (doc, doc_id) in enumerate(zip(documents, ids)):
#                 metadata = metadatas[i] if metadatas else {}
#                 metadata["text"] = doc
#                 metadata["previous_conversation"] = conversation
#                 metadata["timestamp"] = str(datetime.now())
#                 logger.info(f"embedding text: {embed_text(doc)}")
#                 points.append(models.PointStruct(
#                     id=doc_id,
#                     vector=embed_text(doc),
#                     payload=metadata
#                 ))

#             # Upsert points to collection
#             self.client.upsert(
#                 collection_name=self.collection_name,
#                 points=points
#             )
#             logger.info(f"Successfully updated {len(documents)} documents in Qdrant")
#         except Exception as e:
#             logger.error(f"Failed to update documents in Qdrant: {e}")
#             raise

# class EpisodicMemory(QdrantMemory):
#     def __init__(self, name: str, description: str):
#         """Initialize episodic memory using Qdrant storage.

#         Args:
#             name: Name of the collection
#             description: Description of the collection
#         """
#         super().__init__(name, description)
#         self.EPISODIC_MEMORY_PROMPT = EPISODIC_MEMORY_PROMPT

#     async def create_episodic_memory(self, messages: List[Dict], llm_connection: Callable) -> Dict:
#         """Create an episodic memory from a conversation.

#         Args:
#             messages: The conversation messages to analyze
#             llm_connection: The LLM connection to use for memory creation

#         Returns:
#             Dict: The created memory
#         """
#         try:
#             llm_messages = []
#             llm_messages.append({"role": "system", "content": self.EPISODIC_MEMORY_PROMPT})
#             llm_messages.append({"role": "user", "content": str(messages)})
#             response = await llm_connection.llm_call(llm_messages)
#             if response and response.choices:
#                 logger.info(f"response: {response.choices[0].message.content}")
#                 memory = clean_json_response(response.choices[0].message.content)

#                 # Store the memory in Qdrant
#                 self.add_to_collection(
#                     documents=[memory],
#                     metadatas=[{
#                         "type": "episodic_memory",
#                     }],
#                     conversation=str(messages),
#                     ids=[str(uuid.uuid4())]
#                 )

#                 logger.debug(f"Successfully created episodic memory: {memory}")
#                 return memory

#             return None
#         except Exception as e:
#             logger.error(f"Failed to create episodic memory: {e}")
#             return None

#     async def retrieve_relevant_memories(self, query: str, n_results: int = 5) -> List[Dict]:
#         """Retrieve relevant episodic memories based on a query.

#         Args:
#             query: The query to search for relevant memories
#             n_results: Number of memories to retrieve

#         Returns:
#             List[Dict]: List of relevant memories
#         """
#         try:
#             final_results = []
#             results = self.query_collection(query, n_results)
#             if results and "documents" in results:
#                 documents = results.get("documents", [])
#                 # Flatten nested lists defensively
#                 flat_docs = []
#                 for item in documents:
#                     if isinstance(item, list):
#                         flat_docs.extend(item)
#                     else:
#                         flat_docs.append(item)

#                 # Parse each document safely
#                 for i, doc in enumerate(flat_docs):
#                     try:
#                         final_results.append(json.loads(doc))
#                     except (TypeError, json.JSONDecodeError) as e:
#                         logger.warning(f"Failed to parse document at index {i}: {doc} — Error: {e}")
#             logger.debug(f"length of final results: {len(final_results)}")
#             return final_results
#         except Exception as e:
#             logger.error(f"Failed to retrieve episodic memories: {e}")
#             return []
