import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI, AzureOpenAI
from mcpomni_connect.utils import logger, dict_to_namespace
import requests

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class LLMConnection:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.llm_config = None
        self.openai = OpenAI(api_key=self.config.llm_api_key)
        self.groq = Groq(api_key=self.config.llm_api_key)
        self.openrouter = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config.llm_api_key,
        )
        self.gemini = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=self.config.llm_api_key,
        )
        self.deepseek = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=self.config.llm_api_key,
        )
        self.anthropic = OpenAI(
            base_url="https://api.anthropic.com/v1/",
            api_key=self.config.llm_api_key,
        )
        self.ollama = None
        self.ollama_host = OLLAMA_HOST
        self.azure_openai = None
        if not self.llm_config:
            logger.info("updating llm configuration")
            self.llm_configuration()
            logger.info(f"LLM configuration: {self.llm_config}")

            if self.llm_config and self.llm_config["provider"].lower() == "azureopenai":
                azure_endpoint = self.llm_config.get("azure_endpoint")
                azure_api_version = self.llm_config.get(
                    "azure_api_version", "2024-02-01"
                )

                if not azure_endpoint:
                    logger.error("Azure OpenAI endpoint not provided in configuration")
                else:
                    self.azure_openai = AzureOpenAI(
                        api_key=self.config.llm_api_key,
                        api_version=azure_api_version,
                        azure_endpoint=azure_endpoint,
                    )

    def llm_configuration(self):
        """Load the LLM configuration"""
        llm_config = self.config.load_config("servers_config.json")["LLM"]
        try:
            provider = llm_config.get("provider", "openai")
            model = llm_config.get("model", "gpt-4o-mini")
            temperature = llm_config.get("temperature", 0.5)
            max_tokens = llm_config.get("max_tokens", 5000)
            top_p = llm_config.get("top_p", 0)

            self.llm_config = {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            # Add Azure OpenAI specific configuration if provider is azureopenai
            if provider.lower() == "azureopenai":
                azure_endpoint = llm_config.get("azure_endpoint")
                azure_api_version = llm_config.get("azure_api_version", "2024-02-01")
                azure_deployment = llm_config.get("azure_deployment")

                if not azure_endpoint:
                    logger.error("Azure OpenAI endpoint not provided in configuration")

                if not azure_deployment:
                    logger.warning(
                        "Azure deployment name not provided, using model name as deployment"
                    )
                    azure_deployment = model

                self.llm_config.update(
                    {
                        "azure_endpoint": azure_endpoint,
                        "azure_api_version": azure_api_version,
                        "azure_deployment": azure_deployment,
                    }
                )
            # Add Ollama specific configuration if provider is ollama
            if provider.lower() == "ollama":
                ollama_host = llm_config.get("ollama_host", OLLAMA_HOST)
                self.llm_config["ollama_host"] = ollama_host

            return self.llm_config
        except Exception as e:
            logger.error(f"Error loading LLM configuration: {e}")
            return None

    async def llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] = None,
    ):
        """Call the LLM"""
        try:
            if self.llm_config["provider"].lower() == "openai":
                if tools:
                    response = self.openai.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    response = self.openai.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                    )
                return response
            elif self.llm_config["provider"].lower() == "anthropic":
                if tools:
                    response = self.anthropic.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    response = self.anthropic.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                    )
                return response
            elif self.llm_config["provider"].lower() == "groq":
                # messages = self.truncate_messages_for_groq(messages)
                if tools:
                    response = self.groq.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    response = self.groq.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                    )
                return response
            elif self.llm_config["provider"].lower() == "openrouter":
                if tools:
                    response = self.openrouter.chat.completions.create(
                        extra_body={
                            "order": ["openai", "anthropic", "groq"],
                            "allow_fallback": True,
                            "require_provider": True,
                        },
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    response = self.openrouter.chat.completions.create(
                        extra_body={
                            "order": ["Mistral", "Openai", "Groq", "Gemini"],
                            "allow_fallback": True,
                            "require_provider": True,
                        },
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        stop=["\n\nObservation:"],
                    )
                return response
            elif self.llm_config["provider"].lower() == "gemini":
                if tools:
                    response = self.gemini.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    response = self.gemini.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                    )
                return response
            elif self.llm_config["provider"].lower() == "deepseek":
                if tools:
                    response = self.deepseek.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    response = self.deepseek.chat.completions.create(
                        model=self.llm_config["model"],
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                    )
                return response
            elif self.llm_config["provider"].lower() == "azureopenai":
                # Check if Azure OpenAI client is initialized
                if not self.azure_openai:
                    logger.error("Azure OpenAI client not initialized")
                    return None

                deployment = self.llm_config.get("azure_deployment")
                if not deployment:
                    logger.warning("Azure deployment not specified, using model name")
                    deployment = self.llm_config["model"]

                if tools:
                    response = self.azure_openai.chat.completions.create(
                        model=deployment,  # For Azure, this is the deployment name
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    response = self.azure_openai.chat.completions.create(
                        model=deployment,  # For Azure, this is the deployment name
                        max_tokens=self.llm_config["max_tokens"],
                        temperature=self.llm_config["temperature"],
                        top_p=self.llm_config["top_p"],
                        messages=messages,
                    )
                return response
            elif self.llm_config["provider"].lower() == "ollama":
                ollama_host = self.llm_config.get("ollama_host", self.ollama_host)

                if not ollama_host:
                    logger.error("Ollama host not specified")
                    return None

                # Normalize host
                if not ollama_host.startswith("http"):
                    ollama_host = f"http://{ollama_host}"
                ollama_host = ollama_host.rstrip("/")  # no trailing slash

                # Confirm server is running
                # try:
                #     models_response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                #     models_response.raise_for_status()
                #     logger.info(f"Ollama models: {models_response.json()}")
                # except Exception as e:
                #     logger.error(f"Failed to connect to Ollama server at {ollama_host}: {e}")
                #     return None

                # Convert messages to prompt
                def messages_to_prompt(messages):
                    prompt_lines = []
                    for m in messages:
                        role = (
                            m.role if hasattr(m, "role") else m.get("role", "unknown")
                        )
                        content = (
                            m.content if hasattr(m, "content") else m.get("content", "")
                        )
                        prompt_lines.append(f"{role.capitalize()}: {content}")
                    return "\n".join(prompt_lines)

                # Use /api/generate (prompt-based models)
                payload = {
                    "model": self.llm_config["model"],
                    "prompt": messages_to_prompt(messages=messages),
                    "stream": False,
                    "options": {
                        "temperature": self.llm_config["temperature"],
                        "num_predict": self.llm_config["max_tokens"],
                        "top_p": self.llm_config["top_p"],
                    },
                }

                # Add tools if supported (Ollama doesn't yet)
                if tools:
                    payload["tools"] = tools  # ignored by Ollama right now

                try:
                    response = requests.post(
                        f"{ollama_host}/api/generate", json=payload, timeout=120
                    )
                    response.raise_for_status()
                    ollama_data = response.json()
                except Exception as e:
                    logger.error(f"Error calling Ollama API: {e}")
                    return None

                formatted_response = {
                    "id": ollama_data.get("id", ""),
                    "object": "chat.completion",
                    "created": ollama_data.get("created_at", 0),
                    "model": self.llm_config["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": ollama_data.get("response", ""),
                                "tool_calls": ollama_data.get("message", {}).get(
                                    "tool_calls", []
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": ollama_data.get("prompt_eval_count", 0),
                        "completion_tokens": ollama_data.get("eval_count", 0),
                        "total_tokens": ollama_data.get("prompt_eval_count", 0)
                        + ollama_data.get("eval_count", 0),
                    },
                }
                # Convert to OpenAI SDK-like structure
                return dict_to_namespace(formatted_response)
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None

    def truncate_messages_for_groq(self, messages):
        """Truncate messages to stay within Groq's token limits (5000 total)."""
        if not messages:
            return messages

        truncated_messages = []
        total_tokens = 0
        SYSTEM_PROMPT_LIMIT = 1000  # Max tokens for system prompt
        MESSAGE_LIMIT = 500  # Max tokens per message
        TOTAL_LIMIT = 10000  # Total token limit

        # Handle system prompt first
        system_msg = messages[0]
        if len(system_msg["content"]) > SYSTEM_PROMPT_LIMIT:
            logger.info("Truncating system prompt to 1000 tokens")
            system_msg["content"] = system_msg["content"][:SYSTEM_PROMPT_LIMIT]
        truncated_messages.append(system_msg)
        total_tokens += len(system_msg["content"])

        # Process remaining messages, ensuring recent messages are prioritized
        remaining_budget = TOTAL_LIMIT - total_tokens
        logger.debug(f"Remaining Budget for tokens: {remaining_budget}")
        for i, msg in enumerate(messages[1:]):
            if total_tokens >= TOTAL_LIMIT:
                break

            msg_length = len(msg["content"])

            # Keep first 10 messages as they are
            if i < 10:
                truncated_messages.append(msg)
                total_tokens += msg_length
                continue

            # Truncate only if message exceeds 500 characters
            if msg_length > MESSAGE_LIMIT:
                logger.info(f"Truncating message to {MESSAGE_LIMIT} tokens")
                msg["content"] = msg["content"][:MESSAGE_LIMIT]
                msg_length = MESSAGE_LIMIT

            # Ensure messages are added even if total budget is exceeded
            if total_tokens + msg_length > TOTAL_LIMIT:
                msg["content"] = msg["content"][: max(0, TOTAL_LIMIT - total_tokens)]
                if msg["content"]:  # Only add if there's remaining content
                    truncated_messages.append(msg)
                    total_tokens += len(msg["content"])
                break
            else:
                truncated_messages.append(msg)
                total_tokens += msg_length

        logger.info(
            f"Final message count: {len(truncated_messages)}, Total tokens: {total_tokens}"
        )
        logger.info(f"Truncated messages: {truncated_messages}")
        return truncated_messages
