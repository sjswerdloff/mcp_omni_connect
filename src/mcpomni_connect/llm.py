
from openai import OpenAI
from groq import Groq
from typing import Any
from mcpomni_connect.utils import logger


class LLMConnection:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.llm_config = None
        self.openai = OpenAI(api_key=self.config.openai_api_key)
        self.groq = Groq(api_key=self.config.groq_api_key)
        if not self.llm_config:
            logger.info("updating llm configuration")
            self.llm_configuration()
            logger.info(f"LLM configuration: {self.llm_config}")

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
                "top_p": top_p
            }
            return self.llm_config
        except Exception as e:
            logger.error(f"Error loading LLM configuration: {e}")
            return None
    
    async def llm_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] = None):
        """Call the LLM"""
        if self.llm_config["provider"].lower() == "openai":
            if tools:
                response = self.openai.chat.completions.create(
                    model=self.llm_config["model"],
                    max_tokens=self.llm_config["max_tokens"],
                    temperature=self.llm_config["temperature"],
                    top_p=self.llm_config["top_p"],
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            else:
                response = self.openai.chat.completions.create(
                    model=self.llm_config["model"],
                    max_tokens=self.llm_config["max_tokens"],
                    temperature=self.llm_config["temperature"],
                    top_p=self.llm_config["top_p"],
                    messages=messages
                )
            return response
        elif self.llm_config["provider"].lower() == "groq":
            messages = self.truncate_messages_for_groq(messages)
            if tools:
                response = self.groq.chat.completions.create(
                    model=self.llm_config["model"],
                    max_tokens=self.llm_config["max_tokens"],
                    temperature=self.llm_config["temperature"],
                    top_p=self.llm_config["top_p"],
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            else:
                response = self.groq.chat.completions.create(
                    model=self.llm_config["model"],
                    max_tokens=self.llm_config["max_tokens"],
                    temperature=self.llm_config["temperature"],
                    top_p=self.llm_config["top_p"],
                    messages=messages
                )
            return response

    def truncate_messages_for_groq(self, messages):
        """Truncate messages to stay within Groq's token limits (10000 total)"""
        if not messages:
            return messages
            
        truncated_messages = []
        total_tokens = 0
        SYSTEM_PROMPT_LIMIT = 8000  # Max tokens for system prompt
        MESSAGE_LIMIT = 500  # Max tokens per message
        TOTAL_LIMIT = 10000  # Total token limit
        
        # First handle system prompt (first message)
        if messages:
            system_msg = messages[0]
            if len(system_msg["content"]) > SYSTEM_PROMPT_LIMIT:
                logger.info("Truncating system prompt to 1200 tokens")
                system_msg["content"] = system_msg["content"][:SYSTEM_PROMPT_LIMIT]
            truncated_messages.append(system_msg)
            total_tokens += len(system_msg["content"])
        
        # Calculate remaining token budget
        remaining_budget = TOTAL_LIMIT - total_tokens
        
        # Process remaining messages with recent messages first
        for i,msg in enumerate(messages[1:]):
            # If we've used up our budget, stop
            if total_tokens >= TOTAL_LIMIT:
                break
                
            # we dont tuncate the first 10 messages the user query and the assistant response
            if i < 10:
                truncated_messages.append(msg)
                total_tokens += len(msg["content"])
                continue
            else:
                # Truncate message if needed
                if len(msg["content"]) > MESSAGE_LIMIT:
                    logger.info(f"Truncating message to {MESSAGE_LIMIT} tokens")
                    msg["content"] = msg["content"][:MESSAGE_LIMIT]
                
                # Check if adding this message would exceed our budget
                if total_tokens + len(msg["content"]) <= TOTAL_LIMIT:
                    truncated_messages.append(msg)
                    total_tokens += len(msg["content"])
                else:
                    break
        
        logger.info(f"Final message count: {len(truncated_messages)}, Total tokens: {total_tokens}")
        logger.info(f"Truncated messages: {truncated_messages}")
        return truncated_messages