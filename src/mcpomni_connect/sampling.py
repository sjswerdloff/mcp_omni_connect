async def _select_model(
    self, preferences: Optional[ModelPreferences], available_models: List[str]
) -> str:
    """Select the best model based on preferences and available models."""
    if not preferences or not preferences.hints:
        return available_models[0]  # Default to first available model

    # Try to match hints with available models
    for hint in preferences.hints:
        if not hint.name:
            continue
        for model in available_models:
            if hint.name.lower() in model.lower():
                return model

    # If no match found, use priorities to select model
    if (
        preferences.intelligence_priority
        and preferences.intelligence_priority > 0.7
    ):
        # Prefer more capable models
        return max(available_models, key=lambda x: len(x))  # Simple heuristic
    elif preferences.speed_priority and preferences.speed_priority > 0.7:
        # Prefer faster models
        return min(available_models, key=lambda x: len(x))  # Simple heuristic
    elif preferences.cost_priority and preferences.cost_priority > 0.7:
        # Prefer cheaper models
        return min(available_models, key=lambda x: len(x))  # Simple heuristic

    return available_models[0]  # Default fallback


async def _get_context(
    self, include_context: Optional[ContextInclusion], server_name: str
) -> str:
    """Get relevant context based on inclusion type."""
    if not include_context or include_context == ContextInclusion.NONE:
        return ""

    context_parts = []

    if include_context == ContextInclusion.THIS_SERVER:
        # Get context from specific server
        if server_name in self.sessions:
            session_data = self.sessions[server_name]
            if "message_history" in session_data:
                context_parts.extend(session_data["message_history"])

    elif include_context == ContextInclusion.ALL_SERVERS:
        # Get context from all servers
        for session_data in self.sessions.values():
            if "message_history" in session_data:
                context_parts.extend(session_data["message_history"])

    return "\n".join(context_parts)


async def _sampling_callback(
    self,
    context: RequestContext["ClientSession", Any],
    params: CreateMessageRequestParams,
) -> CreateMessageResult | ErrorData:
    """Enhanced sampling callback with support for advanced features."""
    try:
        logger.debug(f"Sampling callback called with params: {params}")

        # Validate required parameters
        if not params.messages or not isinstance(params.max_tokens, int):
            return ErrorData(
                code="INVALID_REQUEST",
                message="Missing required fields: messages or max_tokens",
            )

        # Get the LLM configuration from the client instance
        llm_config = self.config.get("LLM", {})
        provider = llm_config.get("provider", "openai")

        # Get available models for the provider
        available_models = llm_config.get("available_models", [])
        if not available_models:
            available_models = ["gpt-4", "gpt-3.5-turbo"]  # Default models

        # Select model based on preferences
        model = await self._select_model(
            params.model_preferences, available_models
        )

        # Get context if requested
        server_name = (
            context.client_id if hasattr(context, "client_id") else "default"
        )
        additional_context = await self._get_context(
            params.include_context, server_name
        )

        # Prepare messages with context and system prompt
        messages = []
        if params.system_prompt:
            messages.append(
                {"role": "system", "content": params.system_prompt}
            )
        if additional_context:
            messages.append(
                {"role": "system", "content": f"Context: {additional_context}"}
            )
        messages.extend(
            [
                {"role": msg.role, "content": msg.content.text}
                for msg in params.messages
            ]
        )

        logger.debug(f"Using LLM provider: {provider}, model: {model}")

        # Initialize the appropriate client based on provider
        if provider == "openai":
            client = OpenAI(api_key=os.getenv("LLM_API_KEY"))
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=(
                    params.temperature
                    if params.temperature is not None
                    else 0.5
                ),
                stop=params.stop_sequences if params.stop_sequences else None,
                **params.metadata if params.metadata else {},
            )
            completion = response.choices[0].message.content
            stop_reason = response.choices[0].finish_reason

        elif provider == "groq":
            client = Groq(api_key=os.getenv("LLM_API_KEY"))
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=(
                    params.temperature
                    if params.temperature is not None
                    else 0.5
                ),
                stop=params.stop_sequences if params.stop_sequences else None,
                **params.metadata if params.metadata else {},
            )
            completion = response.choices[0].message.content
            stop_reason = response.choices[0].finish_reason

        elif provider == "openrouter":
            client = OpenAI(
                api_key=os.getenv("LLM_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=(
                    params.temperature
                    if params.temperature is not None
                    else 0.5
                ),
                stop=params.stop_sequences if params.stop_sequences else None,
                **params.metadata if params.metadata else {},
            )
            completion = response.choices[0].message.content
            stop_reason = response.choices[0].finish_reason

        else:
            return ErrorData(
                code="INVALID_REQUEST",
                message=f"Unsupported LLM provider: {provider}",
            )

        # Create the result
        result = CreateMessageResult(
            model=model,
            stop_reason=stop_reason,
            role="assistant",
            content=MessageContent(type=ContentType.TEXT, text=completion),
        )

        # Update message history
        if server_name in self.sessions:
            if "message_history" not in self.sessions[server_name]:
                self.sessions[server_name]["message_history"] = []
            self.sessions[server_name]["message_history"].append(
                f"User: {params.messages[-1].content.text}"
            )
            self.sessions[server_name]["message_history"].append(
                f"Assistant: {completion}"
            )

        logger.debug(f"Sampling callback completed successfully: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in sampling callback: {str(e)}")
        return ErrorData(
            code="INTERNAL_ERROR", message=f"An error occurred: {str(e)}"
        )
