from anthropic import Anthropic

from app.core.config import get_settings


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK for sending chat completions."""

    def __init__(self) -> None:
        settings = get_settings()
        # max_retries enables the SDK's built-in retry with exponential
        # backoff for transient failures (connection errors, 408, 429, 5xx).
        # Non-retryable errors (401, 400) are raised immediately.
        self._client = Anthropic(
            api_key=settings.anthropic_api_key,
            max_retries=3,
        )
        self._model = settings.anthropic_model

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Sends a prompt to Claude and returns the text of the response.

        The system prompt carries trusted content (developer instructions
        and retrieved document context); the user prompt carries only the
        untrusted user question. Keeping them in separate API parameters
        preserves Claude's trained instruction hierarchy and improves
        resistance to prompt injection.
        """
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text