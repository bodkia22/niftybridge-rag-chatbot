from anthropic import Anthropic

from app.core.config import get_settings
from app.schemas.chat import StructuredAnswer


RECORD_ANSWER_TOOL = {
    "name": "record_answer",
    "description": "Records the answer to the user's question along with the excerpts actually used.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the user's question, in the same language as the question.",
            },
            "used_sources": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "1-based numbers of the excerpts (from the numbered list in the "
                    "system prompt) that were actually used to produce the answer. "
                    "Return an empty array if the answer does not rely on any excerpt "
                    "(e.g. greetings, small talk, or questions unrelated to the document)."
                ),
            },
        },
        "required": ["answer", "used_sources"],
    },
}


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

    def generate_structured_answer(self, system_prompt: str, user_prompt: str) -> StructuredAnswer:
        """Sends a prompt to Claude and returns a structured answer.

        The system prompt carries trusted content (developer instructions
        and retrieved document context); the user prompt carries only the
        untrusted user question. Keeping them in separate API parameters
        preserves Claude's trained instruction hierarchy and improves
        resistance to prompt injection.

        `tool_choice` forces Claude to always respond via the record_answer
        tool, so the response always contains exactly one tool_use block —
        no need to handle a "Claude answered in plain text" branch.

        Raises pydantic.ValidationError if Claude's tool input doesn't match
        the expected shape (should be rare, but not impossible).
        """
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[RECORD_ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "record_answer"},
        )

        tool_use_block = next(block for block in message.content if block.type == "tool_use")
        return StructuredAnswer(**tool_use_block.input)

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
            """Sends a plain-text prompt to Claude, without forcing tool use.

            Used as a fallback when structured output parsing fails, so a
            malformed tool call doesn't take down the whole request.
            """
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text