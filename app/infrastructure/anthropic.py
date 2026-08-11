from anthropic import Anthropic

from app.core.config import get_settings


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK for sending chat completions."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def generate_response(self, prompt: str) -> str:
        """Sends a prompt to Claude and returns the text of the response."""
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


if __name__ == "__main__":
    client = ClaudeClient()
    response = client.generate_response("What model are you?")
    print(response)