from ragops.generation.client import GeneratedText, GenerationClient, GenerationUsage

DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def clean_model_name(model, provider):
    """Return a non-empty model name for one provider."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{provider} model must be a non-empty string.")

    return model.strip()


def response_text(response, provider):
    """Extract non-empty output text from a provider response."""
    answer = getattr(response, "output_text", "")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"{provider} returned an empty response.")

    return answer.strip()


def _usage_value(usage, *names):
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return value
        if isinstance(usage, dict) and usage.get(name) is not None:
            return usage[name]
    return None


def response_usage(response):
    """Normalize OpenAI Responses or Gemini Interactions token metadata."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = _usage_value(usage, "input_tokens", "total_input_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "total_output_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return GenerationUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


class OpenAIGenerationClient(GenerationClient):

    provider = "openai"

    def __init__(self, model=DEFAULT_OPENAI_MODEL, api_key=None, client=None):
        self.model = clean_model_name(model, "OpenAI")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("The OpenAI provider requires the 'openai' package.") from error

            client = OpenAI(api_key=api_key)

        self.client = client

    def generate(self, prompt):
        """Generate text with the OpenAI Responses API."""
        return self.generate_with_metadata(prompt).text

    def generate_with_metadata(self, prompt):
        """Generate OpenAI text and retain SDK-reported token usage."""
        response = self.client.responses.create(model=self.model, input=prompt)
        return GeneratedText(text=response_text(response, "OpenAI"), usage=response_usage(response))


class GeminiGenerationClient(GenerationClient):

    provider = "gemini"

    def __init__(self, model=DEFAULT_GEMINI_MODEL, api_key=None, client=None):
        self.model = clean_model_name(model, "Gemini")

        if client is None:
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError("The Gemini provider requires the 'google-genai' package.") from error

            client = genai.Client(api_key=api_key)

        self.client = client

    def generate(self, prompt):
        """Generate text with the Gemini Interactions API."""
        return self.generate_with_metadata(prompt).text

    def generate_with_metadata(self, prompt):
        """Generate Gemini text and retain SDK-reported token usage."""
        response = self.client.interactions.create(model=self.model, input=prompt)
        return GeneratedText(text=response_text(response, "Gemini"), usage=response_usage(response))
