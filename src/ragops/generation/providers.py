from ragops.generation.client import GenerationClient

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


class OpenAIGenerationClient(GenerationClient):

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
        response = self.client.responses.create(model=self.model, input=prompt)
        return response_text(response, "OpenAI")


class GeminiGenerationClient(GenerationClient):

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
        response = self.client.interactions.create(model=self.model, input=prompt)
        return response_text(response, "Gemini")
