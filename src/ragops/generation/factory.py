import os

from ragops.generation.client import LocalTemplateGenerationClient
from ragops.generation.providers import DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL, GeminiGenerationClient, OpenAIGenerationClient

DEFAULT_GENERATION_PROVIDER = "template"
SUPPORTED_GENERATION_PROVIDERS = ("template", "openai", "gemini")


def configured_value(name, default=None):
    """Return one stripped environment value or its default."""
    value = os.getenv(name, default)

    if isinstance(value, str):
        return value.strip()

    return value


def required_api_key(name, provider):
    """Return a configured API key with a clear provider-specific error."""
    api_key = configured_value(name)

    if not api_key:
        raise ValueError(f"{name} must be set when RAGOPS_LLM_PROVIDER={provider}.")

    return api_key


def create_generation_client(provider=None, model=None):
    """Create the generation client selected by argument or environment."""
    provider = provider if provider is not None else configured_value("RAGOPS_LLM_PROVIDER", DEFAULT_GENERATION_PROVIDER)
    provider = provider.strip().lower() if isinstance(provider, str) else ""

    if not provider:
        provider = DEFAULT_GENERATION_PROVIDER

    if provider == "template":
        return LocalTemplateGenerationClient()

    if provider == "openai":
        selected_model = model if model is not None else configured_value("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        selected_model = selected_model or DEFAULT_OPENAI_MODEL
        return OpenAIGenerationClient(model=selected_model, api_key=required_api_key("OPENAI_API_KEY", provider))

    if provider == "gemini":
        selected_model = model if model is not None else configured_value("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        selected_model = selected_model or DEFAULT_GEMINI_MODEL
        return GeminiGenerationClient(model=selected_model, api_key=required_api_key("GEMINI_API_KEY", provider))

    supported = ", ".join(SUPPORTED_GENERATION_PROVIDERS)
    raise ValueError(f"Unsupported RAGOPS_LLM_PROVIDER '{provider}'. Choose one of: {supported}.")
