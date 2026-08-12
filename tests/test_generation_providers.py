from types import SimpleNamespace

import pytest

import ragops.generation.factory as factory_module
from ragops.generation.client import LocalTemplateGenerationClient
from ragops.generation.providers import GeminiGenerationClient, OpenAIGenerationClient


class FakeOpenAIResponses:

    def __init__(self, output_text="OpenAI answer [1]"):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeGeminiInteractions:

    def __init__(self, output_text="Gemini answer [1]"):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


def test_openai_generation_client_uses_responses_api():
    responses = FakeOpenAIResponses()
    sdk_client = SimpleNamespace(responses=responses)
    client = OpenAIGenerationClient(model="test-openai-model", client=sdk_client)

    answer = client.generate("Grounded prompt")

    assert answer == "OpenAI answer [1]"
    assert responses.calls == [{"model": "test-openai-model", "input": "Grounded prompt"}]


def test_gemini_generation_client_uses_interactions_api():
    interactions = FakeGeminiInteractions()
    sdk_client = SimpleNamespace(interactions=interactions)
    client = GeminiGenerationClient(model="test-gemini-model", client=sdk_client)

    answer = client.generate("Grounded prompt")

    assert answer == "Gemini answer [1]"
    assert interactions.calls == [{"model": "test-gemini-model", "input": "Grounded prompt"}]


@pytest.mark.parametrize(
    ("client"),
    [
        OpenAIGenerationClient(client=SimpleNamespace(responses=FakeOpenAIResponses("   "))),
        GeminiGenerationClient(client=SimpleNamespace(interactions=FakeGeminiInteractions("   "))),
    ],
)
def test_provider_clients_reject_empty_responses(client):
    with pytest.raises(RuntimeError, match="empty response"):
        client.generate("Grounded prompt")


def test_factory_defaults_to_template_client(monkeypatch):
    monkeypatch.delenv("RAGOPS_LLM_PROVIDER", raising=False)

    client = factory_module.create_generation_client()

    assert isinstance(client, LocalTemplateGenerationClient)


def test_factory_creates_openai_client_from_environment(monkeypatch):
    calls = {}

    def fake_client(model, api_key):
        calls.update(model=model, api_key=api_key)
        return "openai-client"

    monkeypatch.setenv("RAGOPS_LLM_PROVIDER", " OPENAI ")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", " test-openai-model ")
    monkeypatch.setattr(factory_module, "OpenAIGenerationClient", fake_client)

    client = factory_module.create_generation_client()

    assert client == "openai-client"
    assert calls == {"model": "test-openai-model", "api_key": "test-openai-key"}


def test_factory_creates_gemini_client_from_environment(monkeypatch):
    calls = {}

    def fake_client(model, api_key):
        calls.update(model=model, api_key=api_key)
        return "gemini-client"

    monkeypatch.setenv("RAGOPS_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(factory_module, "GeminiGenerationClient", fake_client)

    client = factory_module.create_generation_client()

    assert client == "gemini-client"
    assert calls == {"model": "gemini-3.6-flash", "api_key": "test-gemini-key"}


@pytest.mark.parametrize(("provider", "key_name"), [("openai", "OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY")])
def test_factory_requires_provider_api_key(monkeypatch, provider, key_name):
    monkeypatch.delenv(key_name, raising=False)

    with pytest.raises(ValueError, match=key_name):
        factory_module.create_generation_client(provider)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported RAGOPS_LLM_PROVIDER"):
        factory_module.create_generation_client("unknown")


def test_factory_explicit_model_overrides_provider_environment(monkeypatch):
    calls = {}

    def fake_client(model, api_key):
        calls.update(model=model, api_key=api_key)
        return "openai-client"

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    monkeypatch.setattr(factory_module, "OpenAIGenerationClient", fake_client)

    client = factory_module.create_generation_client("openai", model="configured-judge-model")

    assert client == "openai-client"
    assert calls == {"model": "configured-judge-model", "api_key": "test-openai-key"}
